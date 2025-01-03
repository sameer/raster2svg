use cairo::{Context, Matrix, SvgUnit};
use clap::{Parser, ValueEnum};
use dither::{Dither, FloydSteinberg};
use image::ImageReader;
#[cfg(debug_assertions)]
use image::{Rgb, RgbImage};
use log::*;
use ndarray::{prelude::*, SliceInfo, SliceInfoElem};
use serde::{Deserialize, Serialize};
use std::{
    env,
    fmt::Debug,
    fs::File,
    io::{self, Read},
    path::PathBuf,
    str::FromStr,
    vec,
};
use uom::si::f64::Length;
use uom::si::length::{inch, millimeter};

use crate::color::Color;
use crate::render::render_stipple_based;

/// Adjust image color
mod color;
/// Dither an image given a predefined set of colors
mod dither;
/// Image filter algorithms (i.e. Sobel operator, FDoG, ETF)
mod filter;
/// Graph algorithms
mod graph;
/// Line segment drawing and related algorithms
mod lsd;
/// Pure math routines
mod math;
/// Optimization algorithms
mod optimize;
/// Routines for creating the final SVG using [Cairo](cairographics.org)
mod render;
/// Construct the [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram) and calculate related properties
mod voronoi;

#[derive(Parser, Debug, Deserialize, Serialize)]
#[command(author, about)]
struct Opt {
    /// A path to an image, else reads from stdin
    file: Option<PathBuf>,

    #[arg(long)]
    config: Option<PathBuf>,

    /// Determines the scaling of the output SVG
    #[arg(long, alias = "dpi", default_value = "96.")]
    dots_per_inch: f64,

    /// Color model to use for additive coloring
    #[arg(long, default_value = "cielab", ignore_case = true)]
    #[serde(with = "serde_with::rust::display_fromstr")]
    color_model: ColorModel,

    /// Coloring method to use
    #[arg(long, default_value = "vector", ignore_case = true)]
    #[serde(with = "serde_with::rust::display_fromstr")]
    color_method: ColorMethod,

    /// SVG drawing style
    #[arg(long, default_value = "mst", ignore_case = true)]
    #[serde(with = "serde_with::rust::display_fromstr")]
    style: Style,

    /// The drawing implement(s) used by your plotter
    #[arg(long = "implement", ignore_case = true)]
    implements: Vec<Implement>,

    /// Super-sampling factor for finer detail control
    #[arg(long, default_value = "1")]
    super_sample: usize,

    /// Output file path (overwrites old files), else writes to stdout
    #[arg(short, long)]
    out: Option<PathBuf>,
}

macro_rules! opt {
    ($name: ident {
        $(
            $variant: ident $({
                $($member: ident; $ty: ty,)*
            })?,
        )*
    }) => {
        #[derive(ValueEnum, Debug, Clone, Copy)]
        pub enum $name {
            $(
                $variant $({
                    $($member: $ty,)*
                })?,
            )*
        }

        paste::paste! {
            // impl $name {
            //     const NUM_VARIANTS: usize = 0 $(
            //         + { let _  = stringify!(Self::$variant); 1 }
            //     )*;

            //     fn raw_variants() -> [&'static str; Self::NUM_VARIANTS] {
            //         [
            //             $(
            //                 stringify!([<$variant:snake>]),
            //             )*
            //         ]
            //     }
            // }

            impl std::fmt::Display for $name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    match self {
                        $(
                            Self::$variant => f.write_str(stringify!([<$variant:snake>])),
                        )*
                    }
                }
            }

            impl FromStr for $name {
                type Err = &'static str;

                fn from_str(data: &str) -> Result<Self, Self::Err> {
                    $(
                        if stringify!([<$variant:snake>]) == data {
                            return Ok(Self::$variant);
                        }
                    )*

                    return Err("Must be one of variants");
                }
            }
        }
    };
}

opt! {
    ColorModel {
        Cielab,
        Rgb,
    }
}

opt! {
    ColorMethod {
        Dither,
        Vector,
    }
}

opt! {
    Style {
        Stipple,
        EdgesPlusHatching,
        Tsp,
        Mst,
        Triangulation,
        Voronoi,
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum Implement {
    Pen {
        diameter: Length,
        #[serde(with = "serde_with::rust::display_fromstr")]
        color: Color,
    },
    Pencil,
    Marker,
}

impl FromStr for Implement {
    type Err = serde_json::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

fn main() -> io::Result<()> {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "raster2svg=info")
    }
    env_logger::init();

    let ref opt @ Opt {
        ref file,
        config: _,
        dots_per_inch: _,
        ref color_model,
        ref color_method,
        ref style,
        ref implements,
        super_sample: _,
        out: _,
    } = {
        let mut opt = Opt::parse();

        if let Some(config) = opt.config {
            let mut config = serde_json::from_reader::<_, Opt>(File::open(&config)?)?;
            config.file = opt.file.or(config.file);
            config.out = opt.out.or(config.out);
            config.implements.append(&mut opt.implements);
            config
        } else {
            opt
        }
    };

    let image = match file {
        Some(ref filepath) => ImageReader::open(filepath)?.decode(),
        None => {
            info!("Reading from stdin");
            let mut bytes = vec![];
            io::stdin().read_to_end(&mut bytes)?;
            let cursor = io::Cursor::new(bytes);
            ImageReader::new(cursor).decode()
        }
    }
    .expect("not an image")
    .to_rgb16();

    let image = Array::from_iter(
        image
            .pixels()
            .flat_map(|p| p.0)
            .map(|p| p as f64 / u16::MAX as f64),
    )
    .into_shape((image.height() as usize, image.width() as usize, 3))
    .unwrap()
    .reversed_axes();

    let palette = implements
        .iter()
        .map(|implement| {
            if let Implement::Pen { color, .. } = implement {
                *color
            } else {
                unimplemented!()
            }
        })
        .collect::<Vec<_>>();

    let mut image_in_implements: Array3<f64>;
    match color_method {
        ColorMethod::Dither => {
            let (_, width, height) = image.dim();
            let image_in_color_model = color_model.convert(image.view());
            let implements_in_color_model = palette
                .iter()
                .map(|c| color_model.convert_single(c))
                .collect::<Vec<_>>();
            image_in_implements = Array3::<f64>::zeros((palette.len(), width, height));
            if !matches!(color_model, ColorModel::Rgb) {
                warn!("Non-rgb color model + dither doesn't work well");
            }
            // Add white for background
            let colors_float = implements_in_color_model
                .into_iter()
                .chain(std::iter::once(
                    color_model.convert_single(&Color::from([1.; 3])),
                ))
                .collect::<Vec<_>>();
            let dithered = FloydSteinberg.dither(image_in_color_model.view(), &colors_float);
            #[cfg(debug_assertions)]
            let mut buf = RgbImage::new(width as u32, height as u32);
            for y in 0..height {
                for x in 0..width {
                    let k = dithered[[x, y]];
                    if k == implements.len() {
                        #[cfg(debug_assertions)]
                        buf.put_pixel(x as u32, y as u32, Rgb([255; 3]));
                        continue;
                    }
                    #[cfg(debug_assertions)]
                    buf.put_pixel(x as u32, y as u32, Rgb((palette[k]).into()));
                    image_in_implements[[k, x, y]] = 1.0;
                }
            }

            #[cfg(debug_assertions)]
            buf.save("x.png").unwrap();
        }
        ColorMethod::Vector => {
            image_in_implements = color_model.approximate(image.view(), &palette);
        }
    }

    // Linearize color mapping for line drawings
    if matches!(style, Style::Tsp | Style::Mst) {
        image_in_implements.mapv_inplace(|v| v.powi(2));
    } else if matches!(style, Style::Triangulation | Style::Voronoi) {
        image_in_implements.mapv_inplace(|v| v.powi(3));
    }

    draw(image_in_implements.view(), &opt);
    Ok(())
}

fn draw(image_in_implements: ArrayView3<f64>, opt: &Opt) {
    let mm_per_inch = Length::new::<inch>(1.).get::<millimeter>();
    let dots_per_mm = opt.dots_per_inch / mm_per_inch;

    let width =
        (image_in_implements.raw_dim()[1] as f64 / dots_per_mm / opt.super_sample as f64).round();
    let height =
        (image_in_implements.raw_dim()[2] as f64 / dots_per_mm / opt.super_sample as f64).round();

    let mut surf = match &opt.out {
        Some(_) => cairo::SvgSurface::new(width, height, opt.out.as_ref()).unwrap(),
        None => cairo::SvgSurface::for_stream(width, height, std::io::stdout()).unwrap(),
    };
    surf.set_document_unit(SvgUnit::Mm);
    let ctx = Context::new(&surf).unwrap();

    ctx.set_source_rgb(1., 1., 1.);
    ctx.rectangle(0., 0., width, height);
    ctx.fill().unwrap();

    match opt.style {
        Style::Stipple | Style::Tsp | Style::Mst | Style::Triangulation | Style::Voronoi => {
            render_stipple_based(
                image_in_implements.view(),
                &opt.implements
                    .iter()
                    .map(|implement| {
                        if let Implement::Pen { diameter, .. } = implement {
                            diameter.get::<millimeter>() * dots_per_mm
                        } else {
                            todo!()
                        }
                    })
                    .collect::<Vec<_>>(),
                &opt.implements
                    .iter()
                    .map(|implement| {
                        if let Implement::Pen { color, .. } = implement {
                            *color
                        } else {
                            todo!()
                        }
                    })
                    .collect::<Vec<_>>(),
                opt.super_sample,
                opt.style,
                &ctx,
                {
                    let mut mat = Matrix::identity();
                    mat.scale(
                        1.0 / dots_per_mm / opt.super_sample as f64,
                        1.0 / dots_per_mm / opt.super_sample as f64,
                    );
                    mat
                },
            )
        }
        Style::EdgesPlusHatching => {
            todo!()
            //             render_fdog_based(
            //                 image_in_implements.slice(s![k, .., ..]),
            //                 opt.super_sample,
            //                 implement_diameter_in_pixels,
            //                 opt.style,
            //                 &ctx,
            //                 {
            //                     let mut mat = Matrix::identity();
            //                     mat.scale(
            //                         1.0 / dots_per_mm / opt.super_sample as f64,
            //                         1.0 / dots_per_mm / opt.super_sample as f64,
            //                     );
            //                     mat
            //                 },
            //             )
        }
    }
}

/// Utility function for applying windowed offset functions like convolution on a 2D ndarray array
#[inline]
pub(crate) fn get_slice_info_for_offset(
    x: i32,
    y: i32,
) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> {
    match (x.signum(), y.signum()) {
        (-1, -1) => s![..x, ..y],
        (0, -1) => s![.., ..y],
        (-1, 0) => s![..x, ..],
        (0, 0) => s![.., ..],
        (1, 0) => s![x.., ..],
        (0, 1) => s![.., y..],
        (-1, 1) => s![..x, y..],
        (1, -1) => s![x.., ..y],
        (1, 1) => s![x.., y..],
        _ => unreachable!(),
    }
}

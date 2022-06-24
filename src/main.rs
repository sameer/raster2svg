use crate::{color::Color, direct::Direct};
use cairo::{Context, Matrix, SvgUnit};
use dither::{Dither, FloydSteinberg};
use image::io::Reader as ImageReader;
#[cfg(debug)]
use image::{Rgb, RgbImage};
use log::*;
use lyon_geom::{
    euclid::default::{Vector2D, Vector3D},
    Angle,
};
use ndarray::{prelude::*, SliceInfo, SliceInfoElem};
use num_traits::{PrimInt, Signed};
use rustc_hash::FxHashMap;
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
use structopt::StructOpt;
use uom::si::f64::Length;
use uom::si::length::{inch, millimeter};

use crate::render::render_stipple_based;

/// Adjust image color
mod color;
mod direct;
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
/// Routines for creating the final SVG using [Cairo](cairographics.org)
mod render;
/// Construct the [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram) and calculate related properties
mod voronoi;

#[derive(Debug, StructOpt, Deserialize, Serialize)]
#[structopt(author, about)]
struct Opt {
    /// A path to an image, else reads from stdin
    file: Option<PathBuf>,

    #[structopt(long)]
    config: Option<PathBuf>,

    /// Determines the scaling of the output SVG
    #[structopt(long, default_value = "96")]
    dots_per_inch: f64,

    /// Color model to use for additive coloring
    #[structopt(
        long,
        default_value = "cielab",
        possible_values = &ColorModel::raw_variants(),
        case_insensitive = true
    )]
    #[serde(with = "serde_with::rust::display_fromstr")]
    color_model: ColorModel,

    /// Coloring method to use
    #[structopt(
        long,
        default_value = "vector",
        possible_values = &ColorMethod::raw_variants(),
        case_insensitive = true
    )]
    #[serde(with = "serde_with::rust::display_fromstr")]
    color_method: ColorMethod,

    /// SVG drawing style
    #[structopt(
        long,
        default_value = "mst",
        possible_values = &Style::raw_variants(),
        case_insensitive = true
    )]
    #[serde(with = "serde_with::rust::display_fromstr")]
    style: Style,

    /// The drawing implement(s) used by your plotter
    #[structopt(long = "implement", case_insensitive = true)]
    implements: Vec<Implement>,

    /// Super-sampling factor for finer detail control
    #[structopt(long, default_value = "1")]
    super_sample: usize,

    /// Output file path (overwrites old files), else writes to stdout
    #[structopt(short, long)]
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
        #[derive(Debug, Clone, Copy)]
        pub enum $name {
            $(
                $variant $({
                    $($member: $ty,)*
                })?,
            )*
        }

        paste::paste! {
            impl $name {
                const NUM_VARIANTS: usize = 0 $(
                    + { let _  = stringify!(Self::$variant); 1 }
                )*;

                fn raw_variants() -> [&'static str; Self::NUM_VARIANTS] {
                    [
                        $(
                            stringify!([<$variant:snake>]),
                        )*
                    ]
                }
            }

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

#[derive(Debug, Deserialize, Serialize)]
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
    let mut opt = Opt::from_args();

    if let Some(config) = opt.config {
        let mut config = serde_json::from_reader::<_, Opt>(File::open(&config)?)?;
        config.file = opt.file.or(config.file);
        config.out = opt.out.or(config.out);
        config.implements.append(&mut opt.implements);
        opt = config;
    }

    let image = match opt.file {
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

    let colors = opt
        .implements
        .iter()
        .map(|implement| {
            if let Implement::Pen { color, .. } = implement {
                color
            } else {
                unimplemented!()
            }
        })
        .collect::<Vec<_>>();

    let image_in_color_model = opt.color_model.convert(image.view());
    let implements_in_color_model = colors
        .iter()
        .map(|c| opt.color_model.convert_single(c))
        .collect::<Vec<_>>();
    drop(image);

    let (_, width, height) = image_in_color_model.dim();

    let mut image_in_implements = Array3::<f64>::zeros((opt.implements.len(), width, height));
    let implements = opt.implements.len();
    match opt.color_method {
        ColorMethod::Dither => {
            if !matches!(opt.color_model, ColorModel::Rgb) {
                warn!("Non-rgb color model + dither doesn't work well");
            }
            // Add white for background
            let colors_float = implements_in_color_model
                .into_iter()
                .chain(std::iter::once(
                    opt.color_model.convert_single(&Color::from([1.; 3])),
                ))
                .collect::<Vec<_>>();
            let dithered = FloydSteinberg.dither(image_in_color_model.view(), &colors_float);
            #[cfg(debug)]
            let mut buf = RgbImage::new(width as u32, height as u32);
            for y in 0..height {
                for x in 0..width {
                    let k = dithered[[x, y]];
                    if k == opt.implements.len() {
                        #[cfg(debug)]
                        buf.put_pixel(x as u32, y as u32, Rgb([255; 3]));
                        continue;
                    }
                    #[cfg(debug)]
                    buf.put_pixel(x as u32, y as u32, Rgb((*colors[k]).into()));
                    image_in_implements[[k, x, y]] = 1.0;
                }
            }

            #[cfg(debug)]
            buf.save("x.png");
        }
        ColorMethod::Vector => {
            let mut image_in_cylindrical_color_model =
                opt.color_model.cylindrical(image_in_color_model.view());
            let mut implements_in_cylindrical_color_model = implements_in_color_model
                .into_iter()
                .map(|c| opt.color_model.cylindrical_single(c))
                .collect::<Vec<_>>();
            drop(image_in_color_model);

            let white_in_cylindrical_color_model = opt
                .color_model
                .cylindrical_single(opt.color_model.convert_single(&Color::from([1.; 3])));
            image_in_cylindrical_color_model
                .slice_mut(s![2, .., ..])
                .mapv_inplace(|lightness| white_in_cylindrical_color_model[2] - lightness);
            implements_in_cylindrical_color_model
                .iter_mut()
                .for_each(|[_, _, lightness]| {
                    *lightness = white_in_cylindrical_color_model[2] - *lightness
                });

            let implement_hue_vectors = implements_in_cylindrical_color_model
                .into_iter()
                .map(|[hue, magnitude, darkness]| {
                    let (sin, cos) = hue.sin_cos();
                    Vector3D::new(cos * magnitude, sin * magnitude, darkness)
                })
                .collect::<Vec<_>>();

            let mut cached_colors = FxHashMap::default();
            for y in 0..height {
                dbg!(y);
                for x in 0..width {
                    let desired: [f64; 3] = image_in_cylindrical_color_model
                        .slice(s![.., x, y])
                        .to_vec()
                        .try_into()
                        .unwrap();
                    let best = cached_colors
                        .entry([
                            desired[0].to_bits(),
                            desired[1].to_bits(),
                            desired[2].to_bits(),
                        ])
                        .or_insert_with(|| {
                            // Find an optimal linear combination of the palette

                            let max_angle = match opt.color_model {
                                ColorModel::Cielab => std::f64::consts::FRAC_PI_2,
                                ColorModel::Rgb => std::f64::consts::FRAC_PI_3 * 2.,
                            };

                            // (0..implement_hue_vectors.len()).collect::<Vec<_>>();
                            let allowed_hue_indices = implement_hue_vectors
                                .iter()
                                .enumerate()
                                .filter(|(_, p)| {
                                    let chroma = p.to_2d();
                                    let (sin, cos) = desired[0].sin_cos();
                                    let desired_chroma =
                                        Vector2D::new(cos * desired[1], sin * desired[1]);
                                    let zero = Vector2D::zero();

                                    chroma == zero
                                        || desired_chroma == zero
                                        || chroma.angle_to(desired_chroma).radians.abs()
                                            <= max_angle
                                })
                                .map(|(i, _)| i)
                                .collect::<Vec<_>>();
                            if allowed_hue_indices.is_empty() {
                                return Array1::zeros(implements);
                            }

                            let direct = Direct {
                                epsilon: 1E-4,
                                max_evaluations: Some(10000),
                                max_iterations: None,
                                initial: Array::zeros(allowed_hue_indices.len()),
                                bounds: Array::from_elem(allowed_hue_indices.len(), [0., 1.]),
                                function: |param: ArrayView1<f64>| {
                                    let weighted_vector = allowed_hue_indices
                                        .iter()
                                        .map(|i| implement_hue_vectors[*i])
                                        .zip(param.iter())
                                        .map(|(p, x)| p * *x)
                                        .sum::<Vector3D<f64>>();
                                    // Convert back to cylindrical model (hue, chroma, darkness)
                                    let actual = [
                                        weighted_vector.y.atan2(weighted_vector.x),
                                        weighted_vector.to_2d().length(),
                                        weighted_vector.z,
                                    ];
                                    opt.color_model.cylindrical_diff(actual, desired)
                                    //+ (actual[2] - desired[2]).abs()
                                },
                            };

                            let (best, best_cost) = direct.run();
                            let mut amended_best = Array::zeros(implements);
                            for (i, val) in allowed_hue_indices.into_iter().zip(best.iter()) {
                                amended_best[i] = *val;
                            }
                            amended_best
                        });

                    image_in_implements
                        .slice_mut(s![.., x, y])
                        .assign(&best.view());
                }
            }
        }
    }

    // Linearize color mapping for line drawings
    if matches!(opt.style, Style::Tsp | Style::Mst) {
        image_in_implements.mapv_inplace(|v| v.powi(2));
    } else if matches!(opt.style, Style::Triangulation | Style::Voronoi) {
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

/// Square of the Euclidean distance between signed n-dimension coordinates
#[inline]
fn abs_distance_squared<T: PrimInt + Signed + Debug, const N: usize>(a: [T; N], b: [T; N]) -> T {
    let mut acc = T::zero();
    for i in 0..N {
        acc = acc + (a[i] - b[i]).pow(2);
    }
    acc
}

/// Utility function for applying windowed offset functions like convolution on a 2D ndarray array
#[inline]
pub(crate) fn get_slice_info_for_offset(
    x: i32,
    y: i32,
) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> {
    match (x.signum(), y.signum()) {
        (-1, -1) => (s![..x, ..y]),
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

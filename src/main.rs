use cairo::{Context, LineCap, LineJoin, Matrix, SvgUnit};
use color::{ciexyz_to_cielab, srgb_to_ciexyz, srgb_to_hsl};
use image::io::Reader as ImageReader;
use log::*;
use lyon_geom::vector;
use ndarray::{prelude::*, SliceInfo, SliceInfoElem};
use num_traits::{PrimInt, Signed};
use std::{
    env,
    fmt::Debug,
    io::{self, Read},
    path::PathBuf,
    str::FromStr,
    vec,
};
use structopt::StructOpt;
use uom::si::f64::Length;
use uom::si::length::{inch, millimeter};

use crate::render::{render_fdog_based, render_stipple_based};

/// Adjust image color
mod color;
/// Image filter algorithms (i.e. Sobel operator, FDoG, ETF)
mod filter;
/// Graph algorithms
mod graph;
/// Line segment drawing and related algorithms
mod lsd;
/// Routines for creating the final SVG using [Cairo](cairographics.org)
mod render;
/// Construct the [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram) and calculate related properties
mod voronoi;

#[derive(Debug, StructOpt)]
#[structopt(author, about)]
struct Opt {
    /// A path to an image, else reads from stdin
    file: Option<PathBuf>,

    /// Determines the scaling of the output SVG
    #[structopt(long, default_value = "96")]
    dots_per_inch: f64,

    /// Diameter of the instrument stroke in units of your choice
    #[structopt(long, default_value = "1 mm")]
    diameter: Length,

    /// Color model to use for additive coloring
    #[structopt(
        long,
        default_value = "cielab",
        possible_values = &ColorModel::raw_variants(),
        case_insensitive = true
    )]
    color_model: ColorModel,

    /// SVG drawing style
    #[structopt(
        long,
        default_value = "mst",
        possible_values = &Style::raw_variants(),
        case_insensitive = true
    )]
    style: Style,

    /// The drawing implement used by your plotter
    #[structopt(
        long,
        default_value = "pen",
        possible_values = &Implement::raw_variants(),
        case_insensitive = true
    )]
    implement: Implement,

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
            $variant: ident,
        )*
    }) => {
        #[derive(Debug, Clone, Copy)]
        pub enum $name {
            $(
                $variant,
            )*
        }

        paste::paste! {
            impl $name {
                const NUM_VARIANTS: usize = 0 $(
                    + if let _ = $name::$variant { 1 } else { 0 }
                )*;

                const fn raw_variants() -> [&'static str; Self::NUM_VARIANTS] {
                    [$(
                        stringify!([<$variant:snake>]),
                    )*]
                }

                const fn variants() -> [$name; Self::NUM_VARIANTS] {
                    use $name::*;
                    [$(
                        [<$variant>],
                    )*]
                }
            }
            impl FromStr for $name {
                type Err = &'static str;

                fn from_str(s: &str) -> Result<Self, Self::Err> {
                    use $name::*;
                    match s {
                        $(
                            stringify!([<$variant:snake>]) => Ok([<$variant>]),)*
                        _ => Err(concat!(stringify!($name), " must be one of the following: ", $(stringify!($variant),)*)),
                    }
                }
            }
        }
    };
}

opt! {
    ColorModel {
        Hsl,
        Cielab,
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

opt! {
    Implement {
        Pen,
        Pencil,
        Marker,
    }
}

fn main() -> io::Result<()> {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "raster2svg=info")
    }
    env_logger::init();
    let opt = Opt::from_args();

    if !matches!(opt.implement, Implement::Pen) {
        unimplemented!("instrument not implemented yet")
    }

    let pen_image = {
        let image = match opt.file {
            Some(filepath) => ImageReader::open(filepath)?.decode(),
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
                .map(|p| p.0.iter().copied())
                .flatten()
                .map(|p| p as f64 / u16::MAX as f64),
        )
        .into_shape((image.height() as usize, image.width() as usize, 3))
        .unwrap()
        .reversed_axes();

        let image_in_cielab = ciexyz_to_cielab(srgb_to_ciexyz(image.view()).view());
        let mut pen_image: Array3<f64> = Array3::zeros((
            4,
            image_in_cielab.raw_dim()[1],
            image_in_cielab.raw_dim()[2],
        ));

        // Key
        {
            pen_image
                .slice_mut(s![3, .., ..])
                .assign(&image_in_cielab.slice(s![0, .., ..]));
            pen_image
                .slice_mut(s![3, .., ..])
                .par_mapv_inplace(|v| 1.0 - v);
        }
        // RGB
        match opt.color_model {
            ColorModel::Cielab => [
                vector(80.81351675261305, 69.88458436386973),
                vector(-79.28626260990568, 80.98938522093422),
                vector(68.29938535880123, -112.03112368261236),
            ]
            .iter()
            .enumerate()
            .for_each(|(i, color)| {
                pen_image
                    .slice_mut(s![i, .., ..])
                    .assign(&image_in_cielab.map_axis(Axis(0), |lab| {
                        let hue = vector(lab[1], lab[2]);
                        let angle = hue.angle_to(*color);
                        if angle.radians.abs() > std::f64::consts::FRAC_PI_2 {
                            0.0
                        } else {
                            hue.project_onto_vector(*color).length() / color.length()
                        }
                    }));
            }),
            ColorModel::Hsl => {
                let image_in_hsl = srgb_to_hsl(image.view());
                [
                    vector(1.0, 0.0),
                    vector(-0.5, 3.0f64.sqrt() / 2.0),
                    vector(-0.5, -(3.0f64.sqrt()) / 2.0),
                ]
                .iter()
                .enumerate()
                .for_each(|(i, color)| {
                    pen_image
                        .slice_mut(s![i, .., ..])
                        .assign(&image_in_hsl.map_axis(Axis(0), |hsl| {
                            let (sin, cos) = hsl[0].sin_cos();
                            let hue_vector = vector(cos, sin) * hsl[1];
                            if hue_vector.angle_to(*color).radians.abs()
                                > std::f64::consts::FRAC_PI_2
                            {
                                0.0
                            } else {
                                hue_vector.project_onto_vector(*color).length() * hsl[2]
                            }
                        }));
                })
            }
        }

        // Linearize color mapping for line drawings
        if matches!(opt.style, Style::Tsp | Style::Mst) {
            pen_image.mapv_inplace(|v| v.powi(2));
        } else if matches!(opt.style, Style::Triangulation | Style::Voronoi) {
            pen_image.mapv_inplace(|v| v.powi(3));
        }

        pen_image
    };

    let mm_per_inch = Length::new::<inch>(1.).get::<millimeter>();
    let dots_per_mm = opt.dots_per_inch / mm_per_inch;
    let instrument_diameter = opt.diameter.get::<millimeter>();
    let instrument_diameter_in_pixels = instrument_diameter * dots_per_mm;

    let width = (pen_image.raw_dim()[1] as f64 / dots_per_mm / opt.super_sample as f64).round();
    let height = (pen_image.raw_dim()[2] as f64 / dots_per_mm / opt.super_sample as f64).round();

    let mut surf = match &opt.out {
        Some(_) => cairo::SvgSurface::new(width, height, opt.out).unwrap(),
        None => cairo::SvgSurface::for_stream(width, height, std::io::stdout()).unwrap(),
    };
    surf.set_document_unit(SvgUnit::Mm);
    let ctx = Context::new(&surf).unwrap();

    ctx.set_source_rgb(1., 1., 1.);
    ctx.rectangle(0., 0., width, height);
    ctx.fill();

    ctx.set_line_cap(LineCap::Round);
    ctx.set_line_join(LineJoin::Round);
    ctx.set_line_width(instrument_diameter);

    for k in 0..=3 {
        info!("Processing {}", k);
        if k == 0 {
            ctx.set_source_rgb(1., 0., 0.);
        } else if k == 1 {
            ctx.set_source_rgb(0., 1., 0.);
        } else if k == 2 {
            ctx.set_source_rgb(0., 0., 1.);
        } else if k == 3 {
            ctx.set_source_rgb(0., 0., 0.);
        }

        match opt.style {
            Style::EdgesPlusHatching => {
                if k == 3 {
                    render_fdog_based(
                        pen_image.slice(s![k, .., ..]),
                        opt.super_sample,
                        instrument_diameter_in_pixels,
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
            }
            _ => render_stipple_based(
                pen_image.slice(s![k, .., ..]),
                opt.super_sample,
                instrument_diameter_in_pixels,
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
            ),
        }
    }
    Ok(())
}

/// Square of the Euclidean distance between signed 2D coordinates
#[inline]
fn abs_distance_squared<T: PrimInt + Signed + Debug>(a: [T; 2], b: [T; 2]) -> T {
    let x_diff = a[0] - b[0];
    let y_diff = a[1] - b[1];
    debug_assert!(
        x_diff.pow(2).checked_add(&y_diff.pow(2)).is_some(),
        "x_diff = {:?}, y_diff = {:?}",
        x_diff,
        y_diff
    );
    x_diff.pow(2) + y_diff.pow(2)
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

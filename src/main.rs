use crate::color::{a_to_nd, ciexyz_to_cielab, nd_to_a, srgb_to_ciexyz, srgb_to_hsl, Color};
use cairo::{Context, LineCap, LineJoin, Matrix, SvgUnit};
use image::io::Reader as ImageReader;
use log::*;
use lyon_geom::{vector, Vector};
use ndarray::{prelude::*, SliceInfo, SliceInfoElem};
use num_traits::{PrimInt, Signed};
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

use crate::render::{render_fdog_based, render_stipple_based};

/// Adjust image color
mod color;
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
    // Cyan #158fd4
    // Purple #382375
    // Bold blue: #222c8f
    // Bold green: #517b23
    // Magenta: #d62e69
    // Bold red: #dc12fa
    // Blue: #2547b4
    // Green: #20835f
    // Red: #c72537
    // Black: #312d2f or #000000
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

// Cyan #158fd4
// Purple #382375
// Black: #312d2f or #000000
// Bold blue: #222c8f
// Bold green: #517b23
// Magenta: #d62e69
// Bold red: #dc12fa
// Blue: #2547b4
// Green: #20835f
// Red: #c72537
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
            .flat_map(|p| p.0)
            .map(|p| p as f64 / u16::MAX as f64),
    )
    .into_shape((image.height() as usize, image.width() as usize, 3))
    .unwrap()
    .reversed_axes();

    let image_in_cielab = ciexyz_to_cielab(srgb_to_ciexyz(image.view()).view());
    let image_in_hsl = srgb_to_hsl(image.view());

    let mut image_in_implements = Array3::<f64>::zeros((
        opt.implements.len(),
        image_in_cielab.raw_dim()[1],
        image_in_cielab.raw_dim()[2],
    ));

    let mm_per_inch = Length::new::<inch>(1.).get::<millimeter>();
    let dots_per_mm = opt.dots_per_inch / mm_per_inch;

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

    match opt.color_model {
        ColorModel::Hsl => {
            let (_, width, height) = image_in_hsl.dim();
            let colors_in_hsl = colors
                .iter()
                .map(|c| nd_to_a::<3>(srgb_to_hsl(a_to_nd(c.as_ref()).view())))
                .collect::<Vec<_>>();
            let color_hue_vectors = colors_in_hsl
                .iter()
                .map(|color_hsl| {
                    let (sin, cos) = color_hsl[0].sin_cos();
                    vector(cos, sin) * color_hsl[1]
                })
                .collect::<Vec<_>>();
            for x in 0..width {
                for y in 0..height {
                    let hsl = image_in_hsl.slice(s![.., x, y]);
                    let (sin, cos) = hsl[0].sin_cos();
                    // Direction of hue with magnitude of saturation
                    let image_hue_vector: Vector<_> = vector(cos, sin) * hsl[1];

                    let left = color_hue_vectors
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| {
                            !colors[*i]
                                .as_ref()
                                .iter()
                                .all(|channel| *channel < f64::EPSILON)
                        })
                        .filter(|(_, color_hue_vector)| {
                            let angle = image_hue_vector.angle_to(**color_hue_vector).radians;
                            angle.is_sign_negative() && angle.abs() < std::f64::consts::FRAC_PI_2
                        })
                        .max_by(|(_, color_hue_vector_i), (_, color_hue_vector_j)| {
                            image_hue_vector
                                .project_onto_vector(**color_hue_vector_i)
                                .length()
                                .partial_cmp(
                                    &image_hue_vector
                                        .project_onto_vector(**color_hue_vector_j)
                                        .length(),
                                )
                                .unwrap()
                        });

                    let right = color_hue_vectors
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| {
                            !colors[*i]
                                .as_ref()
                                .iter()
                                .all(|channel| *channel < f64::EPSILON)
                        })
                        .filter(|(_, color_hue_vector)| {
                            let angle = image_hue_vector.angle_to(**color_hue_vector).radians;
                            angle.is_sign_positive() && angle.abs() < std::f64::consts::FRAC_PI_2
                        })
                        .max_by(|(_, color_hue_vector_i), (_, color_hue_vector_j)| {
                            image_hue_vector
                                .project_onto_vector(**color_hue_vector_i)
                                .length()
                                .partial_cmp(
                                    &image_hue_vector
                                        .project_onto_vector(**color_hue_vector_j)
                                        .length(),
                                )
                                .unwrap()
                        });

                    match (left, right) {
                        (Some((i, color_hue_vector_i)), Some((j, color_hue_vector_j))) => {
                            image_in_implements[[i, x, y]] = image_hue_vector
                                .project_onto_vector(*color_hue_vector_i)
                                .length()
                                // Can't increase saturation beyond the max
                                .min(colors_in_hsl[i][1]);

                            image_in_implements[[j, x, y]] = image_hue_vector
                                .project_onto_vector(*color_hue_vector_j)
                                .length()
                                // Can't increase saturation beyond the max
                                .min(colors_in_hsl[j][1]);
                        }
                        (Some((i, color_hue_vector)), None)
                        | (None, Some((i, color_hue_vector))) => {
                            image_in_implements[[i, x, y]] = image_hue_vector
                                .project_onto_vector(*color_hue_vector)
                                .length()
                                // Can't increase saturation beyond the max
                                .min(colors_in_hsl[i][1]);
                        }
                        (None, None) => {}
                    }
                    // for (i, (c, (color_hsl, color_hue_vector))) in colors
                    //     .iter()
                    //     .zip(colors_in_hsl.iter().zip(color_hue_vectors.iter()))
                    //     .enumerate()
                    // {
                    //     // Black = lightness
                    //     if c.as_ref().iter().all(|channel| *channel < f64::EPSILON) {
                    //         image_in_implements[[i, x, y]] = 1. - hsl[2];
                    //     } else {
                    //         image_in_implements[[i, x, y]] =
                    //             // Can't blend colors, this color is useless
                    //             if image_hue_vector.angle_to(*color_hue_vector).radians.abs()
                    //                 > std::f64::consts::FRAC_PI_2
                    //             {
                    //                 0.0
                    //             } else {
                    //                 image_hue_vector
                    //                     .project_onto_vector(*color_hue_vector)
                    //                     .length()
                    //                     // Can't increase saturation beyond the max
                    //                     .min(color_hsl[1])
                    //             };
                    //     }
                    // }
                }
            }
        }
        ColorModel::Cielab => {
            todo!();
        }
    }

    // pen_image
    //     .slice_mut(s![i, .., ..])
    //     .assign(&image_in_cielab.map_axis(Axis(0), |lab| {
    //         let hue = vector(lab[1], lab[2]);
    //         let angle = hue.angle_to(*color);
    //         if angle.radians.abs() > std::f64::consts::FRAC_PI_2 {
    //             0.0
    //         } else {
    //             hue.project_onto_vector(*color).length() / color.length()
    //         }
    //     }));
    // for implement in &opt.implements {
    //     if let Implement::Pen { diameter, color } = implement {
    //         let diameter = diameter.get::<millimeter>();
    //         let diameter_in_pixels = diameter * dots_per_mm;
    //     }
    // }

    // let pen_image = {
    //     let mut pen_image: Array3<f64> = Array3::zeros((
    //         4,
    //         image_in_cielab.raw_dim()[1],
    //         image_in_cielab.raw_dim()[2],
    //     ));

    //     // Key (Black) derived as the inverse of lightness
    //     {
    //         pen_image
    //             .slice_mut(s![3, .., ..])
    //             .assign(&image_in_cielab.slice(s![0, .., ..]));
    //         pen_image
    //             .slice_mut(s![3, .., ..])
    //             .par_mapv_inplace(|v| 1.0 - v);
    //     }
    //     // RGB
    //     // Project hue onto each pen's color to derive a value [0, 1] for
    //     // how helpful a pen will be in reproducing a pixel
    //     match opt.color_model {
    //         ColorModel::Cielab => [
    //             vector(80.81351675261305, 69.88458436386973),
    //             vector(-79.28626260990568, 80.98938522093422),
    //             vector(68.29938535880123, -112.03112368261236),
    //         ]
    //         .iter()
    //         .enumerate()
    //         .for_each(|(i, color)| {
    //             pen_image
    //                 .slice_mut(s![i, .., ..])
    //                 .assign(&image_in_cielab.map_axis(Axis(0), |lab| {
    //                     let hue = vector(lab[1], lab[2]);
    //                     let angle = hue.angle_to(*color);
    //                     if angle.radians.abs() > std::f64::consts::FRAC_PI_2 {
    //                         0.0
    //                     } else {
    //                         hue.project_onto_vector(*color).length() / color.length()
    //                     }
    //                 }));
    //         }),
    //         ColorModel::Hsl => {
    //             let image_in_hsl = srgb_to_hsl(image.view());
    //             [
    //                 vector(1.0, 0.0),
    //                 vector(-0.5, 3.0f64.sqrt() / 2.0),
    //                 vector(-0.5, -(3.0f64.sqrt()) / 2.0),
    //             ]
    //             .iter()
    //             .enumerate()
    //             .for_each(|(i, color)| {
    //                 pen_image
    //                     .slice_mut(s![i, .., ..])
    //                     .assign(&image_in_hsl.map_axis(Axis(0), |hsl| {
    //                         let (sin, cos) = hsl[0].sin_cos();
    //                         let hue_vector = vector(cos, sin) * hsl[1];
    //                         if hue_vector.angle_to(*color).radians.abs()
    //                             > std::f64::consts::FRAC_PI_2
    //                         {
    //                             0.0
    //                         } else {
    //                             hue_vector.project_onto_vector(*color).length() * hsl[2]
    //                         }
    //                     }));
    //             })
    //         }
    //     }

    //     // Linearize color mapping for line drawings
    //     if matches!(opt.style, Style::Tsp | Style::Mst) {
    //         pen_image.mapv_inplace(|v| v.powi(2));
    //     } else if matches!(opt.style, Style::Triangulation | Style::Voronoi) {
    //         pen_image.mapv_inplace(|v| v.powi(3));
    //     }

    //     pen_image
    // };

    // Linearize color mapping for line drawings
    if matches!(opt.style, Style::Tsp | Style::Mst) {
        image_in_implements.mapv_inplace(|v| v.powi(2));
    } else if matches!(opt.style, Style::Triangulation | Style::Voronoi) {
        image_in_implements.mapv_inplace(|v| v.powi(3));
    }

    let width =
        (image_in_implements.raw_dim()[1] as f64 / dots_per_mm / opt.super_sample as f64).round();
    let height =
        (image_in_implements.raw_dim()[2] as f64 / dots_per_mm / opt.super_sample as f64).round();

    let mut surf = match &opt.out {
        Some(_) => cairo::SvgSurface::new(width, height, opt.out).unwrap(),
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

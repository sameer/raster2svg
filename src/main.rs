use cairo::{Context, LineCap};
use color::{ciexyz_to_cielab, srgb_to_ciexyz, srgb_to_hsl};
use image::io::Reader as ImageReader;
use log::*;
use lyon_geom::{euclid::Vector2D, point, vector, Angle, LineSegment};
use ndarray::prelude::*;
use num_traits::PrimInt;
use spade::delaunay::IntDelaunayTriangulation;
use std::{
    env,
    io::{self, Read},
    path::PathBuf,
    str::FromStr,
    vec,
};
use structopt::StructOpt;
use uom::si::f64::Length;
use uom::si::length::{inch, millimeter};

mod color;
mod hull;
mod mst;
mod tsp;
mod voronoi;

#[derive(Debug, StructOpt)]
#[structopt(author, about)]
struct Opt {
    /// A path to an image, else reads from stdin
    file: Option<PathBuf>,

    /// Determines the scaling of the output SVG
    #[structopt(long, default_value = "96")]
    dots_per_inch: f64,

    /// Diameter of the pen stroke in units of your choice
    #[structopt(long, default_value = "1 mm")]
    pen_diameter: Length,

    #[structopt(long, default_value = "CIELAB")]
    color_model: ColorModel,

    /// Output file path (overwrites old files), else writes to stdout
    #[structopt(short, long)]
    out: Option<PathBuf>,
}

#[derive(Debug)]
enum ColorModel {
    Hsl,
    Cielab,
}

impl FromStr for ColorModel {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "HSL" => Ok(ColorModel::Hsl),
            "CIELAB" => Ok(ColorModel::Cielab),
            _ => Err("only valid models are HSL or CIELAB"),
        }
    }
}

fn main() -> io::Result<()> {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "raster2svg=info")
    }
    env_logger::init();
    let opt = Opt::from_args();

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

        let image_in_cielab = ciexyz_to_cielab(&srgb_to_ciexyz(&image));
        let image_in_hsl = srgb_to_hsl(&image);
        let mut pen_image: Array3<f64> =
            Array::zeros((4, image_in_cielab.shape()[1], image_in_cielab.shape()[2]));

        // Key
        {
            pen_image
                .slice_mut(s![3, .., ..])
                .assign(&image_in_cielab.slice(s![0, .., ..]));
            pen_image.slice_mut(s![3, .., ..]).mapv_inplace(|v| 1.0 - v);
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
            ColorModel::Hsl => [
                vector(1.0, 0.0),
                vector(-0.5, 3.0f64.sqrt() / 2.0),
                vector(-0.5, -3.0f64.sqrt() / 2.0),
            ]
            .iter()
            .enumerate()
            .for_each(|(i, color)| {
                pen_image
                    .slice_mut(s![i, .., ..])
                    .assign(&image_in_hsl.map_axis(Axis(0), |hsl| {
                        let (sin, cos) = hsl[0].sin_cos();
                        let hue_vector = vector(cos, sin) * hsl[1];
                        if hue_vector.angle_to(*color).radians.abs() > std::f64::consts::FRAC_PI_2 {
                            0.0
                        } else {
                            hue_vector.project_onto_vector(*color).length() * hsl[2]
                        }
                    }));
            }),
        }

        // Linearize color mapping for line drawings
        pen_image.mapv_inplace(|v| v.powi(2));

        pen_image
    };

    let mm_per_inch = Length::new::<inch>(1.).get::<millimeter>();
    let dots_per_mm = opt.dots_per_inch / mm_per_inch;
    let pen_diameter = opt.pen_diameter.get::<millimeter>();
    let pen_diameter_in_pixels = pen_diameter * dots_per_mm;
    let stipple_area = (pen_diameter_in_pixels / 2.).powi(2) * std::f64::consts::PI;
    let width = (pen_image.shape()[1] as f64 / dots_per_mm).round();
    let height = (pen_image.shape()[2] as f64 / dots_per_mm).round();

    let mut surf = match &opt.out {
        Some(_) => cairo::SvgSurface::new(width, height, opt.out).unwrap(),
        None => cairo::SvgSurface::for_stream(width, height, std::io::stdout()).unwrap(),
    };
    surf.set_document_unit(cairo::SvgUnit::Mm);
    let ctx = Context::new(&surf);

    ctx.set_source_rgb(1., 1., 1.);
    ctx.rectangle(0., 0., width, height);
    ctx.fill();

    ctx.set_line_width(pen_diameter);
    ctx.set_line_cap(LineCap::Round);
    // Makes life easier to work with the same coords as the image
    ctx.scale(
        width / pen_image.shape()[1] as f64,
        height / pen_image.shape()[2] as f64,
    );

    for k in [0, 2, 3].iter().copied::<usize>() {
        info!("Processing {}", k);
        let mut voronoi_sites = vec![[pen_image.shape()[1] / 2, pen_image.shape()[2] / 2]];

        let initial_hysteresis = 0.6;
        let hysteresis_delta = 0.01;
        for iteration in 0..50 {
            if voronoi_sites.is_empty() {
                break;
            }
            debug!("Jump flooding voronoi");
            let colored_pixels = voronoi::jump_flooding_voronoi(
                &voronoi_sites,
                pen_image.shape()[1],
                pen_image.shape()[2],
            );
            let expected_assignment_capacity =
                pen_image.shape()[1] * pen_image.shape()[2] / voronoi_sites.len();
            let mut sites_to_points =
                vec![
                    Vec::<[usize; 2]>::with_capacity(expected_assignment_capacity);
                    voronoi_sites.len()
                ];
            for i in 0..pen_image.shape()[1] as usize {
                for j in 0..pen_image.shape()[2] as usize {
                    sites_to_points[colored_pixels[j][i]].push([i, j]);
                }
            }

            debug!("Linde-Buzo-Gray Stippling");

            let current_hysteresis = initial_hysteresis + iteration as f64 * hysteresis_delta;
            let remove_threshold = (1. - current_hysteresis / 2.) * stipple_area;
            let split_threshold = (1. + current_hysteresis / 2.) * stipple_area;

            let mut new_sites = Vec::with_capacity(voronoi_sites.len());
            let mut changed = false;
            for (_, points) in voronoi_sites.iter().zip(sites_to_points.iter()) {
                let density = points
                    .iter()
                    .map(|[x, y]| pen_image[[k, *x, *y]])
                    .sum::<f64>();
                let scaled_density = density;
                if scaled_density < remove_threshold {
                    changed = true;
                    continue;
                }
                let denominator = density;
                let numerator_y = points
                    .iter()
                    .map(|[x, y]| *y as f64 * pen_image[[k, *x, *y]])
                    .sum::<f64>();
                let numerator_x = points
                    .iter()
                    .map(|[x, y]| *x as f64 * pen_image[[k, *x, *y]])
                    .sum::<f64>();
                let centroid = [
                    ((numerator_x / denominator).round() as usize)
                        .clamp(0, pen_image.shape()[1] - 1),
                    ((numerator_y / denominator).round() as usize)
                        .clamp(0, pen_image.shape()[2] - 1),
                ];
                if scaled_density < split_threshold {
                    new_sites.push(centroid);
                } else {
                    if points.len() < 3 {
                        new_sites.push(centroid);
                        warn!("Can't split, there are too few points");
                        continue;
                    }
                    let hull = hull::convex_hull(&points);

                    let radius = hull
                        .iter()
                        .zip(hull.iter().skip(1).chain(hull.iter().take(1)))
                        .map(|(from, to)| {
                            let edge = LineSegment {
                                from: point(from[0] as f64, from[1] as f64),
                                to: point(to[0] as f64, to[1] as f64),
                            }
                            .to_line();
                            edge.distance_to_point(&point(centroid[0] as f64, centroid[1] as f64))
                        })
                        .min_by(|a, b| a.partial_cmp(&b).unwrap())
                        .unwrap();

                    if let Some((centroid_to_vertex, centroid_to_edge)) = hull
                        .iter()
                        .filter_map(|vertex| {
                            let centroid_to_vertex = LineSegment {
                                from: point(centroid[0] as f64, centroid[1] as f64),
                                to: point(vertex[0] as f64, vertex[1] as f64),
                            };
                            let line = centroid_to_vertex.to_line();

                            hull.iter()
                                .zip(hull.iter().skip(1).chain(hull.iter().take(1)))
                                .filter(|(from, to)| *from != vertex && *to != vertex)
                                .filter_map(|(from, to)| {
                                    let edge = LineSegment {
                                        from: point(from[0] as f64, from[1] as f64),
                                        to: point(to[0] as f64, to[1] as f64),
                                    };
                                    edge.line_intersection(&line)
                                })
                                .map(|opposite_edge_intersection| LineSegment {
                                    from: centroid_to_vertex.from,
                                    to: opposite_edge_intersection,
                                })
                                .map(|centroid_to_edge| (centroid_to_vertex, centroid_to_edge))
                                .next()
                        })
                        .max_by(|(a1, a2): &(LineSegment<f64>, LineSegment<f64>), (b1, b2): &(LineSegment<f64>, LineSegment<f64>)| {
                            a1.length()
                                .min(a2.length())
                                .partial_cmp(&b1.length().min(b2.length()))
                                .unwrap()
                        })
                    {
                        let left =
                            centroid_to_vertex.sample(radius / 2. / centroid_to_vertex.length()).try_cast::<usize>();
                        let right =
                            centroid_to_edge.sample(radius / 2. / centroid_to_edge.length()).try_cast::<usize>();
                        if let Some((left, right)) = left.zip(right) {
                            new_sites.push(left.to_array());
                            new_sites.push(right.to_array());
                            changed = true;
                        } else {
                            new_sites.push(centroid);
                        }
                    } else {
                        warn!("unable to split but it wants to be, moving to centroid instead");
                        new_sites.push(centroid);
                    }
                }
            }

            debug!("Check stopping condition (points: {})", new_sites.len());
            voronoi_sites = new_sites;
            if !changed {
                break;
            }
        }

        // On the off chance 2 points end up being the same...
        voronoi_sites.sort_unstable();
        voronoi_sites.dedup();

        if voronoi_sites.len() < 3 {
            warn!(
                "Color channel {} has too few vertices ({}) to draw, skipping",
                k,
                voronoi_sites.len()
            );
            continue;
        }

        let mut delaunay = IntDelaunayTriangulation::with_tree_locate();
        for vertex in &voronoi_sites {
            delaunay.insert([vertex[0] as i64, vertex[1] as i64]);
        }

        let tree = crate::mst::compute_mst(&voronoi_sites, &delaunay);
        let tsp = crate::tsp::approximate_tsp_with_mst(&voronoi_sites, &tree);

        debug!("Draw to svg");

        if k == 0 || k == 5 {
            ctx.set_source_rgb(1., 0., 0.);
        } else if k == 1 || k == 6 {
            ctx.set_source_rgb(0., 1., 0.);
        } else if k == 2 || k == 4 {
            ctx.set_source_rgb(0., 0., 1.);
        } else if k == 3 {
            ctx.set_source_rgb(0., 0., 0.);
        }

        // ctx.move_to(tree[0][0][0] as f64, tree[0][0][1] as f64);
        // for edge in &tree {
        //     ctx.move_to(edge[0][0] as f64, edge[0][1] as f64);
        //     ctx.line_to(edge[1][0] as f64, edge[1][1] as f64);
        //     ctx.stroke();
        // }

        // for point in voronoi_sites {
        //     ctx.move_to(point[0] as f64, point[1] as f64);
        //     ctx.arc(point[0] as f64, point[1] as f64, pen_diameter / 2., 0., 2. * std::f64::consts::PI);
        //     ctx.stroke();
        // }

        if let Some(first) = tsp.first() {
            ctx.move_to(first[0] as f64, first[1] as f64);
        }
        for next in tsp.iter().skip(1) {
            ctx.line_to(next[0] as f64, next[1] as f64);
        }
        ctx.stroke();
    }

    Ok(())
}

#[inline]
fn abs_distance_squared<T: PrimInt>(a: [T; 2], b: [T; 2]) -> T {
    let x_diff = if a[0] > b[0] {
        a[0] - b[0]
    } else {
        b[0] - a[0]
    };
    let y_diff = if a[1] > b[1] {
        a[1] - b[1]
    } else {
        b[1] - a[1]
    };
    x_diff.pow(2).saturating_add(y_diff.pow(2))
}

#[cfg(test)]
#[test]
fn ramp_to_average() {
    // let image = ImageReader::open("ramp_sites_corrected.png")
    //     .unwrap()
    //     .decode()
    //     .expect("not an image")
    //     .to_rgb16();

    // let image = Array::from_iter(
    //     image
    //         .pixels()
    //         .map(|p| p.0.iter().copied())
    //         .flatten()
    //         .map(|p| p as f64 / u16::MAX as f64),
    // )
    // .into_shape((image.height() as usize, image.width() as usize, 3))
    // .unwrap()
    // .reversed_axes();
    // println!(
    //     "{:?}",
    //     (0..5120)
    //         .map(|x| {
    //             let mut sum = 0.0f64;
    //             for i in 10 * x..10 * (x + 1) {
    //                 for j in 0..2000 {
    //                     sum += image[[0, i, j]];
    //                 }
    //             }
    //             sum / (2000. * 10.)
    //         })
    //         .collect::<Vec<_>>()
    // );
}

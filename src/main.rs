use cairo::Context;
use image::io::Reader as ImageReader;
use log::*;
use lyon_geom::{point, LineSegment};
use ndarray::prelude::*;
use num_traits::PrimInt;
use spade::delaunay::IntDelaunayTriangulation;
use std::{
    env,
    io::{self, Read},
    path::PathBuf,
    vec,
};
use structopt::StructOpt;
use uom::si::f64::Length;
use uom::si::length::{inch, millimeter};

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

    /// Output file path (overwrites old files), else writes to stdout
    #[structopt(short, long)]
    out: Option<PathBuf>,
}

const RED: [f64; 3] = [1., 0., 0.];
const GREEN: [f64; 3] = [0., 1., 0.];
const BLUE: [f64; 3] = [0., 0., 1.];

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

        let mut pen_image = Array::zeros((0, image.shape()[1], image.shape()[2]));

        [RED, GREEN, BLUE].iter().for_each(|color| {
            pen_image
                .push(
                    Axis(0),
                    image
                        .map_axis(Axis(0), |pixel| {
                            1. - pixel
                                .iter()
                                .zip(color)
                                .map(|(x1, x2)| (x1 - x2).powi(2))
                                .sum::<f64>()
                                .sqrt()
                                / 3.0_f64.sqrt()

                            // pixel.iter().zip(color).map(|(x1, x2)| x1 * x2).sum::<f64>()
                        })
                        .view(),
                )
                .unwrap();
        });

        // Add inverse lightness from D65 sRGB to D50 CIE XYZ to D50 CIE LAB
        pen_image
            .push(
                Axis(0),
                image
                    .map_axis(Axis(0), |pixel| {
                        const LUMINANCE_COEFFICIENTS: [f64; 3] = [0.2225045, 0.7168786, 0.0606169];
                        let y = pixel
                            .iter()
                            .copied()
                            .map(|component| {
                                if component <= 0.04045 {
                                    component / 12.92
                                } else {
                                    ((component + 0.055) / 1.055).powf(2.4)
                                }
                            })
                            .zip(&LUMINANCE_COEFFICIENTS)
                            .map(|(component, coefficient)| component * coefficient)
                            .sum::<f64>()
                            .clamp(0., LUMINANCE_COEFFICIENTS.iter().sum());

                        const DELTA: f64 = 6. / 29.;
                        let l =
                            116. * if y > DELTA.powi(3) {
                                y.cbrt()
                            } else {
                                y / (3. * DELTA.powi(2)) + 4. / 29.
                            } - 16.;

                        1. - (l.clamp(0., 100.) / 100.)
                    })
                    .view(),
            )
            .unwrap();

        for k in 0..pen_image.shape()[0] {
            for j in 0..pen_image.shape()[2] {
                for i in 0..pen_image.shape()[1] {
                    pen_image[[k, i, j]] = pen_image[[k, i, j]].powi(2);
                }
            }
        }
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
    // Makes life easier to work with the same coords as the image
    ctx.scale(
        width / pen_image.shape()[1] as f64,
        height / pen_image.shape()[2] as f64,
    );

    for k in [3usize].iter().copied() {
        info!("Processing {}", k);
        let mut voronoi_sites = vec![[pen_image.shape()[1] / 2, pen_image.shape()[2] / 2]];

        let initial_hysteresis = 0.6;
        let hysteresis_delta = 0.01;
        for iteration in 0..50 {
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
                        .max_by(|(a1, a2), (b1, b2)| {
                            a1.length()
                                .min(a2.length())
                                .partial_cmp(&b1.length().min(b2.length()))
                                .unwrap()
                        })
                    {
                        let left =
                            centroid_to_vertex.sample(radius / 2. / centroid_to_vertex.length());
                        let right =
                            centroid_to_edge.sample(radius / 2. / centroid_to_edge.length());
                        new_sites.push(left.to_usize().to_array());
                        new_sites.push(right.to_usize().to_array());
                        changed = true;
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
        //     ctx.move_to(point[0] as f64 - 0.5, point[1] as f64 - 0.5);
        //     ctx.line_to(point[0] as f64 + 0.5, point[1] as f64 + 0.5);
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

use cairo::Context;
use image::io::Reader as ImageReader;
use log::*;
use ndarray::prelude::*;
use rand::prelude::*;
use spade::delaunay::IntDelaunayTriangulation;
use std::{
    env,
    io::{self, Read},
    path::PathBuf,
};
use structopt::StructOpt;
use uom::si::f64::Length;
use uom::si::length::{inch, millimeter};

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

fn main() -> io::Result<()> {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "raster2svg=info")
    }
    env_logger::init();
    let opt = Opt::from_args();

    let mut image = {
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
        .to_rgb8();

        Array::from_iter(image.pixels().map(|p| p.0.iter().copied()).flatten())
            .into_shape((image.height() as usize, image.width() as usize, 3))
            .unwrap()
            .reversed_axes()
    };

    // Add inverse lightness under D50 illumination
    image
        .push(
            Axis(0),
            image
                .map_axis(Axis(0), |pixel| {
                    const XYZ_LUMINANCE: [f64; 3] = [0.212671, 0.715160, 0.072169];
                    let y = pixel
                        .iter()
                        .map(|component| *component as f64 / 255.)
                        .map(|component| {
                            if component <= 0.04045 {
                                component / 12.92
                            } else {
                                ((component + 0.055) / 1.055).powf(2.4)
                            }
                        })
                        .zip(&XYZ_LUMINANCE)
                        .map(|(component, coefficient)| component * coefficient)
                        .sum::<f64>();

                    const DELTA: f64 = 6. / 29.;
                    let l =
                        116. * if y > DELTA.powi(3) {
                            y.cbrt()
                        } else {
                            y / (3. * DELTA.powi(2)) + 4. / 29.
                        } - 16.;
                    // dbg!(y, l);
                    255 - (l.clamp(0., 100.) * 25.).round() as u8
                })
                .view(),
        )
        .unwrap();

    let pen_diameter = opt.pen_diameter.get::<millimeter>();
    let in_to_mm = Length::new::<inch>(1.).get::<millimeter>();
    let width = (image.shape()[1] as f64 / opt.dots_per_inch * in_to_mm).round();
    let height = (image.shape()[2] as f64 / opt.dots_per_inch * in_to_mm).round();

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
        width / image.shape()[1] as f64,
        height / image.shape()[2] as f64,
    );

    // Do green last so it appears on top of the SVG
    for k in [3].iter().copied() {
        let mut voronoi_points = {
            // let mut indices = dither
            //     .slice(s![k, .., ..])
            //     .indexed_iter()
            //     .filter_map(|idx_and_elem: ((usize, usize), &u8)| {
            //         if *idx_and_elem.1 == u8::MAX {
            //             Some([idx_and_elem.0 .0 as i64, idx_and_elem.0 .1 as i64])
            //         } else {
            //             None
            //         }
            //     })
            //     .collect::<Vec<_>>();
            // indices.shuffle(&mut rng);
            // indices.truncate(indices.len().min(100000));
            // indices
            let mut rng = thread_rng();
            (0..10000)
                .map(|_| {
                    [
                        rng.gen_range(0..image.shape()[1] as i64),
                        rng.gen_range(0..image.shape()[2] as i64),
                    ]
                })
                .collect::<Vec<_>>()
        };
        if voronoi_points.len() < 3 {
            warn!(
                "Color channel {} has too few vertices ({}) to draw, skipping",
                k,
                voronoi_points.len()
            );
            continue;
        }

        let mut colored_pixels;
        let mut last_average_distance: Option<f64> = None;
        loop {
            debug!("Naive voronoi");
            colored_pixels =
                voronoi::compute_voronoi(&voronoi_points, image.shape()[1], image.shape()[2]);

            debug!("Lloyd's algorithm");
            let expected_assignment_capacity =
                image.shape()[1] * image.shape()[2] / voronoi_points.len();
            let mut point_assignments =
                vec![
                    Vec::<[i64; 2]>::with_capacity(expected_assignment_capacity);
                    voronoi_points.len()
                ];
            for i in 0..image.shape()[1] as usize {
                for j in 0..image.shape()[2] as usize {
                    point_assignments[colored_pixels[j][i]].push([i as i64, j as i64]);
                }
            }

            let centroids: Vec<[i64; 2]> = point_assignments
                .iter()
                .map(|point_assignment| {
                    let denominator = point_assignment.iter().fold(0., |mut acc, [x, y]| {
                        acc += image[[k, *x as usize, *y as usize]] as f64 / 255.;
                        acc
                    });
                    let numerator_y = point_assignment.iter().fold(0., |mut acc, [x, y]| {
                        acc += *y as f64 * image[[k, *x as usize, *y as usize]] as f64 / 255.;
                        acc
                    });
                    let numerator_x = point_assignment.iter().fold(0., |mut acc, [x, y]| {
                        acc += *x as f64 * (image[[k, *x as usize, *y as usize]] as f64 / 255.);
                        acc
                    });
                    [numerator_x / denominator, numerator_y / denominator]
                })
                .map(|[x, y]| [x.round() as i64, y.round() as i64])
                .collect();

            debug!("Check stopping condition");
            // stopping condition: average distance moved by all stippling points is small
            let distance_sum = voronoi_points
                .iter()
                .zip(centroids.iter())
                .map(|(voronoi, centroid)| abs_distance_squared(*voronoi, *centroid))
                .map(|distance_squared| (distance_squared as f64).sqrt())
                .sum::<f64>();
            voronoi_points = centroids;
            let average_distance = distance_sum / voronoi_points.len() as f64;
            debug!("at {}", average_distance);
            if let Some(last_average_distance) = last_average_distance {
                if (last_average_distance - average_distance).abs() <= f64::EPSILON {
                    break;
                }
            }
            last_average_distance = Some(average_distance)
        }

        // On the off chance 2 points end up being the same...
        voronoi_points.sort();
        voronoi_points.dedup();

        let mut delaunay = IntDelaunayTriangulation::with_tree_locate();
        for vertex in &voronoi_points {
            delaunay.insert(*vertex);
        }

        let tree = crate::mst::compute_mst(&voronoi_points, &delaunay);

        // ctx.move_to(tree[0][0][0] as f64, tree[0][0][1] as f64);
        // for edge in &tree {
        //     ctx.move_to(edge[0][0] as f64, edge[0][1] as f64);
        //     ctx.line_to(edge[1][0] as f64, edge[1][1] as f64);
        //     ctx.stroke();
        // }
        let tsp = crate::tsp::approximate_tsp_with_mst(&voronoi_points, &tree);

        debug!("Draw to svg");

        if k == 0 {
            ctx.set_source_rgb(1., 0., 0.);
        } else if k == 1 {
            ctx.set_source_rgb(0., 1., 0.);
        } else if k == 2 {
            ctx.set_source_rgb(0., 0., 1.);
        } else if k == 3 {
            ctx.set_source_rgb(0., 0., 0.);
        }

        // for point in voronoi_points {
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

pub fn abs_distance_squared(a: [i64; 2], b: [i64; 2]) -> i64 {
    (a[0] - b[0]).pow(2) + (a[1] - b[1]).pow(2)
}

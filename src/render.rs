use std::sync::atomic::{AtomicBool, Ordering};

use crate::{
    color::Color,
    filter::{
        edge_flow_estimation, edge_tangent_flow, flow_based_difference_of_gaussians,
        step_edge_detection,
    },
    graph::{mst, tsp},
    kbn_summation,
    voronoi::{
        calculate_cell_properties, colors_to_assignments, jump_flooding_voronoi, AnnotatedSite,
    },
    Style,
};
use cairo::{Context, LineCap, LineJoin, Matrix};
use log::{debug, info, warn};
use lyon_geom::{point, Point};
use ndarray::prelude::*;
use rand::{thread_rng, Rng};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use spade::delaunay::IntDelaunayTriangulation;

pub fn render_fdog_based(
    image: ArrayView2<f64>,
    _super_sample: usize,
    instrument_diameter_in_pixels: f64,
    _style: Style,
    ctx: &Context,
    matrix: Matrix,
) {
    let mut image = image.to_owned();
    // Need to invert image for the sake of fdog
    image.par_mapv_inplace(|x| 1.0 - x);
    let etf = edge_tangent_flow(image.view());
    let fdog = flow_based_difference_of_gaussians(image.view(), etf.view());
    for (pos, value) in fdog.indexed_iter() {
        if *value < 1.0 {
            ctx.set_source_rgb(*value, *value, *value);
            ctx.set_matrix(matrix);
            ctx.move_to(pos.0 as f64, pos.1 as f64);
            ctx.rectangle(pos.0 as f64, pos.1 as f64, 1.0, 1.0);
            ctx.set_matrix(Matrix::identity());
            ctx.fill().unwrap();
        }
    }
}

/// Run an algorithm to produce a stippling and customize the output according to the desired style.
///
/// TODO: implement is assumed to be circular, can this support non-circular implements?
pub fn render_stipple_based(
    class_images: ArrayView3<f64>,
    implement_diameters_in_pixels: &[f64],
    colors: &[Color],
    super_sample: usize,
    style: Style,
    ctx: &Context,
    matrix: Matrix,
) {
    let (_, width, height) = class_images.dim();
    let class_to_sites =
        run_mlbg_stippling(class_images, implement_diameters_in_pixels, super_sample);

    for (((image, implement_diameter_in_pixels), color), mut voronoi_sites) in class_images
        .axis_iter(Axis(0))
        .zip(implement_diameters_in_pixels.iter())
        .zip(colors.into_iter())
        .zip(class_to_sites.into_iter())
    {
        // On the off chance 2 sites end up being the same...
        voronoi_sites.sort_unstable();
        voronoi_sites.dedup();

        if voronoi_sites.len() < 3 {
            warn!(
                "Channel has too few vertices ({}) to draw, skipping",
                voronoi_sites.len()
            );
            continue;
        }
        info!("Visualizing {}", color);

        ctx.set_line_cap(LineCap::Round);
        ctx.set_line_join(LineJoin::Round);
        ctx.set_source_rgb(color[0], color[1], color[2]);
        ctx.set_line_width(*implement_diameter_in_pixels);

        match style {
            Style::Stipple => {
                debug!("Draw to svg");
                ctx.set_matrix(matrix);
                for site in voronoi_sites {
                    ctx.move_to(site[0] as f64, site[1] as f64);
                    ctx.arc(
                        site[0] as f64,
                        site[1] as f64,
                        implement_diameter_in_pixels / 2.0,
                        0.,
                        std::f64::consts::TAU,
                    );
                }
                ctx.set_matrix(Matrix::identity());
                ctx.fill().unwrap();
            }
            Style::Voronoi => {
                debug!("Draw to svg");
                let sites_to_points = colors_to_assignments(
                    &voronoi_sites,
                    jump_flooding_voronoi(&voronoi_sites, width, height).view(),
                );
                for points in sites_to_points {
                    let properties = calculate_cell_properties(image, &points);
                    if let Some(hull) = properties.hull {
                        ctx.set_matrix(matrix);
                        if let Some(first) = hull.first() {
                            ctx.move_to(first[0] as f64, first[1] as f64);
                        }
                        for point in hull.iter().skip(1) {
                            ctx.line_to(point[0] as f64, point[1] as f64);
                        }
                        ctx.set_matrix(Matrix::identity());
                        ctx.stroke().unwrap();
                    }
                }
            }
            Style::Triangulation | Style::Mst | Style::Tsp => {
                let mut delaunay = IntDelaunayTriangulation::with_tree_locate();
                for vertex in &voronoi_sites {
                    delaunay.insert([vertex[0] as i64, vertex[1] as i64]);
                }

                if let Style::Triangulation = style {
                    debug!("Draw to svg");
                    ctx.set_matrix(matrix);
                    for edge in delaunay.edges() {
                        let from: &[i64; 2] = &edge.from();
                        let to: &[i64; 2] = &edge.to();
                        ctx.move_to(from[0] as f64, from[1] as f64);
                        ctx.line_to(to[0] as f64, to[1] as f64);
                    }
                    ctx.set_matrix(Matrix::identity());
                    ctx.stroke().unwrap();
                } else {
                    let tree = mst::compute_mst(&voronoi_sites, &delaunay);
                    if let Style::Mst = style {
                        debug!("Draw to svg");
                        ctx.set_matrix(matrix);
                        ctx.move_to(tree[0][0][0] as f64, tree[0][0][1] as f64);
                        for edge in &tree {
                            ctx.move_to(edge[0][0] as f64, edge[0][1] as f64);
                            ctx.line_to(edge[1][0] as f64, edge[1][1] as f64);
                        }
                        ctx.stroke().unwrap();
                        ctx.set_matrix(Matrix::identity());
                    } else {
                        let tsp = tsp::approximate_tsp_with_mst(&voronoi_sites, &tree);
                        debug!("Draw to svg");
                        ctx.set_matrix(matrix);
                        if let Some(first) = tsp.first() {
                            ctx.move_to(first[0] as f64, first[1] as f64);
                        }
                        for next in tsp.iter().skip(1) {
                            ctx.line_to(next[0] as f64, next[1] as f64);
                        }
                        ctx.stroke().unwrap();
                        ctx.set_matrix(Matrix::identity());
                    }
                }
            }
            Style::EdgesPlusHatching => unreachable!(),
        }
    }
}

/// Run Weighted Multi-Class Linde-Buzo-Gray Stippling and returns the per class stipples
///
/// TODO: implement is assumed to be circular, can this support non-circular implements?
///
/// <http://graphics.uni-konstanz.de/publikationen/Deussen2017LindeBuzoGray/WeightedLindeBuzoGrayStippling_authorversion.pdf>
/// <https://kops.uni-konstanz.de/bitstream/handle/123456789/55976/Schulz_2-3pieljazuoer1.pdf?sequence=1&isAllowed=y>
fn run_mlbg_stippling(
    class_images: ArrayView3<f64>,
    implement_diameters_in_pixels: &[f64],
    super_sample: usize,
) -> Vec<Vec<[i64; 2]>> {
    const INITIAL_HYSTERESIS: f64 = 0.6;
    const HYSTERESIS_DELTA: f64 = 0.01;
    const ZERO: Point<f64> = Point::new(0., 0.);

    let implement_areas = implement_diameters_in_pixels
        .iter()
        .copied()
        .map(|diameter| (diameter / 2.).powi(2) * std::f64::consts::PI)
        .collect::<Vec<_>>();
    let (classes, width, height) = class_images.dim();
    let upper_bound = point((width - 1) as f64, (height - 1) as f64);
    let mut rng = thread_rng();
    let mut class_to_sites = (0..classes)
        .map(|_| {
            vec![[
                rng.gen_range(0..width as i64),
                rng.gen_range(0..height as i64),
            ]]
        })
        .collect::<Vec<_>>();

    for iteration in 0..140 {
        let current_hysteresis = INITIAL_HYSTERESIS + iteration as f64 * HYSTERESIS_DELTA;
        let remove_threshold = 1. - current_hysteresis / 2.;
        let split_threshold = 1. + current_hysteresis / 2.;

        let changed = AtomicBool::new(false);
        let class_to_site_global_centroids = {
            let mut acc = class_to_sites
                .iter()
                .map(|voronoi_sites| vec![None; voronoi_sites.len()])
                .collect::<Vec<_>>();
            let global_sites = class_to_sites
                .iter()
                .enumerate()
                .map(|(i, voronoi_sites)| {
                    voronoi_sites
                        .iter()
                        .copied()
                        .enumerate()
                        .map(move |(j, site)| AnnotatedSite {
                            site,
                            annotation: (i, j),
                        })
                })
                .flatten()
                .collect::<Vec<_>>();
            let global_colored_pixels = jump_flooding_voronoi(&global_sites, width, height);
            let global_sites_to_points =
                colors_to_assignments(&global_sites, global_colored_pixels.view());
            global_sites_to_points
                .into_iter()
                .zip(global_sites.iter())
                .for_each(|(points, annotated_site)| {
                    let (num_centroids, summed_centroid) = (0..classes)
                        .filter_map(|k| {
                            let class_image = class_images.slice(s![k, .., ..]);
                            calculate_cell_properties(class_image, &points).centroid
                        })
                        .fold(
                            (0, ZERO),
                            |(num_centroids, summed_centroid), class_centroid| {
                                (
                                    num_centroids + 1,
                                    summed_centroid + class_centroid.to_vector(),
                                )
                            },
                        );
                    if num_centroids > 0 {
                        acc[annotated_site.annotation.0][annotated_site.annotation.1] =
                            Some(summed_centroid / num_centroids as f64);
                    }
                });
            acc
        };

        // Don't parallelize outer loop b/c inner loop already uses all cores
        class_to_sites = (0..classes)
            .into_iter()
            .map(|k| class_images.slice(s![k, .., ..]))
            .zip(implement_areas.iter())
            .zip(class_to_sites.iter())
            .zip(class_to_site_global_centroids.iter())
            .map(
                |(((class_image, implement_area), sites), site_global_centroids)| {
                    if sites.is_empty() {
                        return vec![];
                    }
                    let colored_pixels = jump_flooding_voronoi(&sites, width, height);
                    let sites_to_points = colors_to_assignments(&sites, colored_pixels.view());

                    sites
                        .par_iter()
                        .zip(sites_to_points.par_iter())
                        .zip(site_global_centroids.par_iter())
                        .map(|((site, points), global_centroid)| {
                            let mut rng = thread_rng();
                            let cell_properties =
                                calculate_cell_properties(class_image.view(), points);
                            let moments = &cell_properties.moments;

                            // Density is very low, remove this point early
                            if moments.m00 <= f64::EPSILON {
                                changed.fetch_or(true, Ordering::Relaxed);
                                return vec![];
                            }
                            let should_use_global = global_centroid.is_some() && {
                                kbn_summation! {
                                    for other_class_cell_properties in (0..classes).map(|k| {
                                        let other_class_image = class_images.slice(s![k, .., ..]);
                                        calculate_cell_properties(other_class_image, &points)
                                    }) => {
                                        sum_class_densities += other_class_cell_properties.moments.m00 / points.len() as f64;
                                    }
                                }
                                let average_density = cell_properties.moments.m00 / points.len() as f64;
                                rng.gen_bool((sum_class_densities - average_density).clamp(0., 1.))
                            };
                            let centroid = if should_use_global { global_centroid.unwrap() } else {cell_properties.centroid.unwrap()};

                            let scaled_density = moments.m00 / super_sample.pow(2) as f64;

                            let line_area = implement_area;

                            match (
                                scaled_density < remove_threshold * line_area,
                                scaled_density < split_threshold * line_area,
                                cell_properties.phi_oriented_segment_through_centroid,
                            ) {
                                // Below remove threshold, remove point
                                (true, _, _) => {
                                    changed.fetch_or(true, Ordering::Relaxed);
                                    vec![]
                                }
                                // Below split threshold, keep as centroid
                                (false, true, _) | (_, _, None) => {
                                    let new_site = centroid
                                        .clamp(ZERO, upper_bound)
                                        .round()
                                        .cast::<i64>()
                                        .to_array();
                                    changed.fetch_or(*site != new_site, Ordering::Relaxed);
                                    vec![new_site]
                                }
                                // Above split threshold, split along phi from the centroid
                                (false, false, Some(line_segment)) => {
                                    if line_segment.length() < f64::EPSILON {
                                        warn!("It shouldn't be possible to have a phi segment of 0 length here: {:?}", &cell_properties.centroid);
                                    }
                                    let left = line_segment
                                        .sample(1. / 3.)
                                        .clamp(ZERO, upper_bound)
                                        .round()
                                        .cast::<i64>()
                                        .to_array();
                                    let right = line_segment
                                        .sample(2. / 3.)
                                        .clamp(ZERO, upper_bound)
                                        .round()
                                        .cast::<i64>()
                                        .to_array();

                                    changed.fetch_or(true, Ordering::Relaxed);
                                    if left == right {
                                        warn!(
                                            "Splitting a point produced the same point: {:?}",
                                            &cell_properties.centroid
                                        );
                                        vec![left]
                                    } else {
                                        vec![left, right]
                                    }
                                }
                            }
                        })
                        .flatten()
                        .collect()
                },
            )
            .collect();
        debug!(
            "Check stopping condition (iteration = {}, sites: {})",
            iteration,
            class_to_sites
                .iter()
                .map(|sites| sites.len())
                .sum::<usize>()
        );
        if !changed.load(Ordering::Relaxed) {
            break;
        }
    }

    class_to_sites
}

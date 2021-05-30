use crate::abs_distance_squared;

/// Given a set of points in a bounding box from (0, 0) to (width, height),
/// return the assignment of coordinates in that box to their nearest neighbor
/// using the Jump Flooding Algorithm.
///
/// https://www.comp.nus.edu.sg/~tants/jfa/i3d06.pdf
pub fn compute_voronoi(points: &[[i64; 2]], width: usize, height: usize) -> Vec<Vec<usize>> {
    if points.is_empty() {
        return vec![];
    }
    // use usize::MAX to represent colorless cells
    let mut grid = vec![vec![usize::MAX; width]; height];
    points.iter().enumerate().for_each(|(color, point)| {
        grid[point[1] as usize][point[0] as usize] = color;
    });

    // Claim: without a scratchpad, converge to the correct point assignment faster
    // let mut grid_scratchpad = grid.clone();
    let mut round_step = (width.max(height)).next_power_of_two() / 2;
    let width = width as i64;
    let height = height as i64;
    while round_step != 0 {
        for j in 0..height {
            for i in 0..width {
                for y_dir in -1..=1 {
                    let y = j + y_dir * round_step as i64;
                    if y < 0 || y >= height {
                        continue;
                    }
                    for x_dir in -1..=1 {
                        let x = i + x_dir * round_step as i64;
                        if x < 0 || x >= width {
                            continue;
                        }
                        let new = grid[y as usize][x as usize];
                        if new != usize::MAX {
                            let current = grid[j as usize][i as usize];
                            if current == usize::MAX
                                || abs_distance_squared([i, j], points[new])
                                    < abs_distance_squared([i, j], points[current])
                            {
                                grid[j as usize][i as usize] = new;
                            }
                        }
                    }
                }
            }
        }

        round_step /= 2;
    }

    grid
}

#[cfg(test)]
#[test]
fn test_voronoi() {
    use image::ImageBuffer;

    let points = [[0, 0], [63, 63], [24, 5], [0, 127], [127, 127], [127, 240]];
    const WIDTH: usize = 256;
    const HEIGHT: usize = 256;
    let grid = compute_voronoi(&points, WIDTH, HEIGHT);
    let out = ImageBuffer::from_fn(WIDTH as u32, HEIGHT as u32, |x, y| {
        let color = grid[y as usize][x as usize];
        let r = (255. * (color as f64 / (points.len() + 3) as f64)).round() as u8;
        if points.iter().any(|p| x == p[0] as u32 && y == p[1] as u32) {
            image::Rgb([255, 0, 255])
        } else {
            image::Rgb([r, r, r])
        }
    });
    out.save_with_format("out.png", image::ImageFormat::Png)
        .unwrap();
}

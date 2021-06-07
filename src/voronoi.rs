use crate::abs_distance_squared;

/// Given a set of sites in a bounding box from (0, 0) to (width, height),
/// return the assignment of coordinates in that box to their nearest neighbor
/// using the Jump Flooding Algorithm.
///
/// https://www.comp.nus.edu.sg/~tants/jfa/i3d06.pdf
pub fn jump_flooding_voronoi(sites: &[[usize; 2]], width: usize, height: usize) -> Vec<Vec<usize>> {
    if sites.is_empty() {
        return vec![];
    }
    // use usize::MAX to represent colorless cells
    let mut grid = vec![vec![usize::MAX; width]; height];
    sites.iter().enumerate().for_each(|(color, site)| {
        grid[site[1] as usize][site[0] as usize] = color;
    });

    let mut round_step = (width.max(height))
        .checked_next_power_of_two()
        .map(|x| x / 2)
        .unwrap_or_else(|| (width.max(height) / 2).next_power_of_two());
    while round_step != 0 {
        for y_dir in -1..=1 {
            let y_range = if y_dir == -1 { round_step } else { 0 }..if y_dir == 1 {
                height.saturating_sub(round_step)
            } else {
                height
            };
            for x_dir in -1..=1 {
                let x_range = if x_dir == -1 { round_step } else { 0 }..if x_dir == 1 {
                    width.saturating_sub(round_step)
                } else {
                    width
                };
                for j in y_range.clone() {
                    for i in x_range.clone() {
                        let y = match y_dir {
                            -1 => j - round_step,
                            0 => j,
                            1 => j + round_step,
                            _ => unreachable!(),
                        };
                        let x = match x_dir {
                            -1 => i - 1,
                            0 => i,
                            1 => i + round_step,
                            _ => unreachable!(),
                        };
                        let new = grid[y][x];
                        if new != usize::MAX {
                            let current = grid[j][i];
                            if current == usize::MAX
                                || abs_distance_squared([i, j], sites[new])
                                    < abs_distance_squared([i, j], sites[current])
                            {
                                grid[j][i] = new;
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
mod tests {
    use super::jump_flooding_voronoi;
    use crate::abs_distance_squared;

    #[test]
    fn test_jump_flooding_voronoi() {
        const WIDTH: usize = 256;
        const HEIGHT: usize = 256;
        let sites = [
            [0, 0],
            [0, HEIGHT - 1],
            [WIDTH - 1, 0],
            [WIDTH - 1, HEIGHT - 1],
            [WIDTH / 2, HEIGHT / 2],
        ];
        let assignments = jump_flooding_voronoi(&sites, WIDTH, HEIGHT);
        for j in 0..HEIGHT {
            for i in 0..WIDTH {
                let min_distance = sites
                    .iter()
                    .map(|site| abs_distance_squared(*site, [i, j]))
                    .min()
                    .unwrap();
                let actual_distance = abs_distance_squared(sites[assignments[j][i]], [i, j]);

                // Don't check the assigned site because of distance ties
                assert_eq!(min_distance, actual_distance);
            }
        }
    }
}

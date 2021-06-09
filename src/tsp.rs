use crate::abs_distance_squared;
use fxhash::{FxHashMap as HashMap, FxHashSet as HashSet};
use num_traits::{FromPrimitive, PrimInt};
use rand::{distributions::Standard, prelude::Distribution, thread_rng, Rng};
use std::{fmt::Debug, hash::Hash, iter::Sum};

#[derive(PartialEq, Eq, Debug)]
struct Edge<T: PrimInt + Eq + PartialEq + PartialOrd + Ord + Debug>([[T; 2]; 2]);

impl<T: PrimInt + Eq + PartialEq + PartialOrd + Ord + Debug> Edge<T> {
    fn length(&self) -> T {
        abs_distance_squared(self.0[0], self.0[1])
    }
}

impl<T: PrimInt + Eq + PartialEq + PartialOrd + Ord + Debug> PartialOrd for Edge<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.length().partial_cmp(&other.length())
    }
}

impl<T: PrimInt + Eq + PartialEq + PartialOrd + Ord + Debug> Ord for Edge<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.length().cmp(&other.length())
    }
}

#[derive(PartialEq, Eq, Debug)]
struct RemoveAddEdgePair<T: PrimInt + Eq + PartialEq + PartialOrd + Ord + Debug> {
    original_edge: [[T; 2]; 2],
    new_edge: [[T; 2]; 2],
}

impl<T: PrimInt + Eq + PartialEq + PartialOrd + Ord + Debug> RemoveAddEdgePair<T> {
    /// Best improvement to the tree (greatest reduction or smallest increase in length)
    ///
    /// Note that this saturates to 0 for unsized integers but does not affect the
    /// algorithm because we are looking for maxima.
    fn diff(&self) -> T {
        abs_distance_squared(self.original_edge[0], self.original_edge[1])
            .saturating_sub(abs_distance_squared(self.new_edge[0], self.new_edge[1]))
    }
}

impl<T: PrimInt + Eq + PartialEq + PartialOrd + Ord + Debug> PartialOrd for RemoveAddEdgePair<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.diff().partial_cmp(&other.diff())
    }
}

impl<T: PrimInt + Eq + PartialEq + PartialOrd + Ord + Debug> Ord for RemoveAddEdgePair<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.diff().cmp(&other.diff())
    }
}

/// Approximate an open loop TSP solution by way of greedy branch elimination + local improvement.
///
/// http://cs.uef.fi/sipu/pub/applsci-11-00177.pdf
///
/// See the two source code comments prefixed with `NOTE:`
/// for how to use the non-greedy all pairs branch elimination
/// variant, which takes significantly longer.
///
/// TODO: consider using the Christofides algorithm which has a bound on the length of the worst path.
/// The initial solution of greedy branch elimination is actually quite bad.
pub fn approximate_tsp_with_mst<
    T: PrimInt + FromPrimitive + Eq + PartialEq + PartialOrd + Ord + Hash + Debug + Sum,
>(
    vertices: &[[T; 2]],
    tree: &[[[T; 2]; 2]],
) -> Vec<[T; 2]> {
    if vertices.len() <= 2 {
        return vec![];
    }
    let mut adjacency_map: HashMap<[T; 2], HashSet<[T; 2]>> = HashMap::default();
    tree.iter().for_each(|edge| {
        adjacency_map.entry(edge[0]).or_default().insert(edge[1]);
        adjacency_map.entry(edge[1]).or_default().insert(edge[0]);
    });

    let vertex_to_index = vertices
        .iter()
        .enumerate()
        .map(|(i, vertex)| (*vertex, i))
        .collect::<HashMap<_, _>>();
    let mut branch_list = adjacency_map
        .iter()
        .filter(|(_, adjacencies)| adjacencies.len() >= 3)
        .map(|(branch, adjacencies)| {
            adjacencies
                .iter()
                .map(move |adjacency| Edge([*branch, *adjacency]))
        })
        .flatten()
        .collect::<Vec<_>>();
    branch_list.sort();

    while let Some(Edge([branch, disconnected_node])) = branch_list.pop() {
        dbg!(branch_list.len());
        // No longer a branch
        if adjacency_map.get(&branch).unwrap().len() <= 2 {
            continue;
        }
        // Disconnected node was once a branch, already processed this pair
        if !adjacency_map
            .get(&branch)
            .unwrap()
            .contains(&disconnected_node)
        {
            continue;
        }

        // Remove edge
        adjacency_map
            .get_mut(&branch)
            .unwrap()
            .remove(&disconnected_node);
        adjacency_map
            .get_mut(&disconnected_node)
            .unwrap()
            .remove(&branch);

        // Now there are (in theory) two disconnected trees
        // Find the two connected trees in the graph
        let mut disconnected_tree_visited: Vec<bool> = vec![false; vertices.len()];
        let mut disconnected_tree_visit_count = 1;
        {
            disconnected_tree_visited[*vertex_to_index.get(&disconnected_node).unwrap()] = true;
            let mut dfs = vec![&disconnected_node];
            while let Some(head) = dfs.pop() {
                for adjacency in adjacency_map.get(head).unwrap() {
                    let adjacency_idx = *vertex_to_index.get(adjacency).unwrap();
                    if !disconnected_tree_visited[adjacency_idx] {
                        disconnected_tree_visited[adjacency_idx] = true;
                        disconnected_tree_visit_count += 1;
                        dfs.push(adjacency);
                    }
                }
            }
        }

        // Find leaves in the two
        // Pick the shortest possible link between two leaves that would reconnect the trees

        let (branch_tree_leaf, disconnected_tree_leaf) = if disconnected_tree_visit_count
            > vertices.len() - disconnected_tree_visit_count
        {
            let branch_tree_leaves = disconnected_tree_visited
                .iter()
                .enumerate()
                .filter(|(i, in_disconnected_tree)| {
                    !**in_disconnected_tree && adjacency_map.get(&vertices[*i]).unwrap().len() <= 1
                })
                .map(|(branch_idx, _)| vertices[branch_idx])
                .collect::<Vec<_>>();

            disconnected_tree_visited
                .iter()
                .enumerate()
                .filter(|(i, in_disconnected_tree)| {
                    **in_disconnected_tree && adjacency_map.get(&vertices[*i]).unwrap().len() <= 1
                })
                .map(|(disconnected_idx, _)| vertices[disconnected_idx])
                .map(|disconnected_tree_leaf| {
                    branch_tree_leaves
                        .iter()
                        .map(move |branch_tree_leaf| (*branch_tree_leaf, disconnected_tree_leaf))
                })
                .flatten()
                .min_by_key(|(branch_tree_leaf, disconnected_tree_leaf)| {
                    abs_distance_squared(*branch_tree_leaf, *disconnected_tree_leaf)
                })
                .unwrap()
        } else {
            let disconnected_tree_leaves = disconnected_tree_visited
                .iter()
                .enumerate()
                .filter(|(i, in_disconnected_tree)| {
                    **in_disconnected_tree && adjacency_map.get(&vertices[*i]).unwrap().len() <= 1
                })
                .map(|(disconnected_idx, _)| vertices[disconnected_idx])
                .collect::<Vec<_>>();

            disconnected_tree_visited
                .iter()
                .enumerate()
                .filter(|(i, in_disconnected_tree)| {
                    !**in_disconnected_tree && adjacency_map.get(&vertices[*i]).unwrap().len() <= 1
                })
                .map(|(branch_idx, _)| vertices[branch_idx])
                .map(|branch_tree_leaf| {
                    disconnected_tree_leaves
                        .iter()
                        .map(move |disconnected_tree_leaf| {
                            (branch_tree_leaf, *disconnected_tree_leaf)
                        })
                })
                .flatten()
                .min_by_key(|(branch_tree_leaf, disconnected_tree_leaf)| {
                    abs_distance_squared(*branch_tree_leaf, *disconnected_tree_leaf)
                })
                .unwrap()
        };

        // Connect leaves
        adjacency_map
            .get_mut(&branch_tree_leaf)
            .unwrap()
            .insert(disconnected_tree_leaf);
        adjacency_map
            .get_mut(&disconnected_tree_leaf)
            .unwrap()
            .insert(branch_tree_leaf);
    }

    // Extract path from the adjacency list
    let mut path: Vec<[T; 2]> = Vec::with_capacity(vertices.len().saturating_sub(1));
    if let Some((first_vertex, adjacencies)) = adjacency_map
        .iter()
        .find(|(_, adjacencies)| adjacencies.len() == 1)
    {
        path.push(*first_vertex);
        path.push(*adjacencies.iter().next().unwrap());
    }

    // The number of edges in an open loop TSP path is equal to the number of vertices - 1
    while path.len() < vertices.len() {
        let last_vertex = *path.last().unwrap();
        let second_to_last_vertex = *path.iter().rev().nth(1).unwrap();
        let next_vertex = adjacency_map
            .get(&last_vertex)
            .unwrap()
            .iter()
            .find(|adjacency| **adjacency != second_to_last_vertex)
            .unwrap();
        path.push(*next_vertex);
    }
    local_improvement(&path)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Operator {
    /// Move vertex between other vertices
    Relocate,
    /// Swap the positions of two vertices
    ///
    /// As the name suggests, it tends to
    /// remove crossed edges in the path.
    ///
    /// This is by my own design based on intuition around
    /// good vs bad tours generated by the greedy approximation.
    Disentangle,
    /// Swap vertices between two edges
    TwoOpt,
    /// Change the beginning and/or end of the path by swapping an edge
    LinkSwap,
}

impl Distribution<Operator> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Operator {
        match rng.gen_range(0..=3) {
            0 => Operator::Relocate,
            1 => Operator::Disentangle,
            2 => Operator::TwoOpt,
            3 => Operator::LinkSwap,
            _ => unreachable!(),
        }
    }
}

/// Local improvement of an open loop TSP solution using the relocate, disentangle, 2-opt, and link swap operators.
///
/// Uses simulated annealing noise in the first half of iterations for hill-climbing.
///
/// https://www.mdpi.com/2076-3417/9/19/3985/pdf
///
fn local_improvement<
    T: PrimInt + FromPrimitive + Eq + PartialEq + PartialOrd + Ord + Hash + Debug + Sum,
>(
    path: &[[T; 2]],
) -> Vec<[T; 2]> {
    let mut best = path.to_owned();
    let mut best_sum = best
        .iter()
        .zip(best.iter().skip(1))
        .map(|(from, to)| abs_distance_squared(*from, *to))
        .sum::<T>();

    let mut current = best.clone();
    let mut current_sum = best_sum;
    let mut simulated_annealing_noise_min = 0.;
    let mut rng = thread_rng();

    let sample_count = (current.len() as f64 * 0.001) as usize;
    const ITERATIONS: usize = 20000;

    // Right now this doesn't do anything,
    // but there may be a good early stopping technique that could use it.
    let mut stuck_by_operator = [
        Operator::Relocate,
        Operator::Disentangle,
        Operator::TwoOpt,
        Operator::LinkSwap,
    ]
    .iter()
    .map(|operator| (*operator, false))
    .collect::<HashMap<_, _>>();

    // TODO: fix bug where annealing doesn't work because of skipping diffs that are less than 0
    for idx in 0..ITERATIONS {
        let operator: Operator = rng.gen();
        let is_annealing = idx < ITERATIONS / 2;

        match operator {
            Operator::Relocate => {
                let mut relocates = (0..sample_count)
                    .map(|_| rng.gen_range(1..current.len().saturating_sub(1)))
                    .collect::<Vec<_>>();
                // move i between j and j+1
                if let Some((i, j, diff)) = relocates
                    .drain(..)
                    .map(|i| {
                        (0..=i.saturating_sub(2))
                            .chain((i.saturating_add(2)..current.len()).map(|j| j - 1))
                            .map(move |j| (i, j))
                    })
                    .flatten()
                    .map(|(i, j)| {
                        let positive_diff = abs_distance_squared(current[j], current[j + 1])
                            + abs_distance_squared(current[i - 1], current[i])
                            + abs_distance_squared(current[i], current[i + 1]);
                        let negative_diff = abs_distance_squared(current[i - 1], current[i + 1])
                            + abs_distance_squared(current[i], current[j])
                            + abs_distance_squared(current[i], current[j + 1]);

                        let diff = positive_diff.saturating_sub(negative_diff);
                        (i, j, diff)
                    })
                    .max_by_key(|(_, _, diff)| {
                        if is_annealing {
                            T::from_f64(
                                diff.to_f64().unwrap()
                                    * rng.gen_range(simulated_annealing_noise_min..=1.),
                            )
                            .unwrap()
                        } else {
                            *diff
                        }
                    })
                {
                    if diff <= T::zero() {
                        *stuck_by_operator.get_mut(&operator).unwrap() = true;
                        continue;
                    } else {
                        *stuck_by_operator.get_mut(&operator).unwrap() = false;
                    }

                    let vertex = current.remove(i);
                    if j + 1 < i {
                        current.insert(j + 1, vertex);
                    } else {
                        // everything got shifted to the left so it's 1 less
                        current.insert(j, vertex);
                    }

                    current_sum = current_sum - diff;

                    dbg!(idx, current_sum, best_sum);
                    if current_sum < best_sum {
                        best = current.clone();
                        best_sum = current_sum;
                    }
                }
            }
            Operator::Disentangle => {
                let mut swaps = (0..sample_count)
                    .map(|_| rng.gen_range(0..current.len()))
                    .collect::<Vec<_>>();
                // swap i and j
                if let Some((i, j, diff)) = swaps
                    .drain(..)
                    .map(|i| {
                        (0..i.saturating_sub(1))
                            .map(move |j| (j, i))
                            .chain((i.saturating_add(2)..current.len()).map(move |j| (i, j)))
                    })
                    .flatten()
                    .map(|(i, j)| {
                        let mut positive_diff = T::zero();
                        let mut negative_diff = T::zero();
                        if i != 0 {
                            positive_diff =
                                positive_diff + abs_distance_squared(current[i - 1], current[i]);
                            negative_diff =
                                negative_diff + abs_distance_squared(current[i - 1], current[j]);
                        }
                        positive_diff =
                            positive_diff + abs_distance_squared(current[i], current[i + 1]);
                        negative_diff =
                            negative_diff + abs_distance_squared(current[j], current[i + 1]);
                        positive_diff =
                            positive_diff + abs_distance_squared(current[j - 1], current[j]);
                        negative_diff =
                            negative_diff + abs_distance_squared(current[j - 1], current[i]);
                        if j + 1 != current.len() {
                            positive_diff =
                                positive_diff + abs_distance_squared(current[j], current[j + 1]);
                            negative_diff =
                                negative_diff + abs_distance_squared(current[i], current[j + 1]);
                        }
                        let diff = positive_diff.saturating_sub(negative_diff);
                        (i, j, diff)
                    })
                    .max_by_key(|(_, _, diff)| {
                        if is_annealing {
                            T::from_f64(
                                diff.to_f64().unwrap()
                                    * rng.gen_range(simulated_annealing_noise_min..=1.),
                            )
                            .unwrap()
                        } else {
                            *diff
                        }
                    })
                {
                    if diff <= T::zero() {
                        *stuck_by_operator.get_mut(&operator).unwrap() = true;
                        continue;
                    } else {
                        *stuck_by_operator.get_mut(&operator).unwrap() = false;
                    }

                    current.swap(i, j);
                    current_sum = current_sum - diff;

                    dbg!(idx, current_sum, best_sum);
                    if current_sum < best_sum {
                        best = current.clone();
                        best_sum = current_sum;
                    }
                }
            }
            Operator::TwoOpt => {
                let mut edges = (0..sample_count)
                    .map(|_| {
                        let i = rng.gen_range(0..current.len().saturating_sub(1));
                        let j = i.saturating_add(1);
                        (i, j)
                    })
                    .collect::<Vec<_>>();
                // permute the points of the this and other edges
                if let Some((this, other, diff)) = edges
                    .drain(..)
                    .map(|(i, j)| {
                        (0..i).zip(1..i).map(move |other| (other, (i, j))).chain(
                            (j.saturating_add(1)..current.len())
                                .zip(j.saturating_add(2)..current.len())
                                .map(move |other| ((i, j), other)),
                        )
                    })
                    .flatten()
                    .map(|(this, other)| {
                        (
                            this,
                            other,
                            (abs_distance_squared(current[this.0], current[this.1])
                                + abs_distance_squared(current[other.0], current[other.1]))
                            .saturating_sub(
                                abs_distance_squared(current[this.0], current[other.0])
                                    + abs_distance_squared(current[this.1], current[other.1]),
                            ),
                        )
                    })
                    .max_by_key(|(_, _, diff)| {
                        if is_annealing {
                            T::from_f64(
                                diff.to_f64().unwrap()
                                    * rng.gen_range(simulated_annealing_noise_min..=1.),
                            )
                            .unwrap()
                        } else {
                            *diff
                        }
                    })
                {
                    if diff <= T::zero() {
                        *stuck_by_operator.get_mut(&operator).unwrap() = true;
                        continue;
                    } else {
                        *stuck_by_operator.get_mut(&operator).unwrap() = false;
                    }

                    let reversed_middle = current[this.1..=other.0]
                        .iter()
                        .rev()
                        .copied()
                        .collect::<Vec<_>>();
                    current[this.1..=other.0]
                        .iter_mut()
                        .zip(reversed_middle.iter())
                        .for_each(|(dest, origin)| *dest = *origin);

                    current_sum = current_sum - diff;
                    dbg!(idx, current_sum, best_sum);

                    if current_sum < best_sum {
                        best = current.clone();
                        best_sum = current_sum;
                    }
                }
            }
            Operator::LinkSwap => {
                let first = *current.first().unwrap();
                let last = *current.last().unwrap();

                // Swap the first and/or last vertex with inner vertices
                if let Some((i, j, pair, diff)) = (0..current.len())
                    .zip(1..current.len())
                    .filter_map(|(i, j)| {
                        let from = current[i];
                        let to = current[j];
                        [[from, last], [first, to], [first, last]]
                            .iter()
                            .map(|new_edge| {
                                let pair = RemoveAddEdgePair {
                                    new_edge: *new_edge,
                                    original_edge: [from, to],
                                };
                                let diff = pair.diff();
                                (i, j, pair, diff)
                            })
                            .max_by_key(|(_, _, _, diff)| *diff)
                    })
                    .max_by_key(|(_, _, _, diff)| {
                        if is_annealing {
                            T::from_f64(
                                diff.to_f64().unwrap()
                                    * rng.gen_range(simulated_annealing_noise_min..=1.),
                            )
                            .unwrap()
                        } else {
                            *diff
                        }
                    })
                {
                    if diff <= T::zero() {
                        *stuck_by_operator.get_mut(&operator).unwrap() = true;
                        continue;
                    } else {
                        *stuck_by_operator.get_mut(&operator).unwrap() = false;
                    }

                    if pair.new_edge[0] != pair.original_edge[0] {
                        let reversed_prefix =
                            current[..=i].iter().rev().copied().collect::<Vec<_>>();
                        current[..=i]
                            .iter_mut()
                            .zip(reversed_prefix.iter())
                            .for_each(|(dest, origin)| *dest = *origin);
                    }
                    if pair.new_edge[1] != pair.original_edge[1] {
                        let reversed_suffix =
                            current[j..].iter().rev().copied().collect::<Vec<_>>();
                        current[j..]
                            .iter_mut()
                            .zip(reversed_suffix.iter())
                            .for_each(|(dest, origin)| *dest = *origin);
                    }

                    current_sum = current_sum - diff;
                    dbg!(idx, diff, current_sum, best_sum);

                    if current_sum < best_sum {
                        best = current.clone();
                        best_sum = current_sum;
                    }
                }
            }
        }
        if is_annealing {
            simulated_annealing_noise_min =
                (simulated_annealing_noise_min + 2. / ITERATIONS as f64).clamp(0., 1.);
        }
    }

    best
}

#[cfg(test)]
#[test]
fn tsp_is_correct_for_trivial_case() {
    let vertices = [[0, 0], [1, 1], [2, 2]];
    let tree = [[[0, 0], [1, 1]], [[1, 1], [2, 2]]];

    let path = approximate_tsp_with_mst(&vertices, &tree);
    let length: i64 = path
        .iter()
        .zip(path.iter().skip(1))
        .map(|(from, to)| abs_distance_squared(*from, *to))
        .sum();
    assert_eq!(length, 4);
}

#[cfg(test)]
#[test]
fn tsp_is_correct_for_nontrivial_case() {
    let vertices: [[i64; 2]; 5] = [[24, 28], [371, 33], [235, 72], [157, 509], [156, 194]];
    let tree = [
        [[24, 28], [156, 194]],
        [[156, 194], [235, 72]],
        [[235, 72], [371, 33]],
        [[156, 194], [157, 509]],
    ];
    let path = approximate_tsp_with_mst(&vertices, &tree);
    let length: i64 = path
        .iter()
        .zip(path.iter().skip(1))
        .map(|(from, to)| abs_distance_squared(*from, *to))
        .sum();
    assert_eq!(length, 210680);
    assert_eq!(&path, &vertices);
}

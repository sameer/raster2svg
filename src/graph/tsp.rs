use crate::math::abs_distance_squared;
use log::*;
use num_traits::{FromPrimitive, PrimInt, Signed};
use rand::{distributions::Standard, prelude::Distribution, thread_rng, Rng};
use rayon::prelude::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::{
    collections::VecDeque,
    fmt::Debug,
    hash::{BuildHasherDefault, Hash},
    iter::Sum,
};

#[derive(PartialEq, Eq, Debug)]
struct Edge<T: Debug>([[T; 2]; 2]);

impl<T: PrimInt + Signed + Eq + PartialEq + PartialOrd + Ord + Debug> Edge<T> {
    fn length(&self) -> T {
        abs_distance_squared(self.0[0], self.0[1])
    }
}

impl<T: PrimInt + Signed + Eq + PartialEq + PartialOrd + Ord + Debug> PartialOrd for Edge<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PrimInt + Signed + Eq + PartialEq + PartialOrd + Ord + Debug> Ord for Edge<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.length().cmp(&other.length())
    }
}

/// Approximate an open loop TSP solution by way of greedy branch elimination + local improvement.
///
/// <http://cs.uef.fi/sipu/pub/applsci-11-00177.pdf>
///
/// See the two source code comments prefixed with `NOTE:`
/// for how to use the non-greedy all pairs branch elimination
/// variant, which takes significantly longer.
///
/// TODO: consider using the Christofides algorithm which has a bound on the length of the worst path.
/// The initial solution of greedy branch elimination is actually quite bad.
pub fn approximate_tsp_with_mst<
    T: PrimInt
        + Signed
        + FromPrimitive
        + Eq
        + PartialEq
        + PartialOrd
        + Ord
        + Hash
        + Debug
        + Sum
        + Send
        + Sync,
>(
    vertices: &[[T; 2]],
    tree: &[[[T; 2]; 2]],
) -> Vec<[T; 2]> {
    if vertices.len() <= 1 {
        return vec![];
    } else if vertices.len() == 2 {
        return vertices.to_vec();
    }
    let path = approximate_tsp_with_mst_greedy(vertices, tree);
    local_improvement_with_tabu_search::<_, false>(&path)
}

fn approximate_tsp_with_mst_greedy<
    T: PrimInt
        + Signed
        + FromPrimitive
        + Eq
        + PartialEq
        + PartialOrd
        + Ord
        + Hash
        + Debug
        + Sum
        + Send
        + Sync,
>(
    vertices: &[[T; 2]],
    tree: &[[[T; 2]; 2]],
) -> Vec<[T; 2]> {
    let mut adjacency_map: Vec<HashSet<_>> =
        vec![HashSet::with_capacity_and_hasher(1, BuildHasherDefault::default()); vertices.len()];
    {
        let vertex_to_index = vertices
            .iter()
            .copied()
            .enumerate()
            .map(|(i, vertex)| (vertex, i))
            .collect::<HashMap<_, _>>();
        tree.iter().for_each(|edge| {
            adjacency_map[vertex_to_index[&edge[0]]].insert(vertex_to_index[&edge[1]]);
            adjacency_map[vertex_to_index[&edge[1]]].insert(vertex_to_index[&edge[0]]);
        });
    }

    let mut branch_list = adjacency_map
        .iter()
        .enumerate()
        .filter(|(_, adjacencies)| adjacencies.len() >= 3)
        .flat_map(|(branch, adjacencies)| {
            adjacencies
                .iter()
                .copied()
                .map(move |adjacency| (branch, adjacency))
        })
        .collect::<Vec<_>>();
    branch_list
        .sort_by_cached_key(|(branch, adjacency)| Edge([vertices[*branch], vertices[*adjacency]]));

    while let Some((branch, disconnected_node)) = branch_list.pop() {
        debug!(
            "Approximation progress: {} remaining branches",
            branch_list.len()
        );
        // No longer a branching vertex
        if adjacency_map[branch].len() <= 2 {
            continue;
        }
        // This branching edge doesn't exist anymore.
        // The other end is a branch that was already processed.
        // that has already been processed.
        if !adjacency_map[branch].contains(&disconnected_node) {
            continue;
        }

        // Remove edge
        adjacency_map[branch].remove(&disconnected_node);
        adjacency_map[disconnected_node].remove(&branch);

        // Now there are two disconnected trees,
        // do a BFS to find the leaves in both.
        let (disconnected_tree_leaves, branch_tree_leaves) = {
            let mut disconnected_leaves = vec![];
            let mut branch_leaves = vec![];

            let mut disconnected_bfs = VecDeque::from([((disconnected_node, branch))]);
            let mut branch_bfs = VecDeque::from([(branch, disconnected_node)]);
            loop {
                let it = disconnected_bfs
                    .pop_front()
                    .map(|state| (state, &mut disconnected_leaves, &mut disconnected_bfs))
                    .into_iter()
                    .chain(
                        branch_bfs
                            .pop_front()
                            .map(|state| (state, &mut branch_leaves, &mut branch_bfs))
                            .into_iter(),
                    );

                for ((head, source), leaves, bfs) in it {
                    // Handles the first vertex
                    if adjacency_map[head].len() <= 1 {
                        leaves.push(head);
                    }
                    let non_leaf_adjacencies = adjacency_map[head]
                        .iter()
                        .copied()
                        // Optimization: we only need to make sure no backtracking happens since this
                        // is a tree.
                        .filter(|adj| *adj != source)
                        .filter_map(|adj| {
                            let adj_adj = &adjacency_map[adj];
                            if adj_adj.len() <= 1 {
                                leaves.push(adj);
                                debug_assert_eq!(*adj_adj.iter().next().unwrap(), head);
                                None
                            } else {
                                Some((adj, head))
                            }
                        });
                    bfs.extend(non_leaf_adjacencies);
                }

                if disconnected_bfs.is_empty() && branch_bfs.is_empty() {
                    break (disconnected_leaves, branch_leaves);
                }
            }
        };

        // Pick the shortest possible link between two leaves that would reconnect the trees
        let (disconnected_tree_leaf, branch_tree_leaf) = disconnected_tree_leaves
            .into_par_iter()
            .flat_map(|i| {
                branch_tree_leaves
                    .clone()
                    .into_par_iter()
                    .map(move |j| (i, j))
            })
            .min_by_key(|(i, j)| abs_distance_squared(vertices[*i], vertices[*j]))
            .unwrap();

        // Connect leaves
        adjacency_map[branch_tree_leaf].insert(disconnected_tree_leaf);
        adjacency_map[disconnected_tree_leaf].insert(branch_tree_leaf);
    }

    // Extract path from the adjacency list
    let mut path = Vec::with_capacity(vertices.len());
    let (first_vertex, adjacencies) = adjacency_map
        .iter()
        .enumerate()
        .find(|(_, adjacencies)| adjacencies.len() == 1)
        .expect("path always has a first vertex");
    path.push(first_vertex);
    path.push(*adjacencies.iter().next().unwrap());

    // The number of edges in an open loop TSP path is equal to the number of vertices - 1
    while path.len() < vertices.len() {
        let last_vertex = *path.last().unwrap();
        let second_to_last_vertex = *path.iter().rev().nth(1).unwrap();
        let next_vertex = adjacency_map[last_vertex]
            .iter()
            .find(|adjacency| **adjacency != second_to_last_vertex)
            .unwrap();
        path.push(*next_vertex);
    }

    path.into_iter().map(|i| vertices[i]).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Operator {
    /// Move vertex between other vertices
    Relocate,
    /// Swap vertices between two edges
    ///
    /// This does not implement the paper's version of 2-opt where
    /// the terminal nodes are linked to dummy terminals.
    TwoOpt,
    /// Change the beginning and/or end of the path by swapping an edge
    ///
    /// In the words of the paper:
    /// > Link swap is a special case of 3–opt and relocate operator, but as the size of the neighborhood is linear,
    /// > it is a faster operation than both 3–opt and relocate operator.
    LinkSwap,
}

impl Operator {
    const NUM_OPERATORS: usize = 3;
}

impl Distribution<Operator> for Standard {
    /// Based on productivity results in the paper, link swap is given a chance of 50% while relocate and 2-opt have 25% each
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Operator {
        match rng.gen_range(0..=3) {
            0 => Operator::Relocate,
            1 => Operator::TwoOpt,
            2 | 3 => Operator::LinkSwap,
            _ => unreachable!(),
        }
    }
}

/// Local improvement of an open loop TSP solution using the relocate, disentangle, 2-opt, and link swap operators.
/// Tabu search is used to avoid getting stuck early in local minima.
///
/// <https://www.mdpi.com/2076-3417/9/19/3985/pdf>
///
fn local_improvement_with_tabu_search<
    T: PrimInt
        + Signed
        + FromPrimitive
        + Eq
        + PartialEq
        + PartialOrd
        + Ord
        + Hash
        + Debug
        + Sum
        + Send
        + Sync,
    const SHOULD_SAMPLE: bool,
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
    let mut current_distances = current
        .iter()
        .zip(current.iter().skip(1))
        .map(|(from, to)| abs_distance_squared(*from, *to))
        .collect::<Vec<_>>();
    let mut current_sum = best_sum;

    let sample_count = (current.len() as f64 * 0.001) as usize;
    const ITERATIONS: usize = 20000;

    let mut rng = thread_rng();

    /// 10% of the past moves are considered tabu
    const TABU_FRACTION: f64 = 0.1;
    let tabu_capacity = (current.len() as f64 * TABU_FRACTION) as usize;
    let mut tabu: VecDeque<usize> = VecDeque::with_capacity(tabu_capacity);
    let mut tabu_set: HashSet<usize> = HashSet::default();
    tabu_set.reserve(tabu_capacity);

    let mut stuck_operators = HashSet::with_capacity_and_hasher(3, BuildHasherDefault::default());

    for idx in 0..ITERATIONS {
        if stuck_operators.len() == Operator::NUM_OPERATORS && !SHOULD_SAMPLE {
            if tabu.is_empty() {
                info!("Stuck, no more local improvements can be made");
                break;
            } else {
                // Try to unstick by clearing tabu
                tabu.clear();
                tabu_set.clear();
                stuck_operators.clear();
            }
        }

        let operator: Operator = rng.gen();

        match operator {
            // O(v^2)
            Operator::Relocate => {
                // Which i should be considered for relocation (i in [1, N-2])
                let relocates = if SHOULD_SAMPLE {
                    0..sample_count
                } else {
                    1..current.len().saturating_sub(1)
                }
                .into_par_iter()
                .map(|i| {
                    if SHOULD_SAMPLE {
                        thread_rng().gen_range(1..current.len().saturating_sub(1))
                    } else {
                        i
                    }
                });

                // move i between j and j+1
                let best = relocates
                    .filter(|i| !tabu_set.contains(i))
                    .flat_map(|i| {
                        // pre-computed to save time,
                        // relies on triangle property to avoid overflow:
                        // distance from i-1 --> i --> i+1 is strictly greater than
                        // distance from i-1 --> i+1
                        let unlink_i_improvement = (current_distances[i - 1]
                            + current_distances[i])
                            - abs_distance_squared(current[i - 1], current[i + 1]);
                        // j must be in [0, i-2] U [i+1, N-1] for the move to be valid
                        (0..i.saturating_sub(1))
                            .into_par_iter()
                            .chain(
                                (i.saturating_add(1)..current.len().saturating_sub(1))
                                    .into_par_iter(),
                            )
                            .map(move |j| (i, j, unlink_i_improvement))
                    })
                    .map(|(i, j, unlink_i_improvement)| {
                        // Old distances - pre-computed new distance: (j=>j+1, i-1=>i, i=>i+1) - i-1=>i+1
                        let positive_diff = current_distances[j] + unlink_i_improvement;
                        // New distances: j=>i, i=>j+1
                        let negative_diff = abs_distance_squared(current[j], current[i])
                            + abs_distance_squared(current[i], current[j + 1]);
                        (i, j, positive_diff.saturating_sub(negative_diff))
                    })
                    .max_by_key(|(.., diff)| *diff);

                if let Some((i, j, diff)) = best {
                    if diff <= T::zero() {
                        stuck_operators.insert(operator);
                        continue;
                    } else {
                        stuck_operators.clear();
                    }
                    let vertex = current[i];
                    if j < i {
                        // j is before i in the path
                        // shift to the right and insert vertex
                        for idx in (j + 1..i).rev() {
                            current[idx + 1] = current[idx];
                        }
                        current[j + 1] = vertex;
                        tabu.push_back(j + 1);
                        tabu_set.insert(j + 1);
                    } else {
                        // j is after in the path
                        // shift to the left and insert vertex
                        for idx in i..j {
                            current[idx] = current[idx + 1];
                        }
                        current[j] = vertex;
                        tabu.push_back(j);
                        tabu_set.insert(j);
                    }
                } else {
                    stuck_operators.insert(operator);
                    continue;
                }
            }
            // O(v^2)
            Operator::TwoOpt => {
                let edges = if SHOULD_SAMPLE {
                    0..sample_count
                } else {
                    0..current.len().saturating_sub(1)
                }
                .into_par_iter()
                .map(|i| {
                    if SHOULD_SAMPLE {
                        thread_rng().gen_range(0..current.len().saturating_sub(1))
                    } else {
                        i
                    }
                })
                .map(|i| (i, i + 1));
                // permute the points of the this and other edges
                let best = edges
                    .flat_map(|(i, j)| {
                        (1..i)
                            .into_par_iter()
                            .map(move |other_j| {
                                let other_i = other_j - 1;
                                (other_i, other_j)
                            })
                            // Note that other and (i,j) are swapped so that
                            // the first edge is always before the second edge
                            .map(move |other| (other, (i, j)))
                            // If we aren't sampling, it is pointless to use these because
                            // we already saw them.
                            .filter(|_| SHOULD_SAMPLE)
                            .chain(
                                (j.saturating_add(2)..current.len())
                                    .into_par_iter()
                                    .map(move |other_j| {
                                        let other_i = other_j - 1;
                                        (other_i, other_j)
                                    })
                                    .map(move |other| ((i, j), other)),
                            )
                    })
                    .filter(|(this, other)| {
                        !tabu_set.contains(&this.1) && !tabu_set.contains(&other.0)
                    })
                    // Examine all edge pairs
                    .map(|(this, other)| {
                        (
                            this,
                            other,
                            // Lose this.0=>this.1 & other.0=>other.1
                            // Gain this.0=>other.0 & this.1=>other.1
                            (current_distances[this.0] + current_distances[other.0])
                                .saturating_sub(
                                    abs_distance_squared(current[this.0], current[other.0])
                                        + abs_distance_squared(current[this.1], current[other.1]),
                                ),
                        )
                    })
                    .max_by_key(|(.., diff)| *diff);

                if let Some((this, other, diff)) = best {
                    if diff <= T::zero() {
                        stuck_operators.insert(operator);
                        continue;
                    } else {
                        stuck_operators.clear();
                    }
                    let tabu_add = [this.1, other.0];
                    tabu.extend(tabu_add);
                    tabu_set.extend(tabu_add);
                    // Reversing in-place maintains inner links, but swaps outer links
                    current[this.1..=other.0].reverse();
                } else {
                    stuck_operators.insert(operator);
                    continue;
                }
            }
            // O(v) 3*(v-1)
            Operator::LinkSwap => {
                let first = *current.first().unwrap();
                let last = *current.last().unwrap();

                // Change from=>to to one of from=>last, first=>to, or first=>last
                let best = (2..current.len().saturating_sub(1))
                    .into_par_iter()
                    .map(|j| {
                        let i = j - 1;
                        (i, j)
                    })
                    .filter(|(i, j)| !tabu_set.contains(i) && !tabu_set.contains(j))
                    .map(|(i, j)| {
                        let from = current[i];
                        let to = current[j];
                        [[from, last], [first, to], [first, last]]
                            .iter()
                            .map(|new_edge| {
                                let diff = current_distances[i]
                                    .saturating_sub(abs_distance_squared(new_edge[0], new_edge[1]));
                                (i, j, [from, to], *new_edge, diff)
                            })
                            .max_by_key(|(.., diff)| *diff)
                            .expect("array is not empty")
                    })
                    .max_by_key(|(.., diff)| *diff);

                if let Some((i, j, original_edge, new_edge, diff)) = best {
                    if diff <= T::zero() {
                        stuck_operators.insert(operator);
                        continue;
                    } else {
                        stuck_operators.clear();
                    }

                    // Change from=>to to first=>____
                    if new_edge[0] != original_edge[0] {
                        tabu.push_back(i);
                        tabu_set.insert(i);
                        current[..=i].reverse();
                    }
                    // Change from=>to to ____=>last
                    if new_edge[1] != original_edge[1] {
                        tabu.push_back(j);
                        tabu_set.insert(j);
                        current[j..].reverse();
                    }
                } else {
                    stuck_operators.insert(operator);
                    continue;
                }
            }
        }

        let prev_sum = current_sum;
        current_distances = current
            .iter()
            .zip(current.iter().skip(1))
            .map(|(from, to)| abs_distance_squared(*from, *to))
            .collect::<Vec<_>>();
        current_sum = current_distances.iter().copied().sum::<T>();

        debug_assert_eq!(
            current.len(),
            current.iter().copied().collect::<HashSet<_>>().len()
        );
        assert!(
            prev_sum > current_sum,
            "operator = {:?} prev = {:?} current = {:?}",
            operator,
            prev_sum,
            current_sum
        );

        if current_sum < best_sum {
            best = current.clone();
            best_sum = current_sum;
        }

        info!(
            "Iteration {}/{} (best: {:?}, tabu: {}/{}, len: {})",
            idx,
            ITERATIONS,
            best_sum,
            tabu.len(),
            tabu_capacity,
            current.len(),
        );

        while tabu.len() > tabu_capacity {
            tabu_set.remove(&tabu.pop_front().unwrap());
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
}

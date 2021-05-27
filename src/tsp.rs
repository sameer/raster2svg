use fxhash::FxHashMap as HashMap;

use crate::abs_distance_squared;
use log::debug;
/// Approximate a TSP solution by way of all-pairs branch elimination
/// http://cs.uef.fi/sipu/pub/applsci-11-00177.pdf
pub fn approximate_tsp_with_mst(
    vertices: &[[i64; 2]],
    tree: &[[[i64; 2]; 2]],
) -> Vec<[[i64; 2]; 2]> {
    if vertices.len() <= 2 {
        return vec![];
    }
    let mut adjacency_map: HashMap<[i64; 2], Vec<[i64; 2]>> = HashMap::default();
    tree.iter().for_each(|edge| {
        adjacency_map.entry(edge[0]).or_default().push(edge[1]);
        adjacency_map.entry(edge[1]).or_default().push(edge[0]);
    });
    // let mut path = tree.to_owned();
    loop {
        if let Some((branch, disconnected_node)) = adjacency_map
            .iter()
            .map(|(vertex, edges)| (*vertex, edges.clone()))
            .filter(|(_, vertex_edges)| vertex_edges.len() >= 3)
            .map(|(branch, adjacencies)| {
                adjacencies
                    .iter()
                    .map(move |adjacency| (branch, *adjacency))
                    .collect::<Vec<_>>()
            })
            .flatten()
            .max_by_key(|(vertex, adjacency)| abs_distance_squared(*vertex, *adjacency))
        {
            // Remove edge
            adjacency_map
                .get_mut(&branch)
                .unwrap()
                .retain(|adjacency| *adjacency != disconnected_node);
            adjacency_map
                .get_mut(&disconnected_node)
                .unwrap()
                .retain(|adjacency| *adjacency != branch);

            // Now there are two disconnected trees

            // Determine membership
            let mut branch_tree_visited: HashMap<[i64; 2], bool> =
                vertices.iter().map(|v| (*v, false)).collect();
            *branch_tree_visited.get_mut(&branch).unwrap() = true;
            let mut branch_dfs = vec![branch];
            while let Some(head) = branch_dfs.pop() {
                for adjacency in adjacency_map.get(&head).unwrap() {
                    if !branch_tree_visited.get(adjacency).unwrap() {
                        *branch_tree_visited.get_mut(adjacency).unwrap() = true;
                        branch_dfs.push(*adjacency);
                    }
                }
            }

            // Find leaves in the two
            // Pick the shortest possible link between two leaves that would reconnect them
            let mut branch_tree_leaves = branch_tree_visited
                .iter()
                .filter_map(|(k, v)| if *v { Some(*k) } else { None })
                .filter(|branch_tree_vertex| {
                    adjacency_map.get(branch_tree_vertex).unwrap().len() <= 1
                })
                .collect::<Vec<_>>();

            let disconnected_tree_leaves = branch_tree_visited
                .iter()
                .filter_map(|(k, v)| if *v { None } else { Some(*k) })
                .filter(|disconnected_tree_vertex| {
                    adjacency_map.get(disconnected_tree_vertex).unwrap().len() <= 1
                })
                .collect::<Vec<_>>();
            // if branch_tree_leaves.is_empty() || disconnected_tree_leaves.is_empty() {
            //     println!(
            //         "This should not be possible: {:?} {:?}",
            //         branch_tree_visited
            //             .iter()
            //             .filter_map(|(k, v)| if *v { Some(*k) } else { None })
            //             .collect::<Vec<_>>(),
            //         branch_tree_visited
            //             .iter()
            //             .filter_map(|(k, v)| if *v { None } else { Some(*k) })
            //             .collect::<Vec<_>>(),
            //     );
            // }
            // println!("disconnect: {:?}", [branch, disconnected_node]);
            // println!("v: {:?} t: {:?}", &vertices, &tree);

            let (branch_tree_leaf, disconnected_tree_leaf) = branch_tree_leaves
                .drain(..)
                .map(|branch_tree_leaf| {
                    disconnected_tree_leaves
                        .iter()
                        .map(move |disconnected_tree_leaf| {
                            (branch_tree_leaf, disconnected_tree_leaf)
                        })
                })
                .flatten()
                .min_by_key(|(branch_tree_leaf, disconnected_tree_leaf)| {
                    abs_distance_squared(*branch_tree_leaf, **disconnected_tree_leaf)
                })
                .unwrap();
            // let (edge_idx, _) = path
            //     .iter()
            //     .enumerate()
            //     .find(|(_, path_edge)| path_edge == long_edge)
            //     .unwrap();

            // pick a leaf edge for the node being disconnected
            adjacency_map
                .get_mut(&branch_tree_leaf)
                .unwrap()
                .push(*disconnected_tree_leaf);
            adjacency_map
                .get_mut(disconnected_tree_leaf)
                .unwrap()
                .push(branch_tree_leaf);
        } else {
            break;
        }
    }

    // MST is now a "path", but needs to be made into a round trip
    let leaves: Vec<_> = adjacency_map
        .iter()
        .filter(|(_, vertex_edges)| vertex_edges.len() == 1)
        .map(|(vertex, _)| *vertex)
        .collect();
    if leaves.len() != 2 {
        unreachable!();
    }

    adjacency_map
        .get_mut(&leaves[0])
        .unwrap()
        .push(leaves[1].clone());
    adjacency_map
        .get_mut(&leaves[1])
        .unwrap()
        .push(leaves[0].clone());

    // There's a "path" but it needs to be extracted from the adjacency list
    let mut path: Vec<[[i64; 2]; 2]> = Vec::with_capacity(vertices.len());
    path.push([
        *vertices.first().unwrap(),
        *adjacency_map
            .get(vertices.first().unwrap())
            .unwrap()
            .first()
            .unwrap(),
    ]);

    // The number of edges in a Hamiltonian cycle is equal to the number of vertices
    while path.len() < vertices.len() {
        let [second_to_last_vertex, last_vertex]: [[i64; 2]; 2] = *path.last().unwrap();
        let next_vertex = adjacency_map
            .get(&last_vertex)
            .unwrap()
            .iter()
            .find(|adjacency| **adjacency != second_to_last_vertex)
            .unwrap();
        path.push([last_vertex, *next_vertex]);
    }
    path
}

#[cfg(test)]
#[test]
fn tsp_is_correct_for_trivial_case() {
    let vertices = [[0, 0], [1, 1], [2, 2]];
    let tree = [[[0, 0], [1, 1]], [[1, 1], [2, 2]]];

    assert_eq!(
        approximate_tsp_with_mst(&vertices, &tree),
        &[[[0, 0], [1, 1]], [[1, 1], [2, 2]], [[2, 2], [0, 0]]]
    );
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
    let distances = vertices
        .iter()
        .map(|vertex| {
            vertices
                .iter()
                .map(move |other_vertex| [vertex, other_vertex])
                .map(|[a, b]| {
                    (((a[0] - b[0]).pow(2) + (a[1] - b[1]).pow(2)) as f64)
                        .sqrt()
                        .round()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    println!("{:?}", distances);
    println!(
        "{:?}",
        approximate_tsp_with_mst(&vertices, &tree)
            .iter()
            .map(|[a, b]| {
                (((a[0] - b[0]).pow(2) + (a[1] - b[1]).pow(2)) as f64)
                    .sqrt()
                    .round()
            })
            .sum::<f64>()
    );
}

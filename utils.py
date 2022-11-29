import json
import os
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from tqdm import tqdm

warnings.filterwarnings('ignore')


def load_singleton_graphs_from_TUDataset(root: str,
                                         dataset: str,
                                         node_attr: str = 'x',
                                         use_degree: bool = False) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Use the Pytorch Geometric (PyG) loader to download the graph dataset from the TUDataset repo.
    The raw graphs from PyG are saved in `root`.

    A singleton graph is created by summing all the node attributes `x` of the original graph.
    The corresponding class of each graph is also retrieved from TUDataset dataset.

    Args:
        root: Path where to save the raw graph dataset
        dataset: Name of the graph dataset to load
        node_attr:
        use_degree
    Returns:
        List of the loaded `np.ndarray` singleton graphs and `np.ndarray` of the corresponding class of each graph
    """
    dataset = TUDataset(root=root, name=dataset)

    tmp_graph = dataset[0]
    is_graph_labelled = node_attr in tmp_graph.keys

    # Convert the PyG graphs into singleton graphs
    graphs = []
    graph_labels = []
    for graph in tqdm(dataset, desc='Convert graph to singleton'):

        if not is_graph_labelled:
            # Create graph with dummy node vector
            graph = Data(x=torch.tensor(np.ones((graph.num_nodes, 1))),
                         y=graph.y,
                         edge_index=graph.edge_index)

        if use_degree:
            degrees = degree(graph.edge_index[0], graph.num_nodes)
            node_feature = torch.mul(graph.x, degrees.view(-1, 1))
        else:
            node_feature = graph.x

        graphs.append(np.array(node_feature.sum(axis=0)))
        graph_labels.append(int(graph.y))

    graph_cls = np.array(graph_labels)

    return graphs, graph_cls


def save_cv_predictions(file_results: str, cv_predictions: List) -> None:
    """
    Save the list of cross-validation scores

    Args:
        file_results:
        cv_predictions:

    Returns:

    """
    with open(file_results, 'w') as write_file:
        json.dump(cv_predictions, write_file, indent=4)


def get_folder_results(dataset: str, classifier: str) -> None:
    path = os.path.join('./results',
                        dataset,
                        classifier)

    return path


def get_file_results(folder_results: str,
                     dataset: str,
                     classifier: str,
                     use_degree: bool) -> str:
    """

    Args:
        folder_results:
        dataset:
        classifier:
        use_degree:

    Returns:

    """
    if not folder_results:
        root = get_folder_results(dataset, classifier)
    else:
        root = folder_results

    Path(root).mkdir(parents=True, exist_ok=True)

    filename = f'results{"_use_degree" if use_degree else ""}.json'

    return os.path.join(root, filename)

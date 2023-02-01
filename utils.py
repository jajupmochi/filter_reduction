import json
import os
import warnings
from os.path import join
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch_geometric.utils as tutils
from sklearn.exceptions import UndefinedMetricWarning
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


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
    is_graph_lbl_empty = tmp_graph.x.size(1) == 0 if is_graph_labelled else True

    # Convert the PyG graphs into singleton graphs
    graphs = []
    graph_labels = []
    for graph in tqdm(dataset, desc='Convert graph to singleton'):

        if not is_graph_labelled or is_graph_lbl_empty:
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


def get_folder_results(dataset: str, classifier: str) -> str:
    path = os.path.join('./results',
                        dataset,
                        classifier)

    return path


def get_file_results(folder_results: str,
                     dataset: str,
                     classifier: str,
                     use_degree: bool,
                     remove_node_labels: bool) -> str:
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

    Path(root).mkdir(parents=True,
                     exist_ok=True)

    degree = '_use_degree' if use_degree else ''
    node_labels = '_without_node_labels' if remove_node_labels else ''

    filename = f'results{degree}{node_labels}.json'

    return os.path.join(root, filename)


def get_folder_distances(folder_results: str,
                         dataset: str,
                         classifier: str,
                         remove_node_attr: bool,
                         use_degree: bool):
    if not folder_results:
        folder_results = get_folder_results(dataset, classifier)

    degree = '_use_degree' if use_degree else ''
    node_attr = '_without_node_labels' if remove_node_attr else ''

    folder_distances = join(folder_results, f'distances{degree}{node_attr}')

    Path(folder_distances).mkdir(parents=True, exist_ok=True)

    return folder_distances


def load_graphs(root: str,
                dataset: str,
                remove_node_attr: bool,
                use_degree: bool,
                node_attr: str = 'x') -> Tuple[List[nx.Graph], np.ndarray]:
    """

    Args:
        root:
        dataset:
        remove_node_attr:
        use_degree:
        node_attr:

    Returns:

    """
    dataset = TUDataset(root=root, name=dataset)

    tmp_graph = dataset[0]
    is_graph_labelled = node_attr in tmp_graph.keys
    is_graph_lbl_empty = tmp_graph.x.size(1) == 0 if is_graph_labelled else True

    nx_graphs = []
    graph_labels = []
    for graph in tqdm(dataset, desc='Load Graphs'):
        if not is_graph_labelled or is_graph_lbl_empty:
            # Create graph with dummy node vector
            degrees = degree(graph.edge_index[0], graph.num_nodes)
            graph = Data(x=degrees,
                         y=graph.y,
                         edge_index=graph.edge_index)
        if remove_node_attr:
            if use_degree:
                degrees = degree(graph.edge_index[0], graph.num_nodes)
                graph.x = degrees
            else:
                graph.num_nodes = graph.x.shape[0]
                graph.x = torch.ones(graph.num_nodes, dtype=torch.float32)

        nx_graph = tutils.to_networkx(graph,
                                      node_attrs=[node_attr],
                                      to_undirected=True)
        nx_graphs.append(nx_graph)
        graph_labels.append(int(graph.y))

    graph_cls = np.array(graph_labels)
    return nx_graphs, graph_cls


def make_hashable_attr(nx_graphs: List[nx.Graph],
                       node_attr: str = 'x') -> None:
    """
    Transform the node attribute `x` to str to be hashable.
    Args:
        nx_graphs:
        node_attr:
    Returns:
    """
    for nx_graph in nx_graphs:
        for idx_node, data_node in nx_graph.nodes(data=True):
            str_data = str(data_node[node_attr])
            nx_graph.nodes[idx_node][node_attr] = str_data


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

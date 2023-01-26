import argparse
import logging
from typing import List

import numpy as np
import torch
import torch.optim
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from models.DGCNN import DGCNN
from utils import load_graphs, get_file_results, save_cv_predictions

logging.captureWarnings(True)

NODE_ATTRIBUTE = 'x'


def _get_k(graphs: List[Data], percentile: float = 0.6) -> int:
    nbr = [graph.num_nodes for graph in graphs]
    sorted_nbr = sorted(nbr)

    for i, x in enumerate(sorted_nbr):
        if (1 + i) / len(sorted_nbr) >= percentile:
            return x

    return False


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def gnn_classification(root_dataset: str,
                       dataset: str,
                       use_degree: bool,
                       classifier: str,
                       n_trials: int,
                       n_outer_cv: int,
                       n_inner_cv: int,
                       folder_results: str):
    seed_everything(7)

    graphs, labels = load_graphs(root=root_dataset,
                                 dataset=dataset,
                                 remove_node_attr=False,
                                 node_attr=NODE_ATTRIBUTE,
                                 use_degree=use_degree)

    graphs = [from_networkx(graph, [NODE_ATTRIBUTE]) for graph in graphs]
    y = np.array(labels).astype(np.int64)
    ks = [_get_k(graphs, percentile=perc) for perc in [0.6, 0.9]]
    X = torch.arange(len(graphs)).long()

    max_epochs = 800
    batch_size = 50

    net = NeuralNetClassifier(
        module=DGCNN,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        batch_size=batch_size,
        max_epochs=max_epochs,
        module__dataset=graphs,
        module__dim_features=graphs[0].num_features,
        module__dim_target=len(set(y)),
        module__hidden_dense_dim=128,
        callbacks=[EarlyStopping(monitor='valid_loss', patience=max_epochs // 2, lower_is_better=True)]
    )

    params = {
        'lr': [10 ** -4, 10 ** -5],
        'module__embedding_dim': [32, 64],
        'module__num_layers': [2, 3, 4],
        'module__k': ks,
    }
    scoring = {'acc': 'accuracy'}
    file_results = get_file_results(folder_results,
                                    dataset,
                                    classifier,
                                    use_degree,
                                    False)
    trial_predictions = []
    for c_seed in range(n_trials):
        outer_cv = StratifiedKFold(n_splits=n_outer_cv, shuffle=True, random_state=c_seed)
        inner_cv = StratifiedKFold(n_splits=n_inner_cv, shuffle=True, random_state=c_seed)
        gs = GridSearchCV(estimator=net,
                          param_grid=params,
                          refit=True,
                          cv=inner_cv,
                          scoring='accuracy',
                          n_jobs=5)
        test_predictions = cross_validate(gs,
                                          X,
                                          y,
                                          cv=outer_cv,
                                          scoring=scoring,
                                          n_jobs=2)

        dict_cv_predictions = {k: v.tolist() for k, v in dict(test_predictions).items()}
        trial_predictions.append(dict_cv_predictions)
        save_cv_predictions(file_results, trial_predictions)


def main(args):
    """

    Args:
        args:

    Returns:

    """
    gnn_classification(**vars(args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph classification with singleton graphs')

    parser.add_argument('--root-dataset',
                        default='./data',
                        help='Folder where to save the raw dataset')
    parser.add_argument('--dataset',
                        required=True,
                        type=str,
                        help='Name of the dataset')

    parser.add_argument('--remove-node-attr',
                        action='store_true',
                        help='remove the node attributes if set to false')
    parser.add_argument('--use-degree',
                        action='store_true',
                        help='Use the degree of the nodes during the graph reduction process')

    parser.add_argument('--classifier',
                        default='gnn',
                        help='Classification method to use')

    parser.add_argument('--n-trials',
                        default=10,
                        type=int,
                        help='Number of times to execute the cross-validation')
    parser.add_argument('--n-outer-cv',
                        default=10,
                        type=int,
                        help='Number of times to execute the cross-validation')
    parser.add_argument('--n-inner-cv',
                        default=5,
                        type=int,
                        help='Number of times to execute the cross-validation')

    parser.add_argument('--n-core-inner',
                        default=5,
                        type=int,
                        help='Number of cores used in the inner loop')
    parser.add_argument('--n-core-outer',
                        default=5,
                        type=int,
                        help='Number of cores used in the outer loop')

    parser.add_argument('--folder-results',
                        type=str,
                        help='Folder where to save the results')

    args = parser.parse_args()

    main(args)

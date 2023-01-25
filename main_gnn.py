import argparse
import logging

import numpy as np
import torch
import torch.optim
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.datasets import TUDataset
from models.DGCNN import DGCNN
from utils import load_graphs

logging.captureWarnings(True)

NODE_ATTRIBUTE = 'x'


def _get_k(X: Data, percentile: float = 0.6) -> int:
    nbr = [graph.num_nodes for graph in X]
    sorted_nbr = sorted(nbr)

    for i, x in enumerate(sorted_nbr):
        if (1 + i) / len(sorted_nbr) >= percentile:
            return x

    return False


def gnn_classification(root_dataset: str,
                       dataset: str,
                       use_degree: bool,
                       classifier: str,
                       n_trials: int,
                       n_outer_cv: int,
                       n_inner_cv: int,
                       folder_results: str):
    graphs, labels = load_graphs(root=root_dataset,
                                 dataset=dataset,
                                 remove_node_attr=False,
                                 node_attr=NODE_ATTRIBUTE,
                                 use_degree=use_degree)

    dataset = [from_networkx(graph, [NODE_ATTRIBUTE]) for graph in graphs]
    y = np.array(labels).astype(np.int64)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = NeuralNetClassifier(
        module=DGCNN,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        batch_size=50,
        max_epochs=1000,
        module__dataset=dataset,
        module__dim_features=dataset[0].num_features,
        module__dim_target=len(set(y)),
        module__hidden_dense_dim=128,
        # verbose=0,
    )
    ks = [_get_k(dataset, percentile=perc) for perc in [0.6, 0.9]]
    early_stoppings = [[EarlyStopping(monitor=temp,
                                      patience=500,
                                      threshold=0.0001,
                                      threshold_mode='rel',
                                      lower_is_better=low)] for temp, low in [('valid_loss', True), ('valid_acc', False)]]
    # enzymes = DataListLoader(enzymes, batch_size=1)
    X = torch.arange(len(dataset)).long()
    # net.fit(X, y)
    # for c_seed in range(n_trials):

    params = {
        'lr': [10 ** -4, 10 ** -5],
        'module__embedding_dim': [32, 64],
        'module__num_layers': [2, 3, 4],
        'module__k': ks,
        'callbacks': early_stoppings,
    }
    scoring = {'acc': 'accuracy',
#            # 'balanced_acc': 'balanced_accuracy',
#            # 'f1_macro': 'f1_macro',
#            # 'f1_micro': 'f1_micro',
#            # 'precision_macro': 'precision_macro',
#            # 'recall_macro': 'recall_macro',
           }
    c_seed = 1

    outer_cv = StratifiedKFold(n_splits=n_outer_cv, shuffle=True, random_state=c_seed)
    inner_cv = StratifiedKFold(n_splits=n_inner_cv, shuffle=True, random_state=c_seed)
    gs = GridSearchCV(estimator=net,
                      param_grid=params,
                      refit=True,
                      cv=inner_cv,
                      scoring='accuracy',
                      # verbose=0,
                      n_jobs=5)
    #     # gs.fit([[[gr]] for gr in X], y)
    #     gs.fit(X, y)
    test_predictions = cross_validate(gs,
                                      X,
                                      y,
                                      cv=outer_cv,
                                      scoring=scoring,
                                      n_jobs=2)

    print(test_predictions)


#
# dict_cv_predictions = {k: v.tolist() for k, v in dict(test_predictions).items()}
# trial_predictions.append(dict_cv_predictions)
# save_cv_predictions(file_results, trial_predictions)

# params = {
#     'lr': [0.0001, 0.00001],
#     'module__embedding_dim': [32, 64],
#     'module__num_layers': [2, 3, 4],
#     #         'net__module__num_units': [5, 10, 20, 30],
#     #         'net__module__dropout': [0, 0.25, 0.5],
#     #         'net__module__depth': [1, 3, 5, 10],
# }
# X = TUDataset(root='./data', name='ENZYMES')
# X = DataListLoader(X, batch_size=len(X))
# print(len(X))
# print(dir(X))
# print(X)
# print(np.array(X[0]))
# print(dir(dataset))
# X = Dataset(X)
# print(X[0])
# scoring = {'acc': 'accuracy',
#            # 'balanced_acc': 'balanced_accuracy',
#            # 'f1_macro': 'f1_macro',
#            # 'f1_micro': 'f1_micro',
#            # 'precision_macro': 'precision_macro',
#            # 'recall_macro': 'recall_macro',
#            }
#
# trial_predictions = []
# file_results = get_file_results(folder_results,
#                                 dataset=dataset,
#                                 classifier=classifier,
#                                 use_degree=use_degree,
#                                 remove_node_labels=False)
# #
# for c_seed in range(n_trials):
#     outer_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=c_seed)
#     inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=c_seed)
#
#     gs = GridSearchCV(estimator=net,
#                       param_grid=params,
#                       refit=True,
#                       cv=inner_cv,
#                       # scoring='accuracy',
#                       verbose=1,
#                       n_jobs=1)
#     # gs.fit([[[gr]] for gr in X], y)
#     gs.fit(X, y)
#     # test_predictions = cross_validate(gs,
#     #                                   X,
#     #                                   y,
#     #                                   cv=outer_cv,
#                                   # scoring=scoring,
#                                   n_jobs=1,
#                                   error_score='raise')
#
# dict_cv_predictions = {k: v.tolist() for k, v in dict(test_predictions).items()}
# trial_predictions.append(dict_cv_predictions)
# save_cv_predictions(file_results, trial_predictions)


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

    parser.add_argument('--folder-results',
                        type=str,
                        help='Folder where to save the results')

    args = parser.parse_args()

    main(args)

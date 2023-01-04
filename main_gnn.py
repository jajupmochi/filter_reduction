import argparse

import numpy as np
import torch.optim
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from torch import nn
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

from models.DGCNN import DGCNN
from utils import load_graphs, get_file_results, save_cv_predictions
from torch_geometric.loader import DataLoader

NODE_ATTRIBUTE = 'x'
from torch_geometric.utils import from_networkx


def _get_k(X: Data, percentile: float = 0.6) -> int:

    nbr = [graph.num_nodes for graph in X]
    sorted_nbr = sorted(nbr)

    for i, x in enumerate(sorted_nbr):
        if (1 + i)/len(sorted_nbr) >= percentile:
            return x

    return False



def gnn_classification(root_dataset: str,
                                dataset: str,
                                use_degree: bool,
                                classifier: str,
                                n_trials: int,
                                folder_results: str):
    graphs, labels = load_graphs(root=root_dataset,
                                 dataset=dataset,
                                 remove_node_attr=False,
                                 node_attr=NODE_ATTRIBUTE,
                                 use_degree=use_degree)

    # X = np.array(graphs).astype(np.float32)
    X = [from_networkx(graph, [NODE_ATTRIBUTE]) for graph in graphs]
    y = np.array(labels).astype(np.int64)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    k = _get_k(X, percentile=0.6)


    net = NeuralNetClassifier(
        DGCNN,
        max_epochs=10,
        criterion=nn.CrossEntropyLoss(),
        lr=0.1,
        batch_size=50,
        optimizer=torch.optim.Adam,
        verbose=0,
        train_split=None,
        module__dim_features=X[0].num_features,
        module__dim_target=len(set(y)),
        module__k=k,
    )
    print(net)
    params = {
        'net__lr': [0.0001, 0.00001],
        'net__module__embedding_dim': [32, 64],
        'net__module__num_layers': [2, 3, 4],
        #         'net__module__num_units': [5, 10, 20, 30],
        #         'net__module__dropout': [0, 0.25, 0.5],
        #         'net__module__depth': [1, 3, 5, 10],
    }
    from skorch.dataset import Dataset
    X = TUDataset(root='./data', name='ENZYMES')
    print(dir(X))
    # print(np.array(X[0]))
    # print(dir(dataset))
    # X = Dataset(X)
    print(dir(X))
    # print(X[0])
    scoring = {'acc': 'accuracy',
               # 'balanced_acc': 'balanced_accuracy',
               # 'f1_macro': 'f1_macro',
               # 'f1_micro': 'f1_micro',
               # 'precision_macro': 'precision_macro',
               # 'recall_macro': 'recall_macro',
               }

    trial_predictions = []
    file_results = get_file_results(folder_results,
                                    dataset=dataset,
                                    classifier=classifier,
                                    use_degree=use_degree,
                                    remove_node_labels=False)
    #
    for c_seed in range(n_trials):
        outer_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=c_seed)
        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=c_seed)

        pipe_clf = Pipeline([
            ('net', net)
        ])
        gs = GridSearchCV(estimator=pipe_clf,
                          param_grid=params,
                          refit=True,
                          cv=inner_cv,
                          scoring='accuracy',
                          verbose=1,
                          n_jobs=1)
        test_predictions = cross_validate(gs,
                                          X,
                                          y,
                                          cv=outer_cv,
                                          # scoring=scoring,
                                          n_jobs=1,
                                          error_score='raise')
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

    parser.add_argument('--folder-results',
                        type=str,
                        help='Folder where to save the results')

    args = parser.parse_args()

    main(args)

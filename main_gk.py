import argparse

import numpy as np
from grakel import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC

from utils import get_file_results, save_cv_predictions, load_graphs, make_hashable_attr

NODE_ATTRIBUTE = 'x'


def gk_classification(root_dataset: str,
                      dataset: str,
                      n_trials: int,
                      folder_results: str,
                      remove_node_attr: bool,
                      use_degree: bool):
    """

    Args:
        root_dataset:
        dataset:
        n_trials:
        folder_results:

    Returns:

    """
    nx_graphs, labels = load_graphs(root=root_dataset,
                                    dataset=dataset,
                                    node_attr=NODE_ATTRIBUTE,
                                    remove_node_attr=remove_node_attr,
                                    use_degree=use_degree)
    make_hashable_attr(nx_graphs, node_attr='x')
    grakel_graphs = [graph for graph in graph_from_networkx(nx_graphs,
                                                            node_labels_tag='x',
                                                            as_Graph=True)]
    trial_predictions = []

    scoring = {'acc': 'accuracy',
               'balanced_acc': 'balanced_accuracy',
               'f1_macro': 'f1_macro',
               'f1_micro': 'f1_micro',
               'precision_macro': 'precision_macro',
               'recall_macro': 'recall_macro',
               }

    file_results = get_file_results(folder_results,
                                    dataset,
                                    'WL',
                                    use_degree,
                                    remove_node_attr)

    gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
    K_graphs = gk.fit_transform(grakel_graphs)
    param_grid = {'C': np.logspace(-2, 2, 5)}
    for c_seed in range(n_trials):
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=c_seed)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=c_seed)

        clf = GridSearchCV(estimator=SVC(kernel='precomputed'),
                           param_grid=param_grid,
                           n_jobs=5,
                           cv=inner_cv)

        test_predictions = cross_validate(clf,
                                          K_graphs,
                                          labels,
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
    gk_classification(**vars(args))


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
                        help='if remove-node-attr is also True then it uses the degree of the nodes as attribute')

    parser.add_argument('--n-trials',
                        default=10,
                        type=int,
                        help='Number of times to execute the cross-validation')

    parser.add_argument('--folder-results',
                        type=str,
                        help='Folder where to save the results')

    args = parser.parse_args()

    main(args)

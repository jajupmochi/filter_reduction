import argparse

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import load_singleton_graphs_from_TUDataset, get_file_results, save_cv_predictions

NODE_ATTRIBUTE = 'x'

CLF_METHODS = {
    'knn': (KNeighborsClassifier, {'kneighborsclassifier__n_neighbors': [3, 5, 7, 9, 11]}),
    'rbf': (SVC, {'svc__C': np.logspace(-3, 3, 7)})
}


def singleton_classification(root_dataset: str,
                             dataset: str,
                             use_degree: bool,
                             classifier: str,
                             n_trials: int,
                             folder_results: str):
    """

    Args:
        root_dataset:
        dataset:
        use_degree:
        classifier:
        n_trials:
        folder_results:

    Returns:

    """
    graphs, labels = load_singleton_graphs_from_TUDataset(root=root_dataset,
                                                          dataset=dataset,
                                                          node_attr=NODE_ATTRIBUTE,
                                                          use_degree=use_degree)
    clf_method, param_grid = CLF_METHODS[classifier]
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
                                    classifier,
                                    use_degree)

    for c_seed in range(n_trials):
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=c_seed)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=c_seed)

        pipe_clf = make_pipeline(StandardScaler(),
                                 clf_method())
        clf = GridSearchCV(estimator=pipe_clf,
                           param_grid=param_grid,
                           n_jobs=5,
                           cv=inner_cv)

        test_predictions = cross_validate(clf,
                                          graphs,
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
    singleton_classification(**vars(args))


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
                        default='rbf',
                        choices=['knn', 'rbf'],
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

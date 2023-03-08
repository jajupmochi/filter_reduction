import argparse
from os.path import join, isfile
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
from cyged import Coordinator
from cyged import MatrixDistances
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from utils import load_graphs, get_file_results, save_cv_predictions, seed_everything, get_folder_results, \
    get_folder_distances

NODE_ATTRIBUTE = 'x'
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
KS = [3, 5, 7]


def write_distances(filename: str, distances: np.ndarray) -> None:
    """
    Save the GEDs in `.npy` file

    Args:
        filename: File where to save the GEDs.
        distances: `np.array` containing the GEDs

    Returns:

    """
    with open(filename, 'wb') as file:
        np.save(file, distances)


def load_distances(coordinator: Coordinator,
                   alphas: List[float],
                   n_cores: int,
                   folder_distances: str) -> List[np.ndarray]:
    """
    The function loads or computes the GED matrices for each alpha value in the given list of alphas.
    If the GED matrices for a particular alpha value already exist in the specified folder, it loads the matrices from there.
    Otherwise, it computes the GED matrices using the MatrixDistances class and the coordinator.graphs attribute.
    The matrices are then stored in the specified folder for future use.

    Args:
        coordinator:
        alphas: A list of floats representing the alpha values for which the GED matrices need
                to be computed or loaded.
        n_cores: An integer representing the number of cores to be used for parallel computation of GED matrices
        folder_distances: A string representing the path of the folder where the GED matrices
                        will be stored or loaded from

    Returns:
        A list of numpy arrays representing the GED matrices for all the alpha values.
    """
    is_parallel = n_cores > 0
    matrix_dist = MatrixDistances(coordinator.ged,
                                  parallel=is_parallel)
    distances = []

    for alpha in tqdm(alphas, desc='Load or Compute GED matrices'):
        file_distances = join(folder_distances,
                              f'distances_alpha{alpha}.npy')

        # Check if the file containing the distances for the particular alpha exists
        if isfile(file_distances):
            # If yes load the distances
            dist = np.load(file_distances)
        else:
            # Otherwise compute the GEDs
            coordinator.edit_cost.update_alpha(alpha)
            dist = np.array(matrix_dist.calc_matrix_distances(coordinator.graphs,
                                                              coordinator.graphs,
                                                              num_cores=n_cores))

            write_distances(file_distances, dist)

        distances.append(dist)

    return distances


def reduce_best_params(best_params: List) -> Tuple[float, int, int]:
    """
    This function takes in a list of tuples, where each tuple contains a score,
    an integer k, and another integer idx_alpha.
    The function iterates through each tuple in the list, compares the score of
     the current tuple to a variable "best_score" initialized as negative infinity,
     and if the current score is greater, updates "best_score" and "best_idx" to
     the current score and index.

    Args:
        best_params: List of Tuples where each tuple contains a score, an integer k, and another integer idx_alpha.

    Returns:
       A Tuple containing the highest score , k and idx_alpha.
    """
    best_idx, best_score = None, float('-inf')
    for idx, (score, k, idx_alpha) in enumerate(best_params):
        if score > best_score:
            best_idx, best_score = idx, score

    return best_params[best_idx]



def cross_validate(distances: List[np.ndarray],
                   classes: np.ndarray,
                   param_grid: dict,
                   inner_cv, outer_cv,
                   n_cores: int,
                   scoring: dict):
    scores = {'test_' + name_score: [] for name_score in scoring.keys()}

    for idx_outer, (train_index, test_index) in enumerate(outer_cv.split(distances[0], classes)):
        best_params = []

        # Perform grid search on all the alphas and ks to select the bests
        for alpha_idx, alpha_dist in enumerate(distances):
            clf = GridSearchCV(estimator=KNeighborsClassifier(metric='precomputed'),
                               param_grid=param_grid,
                               n_jobs=n_cores,
                               cv=inner_cv)
            clf.fit(alpha_dist[np.ix_(train_index, train_index)], classes[train_index])

            best_params.append((clf.best_score_, clf.best_params_['n_neighbors'], alpha_idx))

        # Retrieve the best hyperparameters
        _, best_k, best_idx_alpha = reduce_best_params(best_params)
        alpha_dist = distances[best_idx_alpha]

        # Retrain KNN with the best alpha and k and perform the final classification on test set
        knn_test = KNeighborsClassifier(n_neighbors=best_k,
                                        metric='precomputed')
        knn_test.fit(alpha_dist[np.ix_(train_index, train_index)],
                     classes[train_index])
        test_predictions = knn_test.predict(alpha_dist[np.ix_(test_index, train_index)])

        for scorer_name, scorer in scoring.items():
            current_scorer = get_scorer(scorer)
            score = current_scorer._score_func(classes[test_index], test_predictions)

            scores['test_' + scorer_name].append(score)

    return scores

def make_np_attr(graphs: List[nx.Graph]) -> None:
    """
    Change the node attribute from list to np.ndarray.

    Args:
        graphs:

    Returns:

    """
    for graph in graphs:
        for idx_node, node_data in graph.nodes(data=NODE_ATTRIBUTE):
            if type(node_data) != list:
                node_data = [node_data]
            graph.nodes[idx_node][NODE_ATTRIBUTE] = np.array(node_data)


def graph_classifier(root_dataset: str,
                     dataset: str,
                     remove_node_attr: bool,
                     use_degree: bool,
                     alphas: List[float],
                     classifier: str,
                     n_trials: int,
                     n_inner_cv: int,
                     n_outer_cv: int,
                     n_cores: int,
                     folder_results: str):
    """

    Args:
        root_dataset:
        parameters_edit_cost:
        alphas:
        ks:
        n_trial:
        n_outer_cv:
        n_inner_cv:
        n_cores:
        folder_results:
        save_gt_labels:
        save_predictions:
        verbose:
        args:

    Returns:

    """
    seed_everything(7)

    # # Create folders used later
    # Path(folder_results).mkdir(parents=True, exist_ok=True)
    # Path(join(folder_results, 'distances')).mkdir(parents=True, exist_ok=True)

    parameters_edit_cost = (1., 1., 1., 1., 'euclidean')
    graphs, labels = load_graphs(root=root_dataset,
                                 dataset=dataset,
                                 remove_node_attr=remove_node_attr,
                                 node_attr=NODE_ATTRIBUTE,
                                 use_degree=use_degree)
    make_np_attr(graphs)
    coordinator = Coordinator(parameters_edit_cost,
                              graphs=graphs,
                              classes=labels)

    file_results = get_file_results(folder_results,
                                    dataset,
                                    classifier,
                                    use_degree=use_degree,
                                    remove_node_labels=remove_node_attr)

    folder_distances = get_folder_distances(folder_results,
                                            dataset,
                                            classifier,
                                            remove_node_attr,
                                            use_degree)

    distances = load_distances(coordinator, alphas, n_cores, folder_distances)
    param_grid = {'n_neighbors': KS}
    scoring = {'acc': 'accuracy',
               'balanced_acc': 'balanced_accuracy',
               #'f1_macro': 'f1_macro',
               }
    trial_results = []

    for c_seed in range(n_trials):
        outer_cv = StratifiedKFold(n_splits=n_outer_cv, shuffle=True, random_state=c_seed)
        inner_cv = StratifiedKFold(n_splits=n_inner_cv, shuffle=True, random_state=c_seed)

        scores = cross_validate(distances=distances,
                                classes=labels,
                                param_grid=param_grid,
                                inner_cv=inner_cv,
                                outer_cv=outer_cv,
                                n_cores=n_cores,
                                scoring=scoring)
        trial_results.append(scores)
        save_cv_predictions(file_results, trial_results)
        print(scores)



def main(args):
    graph_classifier(**vars(args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Classification Using KNN with GED')
    subparser = parser.add_subparsers()

    parser.add_argument('--root_dataset',
                        type=str,
                        default='./data',
                        help='Root of the dataset')
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
                        default='ged',
                        help='Classification method to use')

    # Hyperparameters to test
    parser.add_argument('--alphas',
                        nargs='*',
                        default=ALPHAS,
                        type=float,
                        help='List of alphas to test')

    # Parameters used during the optimization process
    parser.add_argument('--n-trials',
                        default=10,
                        type=int,
                        help='Number of cross-validation to perform')
    parser.add_argument('--n-inner-cv',
                        default=5,
                        type=int,
                        help='Number of inner loops in the cross-validation')
    parser.add_argument('--n-outer-cv',
                        default=10,
                        type=int,
                        help='Number of outer loops in the cross-validation')

    parser.add_argument('--n-cores',
                        default=0,
                        type=int,
                        help='Set the number of cores to use.'
                             'If n_cores == 0 then it is run without parallelization.'
                             'If n_cores > 0 then use this number of cores')

    parser.add_argument('--folder_results',
                        type=str,
                        help='Folder where to save the classification results')

    parse_args = parser.parse_args()

    main(parse_args)

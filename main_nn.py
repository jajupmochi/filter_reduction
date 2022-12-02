import argparse

import numpy as np
import torch.optim
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from torch import nn
from skorch.callbacks import EarlyStopping

from utils import load_singleton_graphs_from_TUDataset, get_file_results, save_cv_predictions

NODE_ATTRIBUTE = 'x'


class MyModule(nn.Module):
    def __init__(self, input=2, output=2, depth=1, num_units=20, dropout=0.5, nonlin=nn.ReLU()):
        super().__init__()
        self.layers = [nn.Sequential(nn.Linear(input, num_units),
                                     nn.Dropout(dropout))]

        for _ in range(depth):
            self.layers.append(nn.Sequential(nn.Linear(num_units, num_units),
                                             nn.Dropout(dropout)))

        self.nonlin = nonlin
        self.output = nn.Linear(num_units, output)

    def forward(self, X, **kwargs):
        for layer in self.layers:
            X = self.nonlin(layer(X))
        X = self.output(X)
        return X


def singleton_nn_classification(root_dataset: str,
                                dataset: str,
                                use_degree: bool,
                                classifier: str,
                                n_trials: int,
                                folder_results: str):
    graphs, labels = load_singleton_graphs_from_TUDataset(root=root_dataset,
                                                          dataset=dataset,
                                                          node_attr=NODE_ATTRIBUTE,
                                                          use_degree=use_degree)

    X = np.array(graphs).astype(np.float32)
    y = np.array(labels).astype(np.int64)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = NeuralNetClassifier(
        MyModule,
        max_epochs=100,
        criterion=nn.CrossEntropyLoss(),
        lr=0.1,
        batch_size=32 if len(y) < 1000 else 128,
        optimizer=torch.optim.Adam,
        verbose=0,
        train_split=None,
        module__input=X.shape[1],
        module__output=len(set(y)),
    )

    params = {
        'net__lr': [0.005, 0.01, 0.05, 0.1],
        'net__module__num_units': [5, 10, 20, 30],
        'net__module__dropout': [0, 0.25, 0.5],
        'net__module__depth': [1, 3, 5, 10],
    }

    scoring = {'acc': 'accuracy',
               'balanced_acc': 'balanced_accuracy',
               'f1_macro': 'f1_macro',
               'f1_micro': 'f1_micro',
               'precision_macro': 'precision_macro',
               'recall_macro': 'recall_macro',
               }

    trial_predictions = []
    file_results = get_file_results(folder_results,
                                    dataset,
                                    classifier,
                                    use_degree,
                                    False)

    for c_seed in range(n_trials):
        outer_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=c_seed)
        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=c_seed)

        pipe_clf = Pipeline([
            ('scale', StandardScaler()),
            ('net', net)
        ])
        gs = GridSearchCV(estimator=pipe_clf,
                          param_grid=params,
                          refit=True,
                          cv=inner_cv,
                          scoring='accuracy',
                          verbose=1,
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
    singleton_nn_classification(**vars(args))


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
                        default='nn',
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

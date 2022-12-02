from itertools import product

from subprocess import run

datasets = [
    # 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K',
    # 'AIDS', 'BZR', 'BZR_MD', 'COX2', 'COX2_MD', 'DHFR', 'DHFR_MD', 'ER_MD', 'MUTAG',
    # 'Mutagenicity', 'NCI1', 'NCI109', 'PTC_FM', 'PTC_FR', 'PTC_MM', 'PTC_MR',
    # 'DD', 'ENZYMES', 'KKI', 'OHSU', 'Peking_1', 'PROTEINS_full', 'MSRC_9', 'MSRC_21',
    'FRANKENSTEIN', 'PROTEINS', 'COIL-DEL', 'COIL-RAG', 'Letter-high', 'Letter-low',
    'Letter-med',
]
use_degrees = [True, False]

clf_methods = ['rbf']


def main():
    for dataset, classifier, use_degree in product(datasets, clf_methods, use_degrees):
        print(dataset, classifier, use_degree)
        use_deg = '--use-degree' if use_degree else ''
        command = f'python main.py --dataset {dataset} --classifier {classifier} {use_deg}'
        run(command.split())


if __name__ == '__main__':
    main()

from itertools import product

from subprocess import run

# datasets = [
#     'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K',
#     'AIDS', 'BZR', 'BZR_MD', 'COX2', 'COX2_MD', 'DHFR', 'DHFR_MD', 'ER_MD', 'MUTAG',
#     'Mutagenicity', 'NCI1', 'NCI109', 'PTC_FM', 'PTC_FR', 'PTC_MM', 'PTC_MR',
#     'DD', 'ENZYMES', 'KKI', 'OHSU', 'Peking_1', 'PROTEINS_full', 'MSRC_9', 'MSRC_21',
#     'FRANKENSTEIN', 'PROTEINS', 'COIL-DEL', 'COIL-RAG', 'Letter-high', 'Letter-low', 'Letter-med',
# ]

# datasets = ['MCF-7', 'MCF-7H', 'PC-3', 'PC-3H',
#             'Tox21_AhR', 'Tox21_AR', 'Tox21_AR-LBD', 'Tox21_ARE', 'Tox21_aromatase', 'Tox21_ATAD5', 'Tox21_ER',
#             'Tox21_ER-LBD', 'Tox21_HSE', 'Tox21_MMP', 'Tox21_p53', 'Tox21_PPAR-gamma',
#             'deezer_ego_nets']

datasets = [
    'Tox21_AhR_training', 'Tox21_AR_training', 'Tox21_AR-LBD_training', 'Tox21_ARE_training', 'Tox21_aromatase_training', 'Tox21_ATAD5_training', 'Tox21_ER_training', 'Tox21_ER-LBD_training',
    'Tox21_HSE_training', 'Tox21_MMP_training', 'Tox21_p53_training', 'Tox21_PPAR-gamma_training',
]

use_degrees = [False]  # [True, False]

clf_methods = ['knn', 'rbf']


def main():
    for dataset, classifier, use_degree in product(datasets, clf_methods, use_degrees):
        print(dataset, classifier, use_degree)
        use_deg = '--use-degree' if use_degree else ''
        command = f'python main_knn.py --dataset {dataset} --classifier {classifier} {use_deg}'
        run(command.split())


if __name__ == '__main__':
    main()

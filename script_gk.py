from subprocess import run
from itertools import product

# datasets = [
#     # 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K',
#     # 'AIDS', 'BZR', 'BZR_MD', 'COX2', 'COX2_MD', 'DHFR', 'DHFR_MD', 'ER_MD', 'MUTAG',
#     # 'Mutagenicity', 'NCI1', 'NCI109', 'PTC_FM', 'PTC_FR', 'PTC_MM', 'PTC_MR',
#     # 'DD', 'ENZYMES', 'KKI', 'OHSU', 'Peking_1', 'PROTEINS_full', 'MSRC_9', 'MSRC_21',
#     # 'FRANKENSTEIN', 'PROTEINS', 'COIL-DEL', 'COIL-RAG', 'Letter-high', 'Letter-low', 'Letter-med',
# ]

datasets = ['AIDS', 'BZR', 'BZR_MD', 'COX2', 'COX2_MD', 'DHFR', 'DHFR_MD', 'ER_MD', 'MUTAG',
            'Mutagenicity', 'NCI1', 'NCI109', 'PTC_FM', 'PTC_FR', 'PTC_MM', 'PTC_MR',
            'DD', 'ENZYMES', 'KKI', 'OHSU', 'Peking_1', 'PROTEINS_full', 'MSRC_9', 'MSRC_21', ]

use_degrees = [True, False]

remove_node_attr = True


def main():
    for dataset, use_degree in product(datasets, use_degrees):
        print(dataset, use_degree)
        use_deg = '--use-degree' if use_degree else ''
        remove_attr = '--remove-node-attr' if remove_node_attr else ''
        command = f'python gk_main.py --dataset {dataset} {remove_attr} {use_deg}'
        run(command.split())


if __name__ == '__main__':
    main()

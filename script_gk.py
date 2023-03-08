from subprocess import run
from itertools import product

# datasets = [
#     'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K',
#     'AIDS', 'BZR', 'BZR_MD', 'COX2', 'COX2_MD', 'DHFR', 'DHFR_MD', 'ER_MD', 'MUTAG',
#     'Mutagenicity', 'NCI1', 'NCI109', 'PTC_FM', 'PTC_FR', 'PTC_MM', 'PTC_MR',
#     'DD', 'ENZYMES', 'KKI', 'OHSU', 'Peking_1', 'PROTEINS_full', 'MSRC_9', 'MSRC_21',
#     'FRANKENSTEIN', 'PROTEINS',
# ]
#
# datasets = ['AIDS', 'BZR', 'BZR_MD', 'COX2', 'COX2_MD', 'DHFR', 'DHFR_MD', 'ER_MD', 'MUTAG',
#             'Mutagenicity', 'NCI1', 'NCI109', 'PTC_FM', 'PTC_FR', 'PTC_MM', 'PTC_MR',
#             'DD', 'ENZYMES', 'KKI', 'OHSU', 'Peking_1', 'PROTEINS_full', 'MSRC_9', 'MSRC_21', ]


datasets = ['MCF-7', 'MCF-7H', 'PC-3', 'PC-3H',
            'deezer_ego_nets',
            'Tox21_AhR_training', 'Tox21_AR_training', 'Tox21_ARE_training',
            'Tox21_aromatase_training', 'Tox21_ATAD5_training', 'Tox21_ER_training',
            'Tox21_HSE_training', 'Tox21_MMP_training', 'Tox21_p53_training', 'Tox21_PPAR-gamma_training',
            ]
# use_degrees = [True]  # , False]
#
# remove_node_attr = False


def main():
    # (use_degree, remove_node_attr)
    options = [(False, False), (True, True)]
    for dataset, (use_degree, remove_node_attr) in product(datasets, options):
        print(dataset, use_degree, remove_node_attr)
        use_deg = '--use-degree' if use_degree else ''
        remove_attr = '--remove-node-attr' if remove_node_attr else ''
        command = f'python main_gk.py --dataset {dataset} {remove_attr} {use_deg}'
        run(command.split())


if __name__ == '__main__':
    main()

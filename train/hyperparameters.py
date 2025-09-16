import argparse

def get_hyperparameters():
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu id(s) to use, e.g. --gpus 0 1 for multiple GPUs')
    parser.add_argument('--num_clients', type=int, default=3, help='number of clients (default: 3)')
    parser.add_argument('--num_rounds', type=int, default=3, help='number of federated learning rounds (default: 3)')
    parser.add_argument('--data_root_dir', type=str, 
                        default="/local/scratch/phempel/chimera/features_1536",
                        help='data directory')
    parser.add_argument('--embed_dim', type=int, default=1536)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--max_epochs', type=int, default=150,
                        help='maximum number of epochs to train (default: 150)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--label_frac', type=float, default=1.0,
                        help='fraction of training labels (default: 1.0)')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument('--split_dir', type=str, 
                        default='chimera_3_0.1', 
                        help='manually specify the set of splits to use')
    parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')
    parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')
    parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
    parser.add_argument('--drop_out', type=float, default=0.5, help='dropout')
    parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                        help='slide-level classification loss function (default: ce)')
    parser.add_argument('--exp_code', type=str, required=True, help='experiment name for saving results')
    parser.add_argument('--weighted_sample', action='store_true', default=True, help='enable weighted sampling')
    parser.add_argument('--model_size', type=str, choices=['small', 'big', 'tiny'], default='tiny', help='size of model, does not affect mil')
    ### CLAM specific options
    parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default='svm',
                        help='instance-level clustering loss function (default: None)')
    parser.add_argument('--subtyping', action='store_true', default=True, 
                        help='subtyping problem')
    parser.add_argument('--bag_weight', type=float, default=0.8,
                        help='clam: weight coefficient for bag-level loss (default: 0.8)')
    parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
    parser.add_argument('--no_verbose', action='store_true', default=False,
                        help='disable verbose output')
    parser.add_argument('--clinical_dim', type=int, default=256,
                        help='dimension of clinical features (default: 256), use 23 for directly using CD features')
    parser.add_argument('--norm', action='store_true', default=True, 
                        help='normalize the features before fusion, only for multimodal models')
    parser.add_argument('--pages', type=int, nargs='+', default=[1, 2, 3],
                        help='list of pages to use for multi-scale model, if None, all pages are used (default: None)')
    parser.add_argument('--page', type=int, default=1,
                        help='page to use for single-scale model, if None, all pages are used (default: 1)')
    parser.add_argument('--return_coords', action='store_true', default=True,
                        help='return coordinates of patches for multi-scale model, only for multi-scale model')
    # Fine-granular learning rates & regularization
    parser.add_argument('--lr_clinical', type=float, default=2e-5,
                        help='learning rate for clinical model')
    parser.add_argument('--lr_clam', type=float, default=2.5e-5,
                        help='learning rate for CLAM model')
    parser.add_argument('--lr_mm', type=float, default=9e-5,
                        help='learning rate for multimodal fusion components (default: 1e-4)')

    # Add new hierarchical learning rate arguments
    parser.add_argument('--lr_attention', type=float, default=None,
                        help='learning rate for attention networks within CLAM hierarchy (default: same as lr_clam)')
    parser.add_argument('--lr_attention_levels', type=str, default='4e-5,3e-5,2e-5',
                        help='comma-separated learning rates for each attention level, e.g., "1e-4,5e-5,1e-5" for levels 0,1,2 (default: use lr_attention for all)')
    parser.add_argument('--lr_fusion_net', type=float, default=None,
                        help='learning rate for fusion network layers (default: same as lr_mm)')
    parser.add_argument('--lr_classifier', type=float, default=None,
                        help='learning rate for final classifier layers (default: same as lr_mm)')
    parser.add_argument('--lr_level_weights', type=float, default=None,
                        help='learning rate for level weighting parameters in hierarchical CLAM (default: higher than lr_clam)')

    parser.add_argument('--reg_clinical', type=float, default=1e-6, help='weight decay for clinical model')
    parser.add_argument('--reg_clam', type=float, default=0.001, help='weight decay for CLAM model')
    parser.add_argument('--reg_mm', type=float, default=1e-4, help='weight decay for multimodal model')
    parser.add_argument('--use_scheduler', action='store_true', default=True,
                        help='use learning rate scheduler, only for CLAM models')
    parser.add_argument('--multi_split', action='store_true', default=False,
                        help='To train on multiple splits to compare performance with less noise. To use, provide in split_dir the path to the common parent directory of the split folders instead')
    parser.add_argument('--top_p', type=float, default=0.3, help='For the hierarchical approach: top percentage of patches that are selected in the next layer')
    args = parser.parse_args()
    return args
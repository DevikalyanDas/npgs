from argparse import ArgumentParser,

def ParseArgs():
    parser = ArgumentParser()
    # Main options
    parser.add_argument('--data_dir',
        type=str,
        required=True,
        help = 'data directory file to use')
    
    parser.add_argument('--conf',
        type=str,
        required=True,
        help = 'configuration file to use')

    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='mode to run (see README or code for options)'
    )

    parser.add_argument(
        '--is_continue',
        default='False',
        type=str,
        help='continue with previously trained model'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='index of GPU to use'
    )
    parser.add_argument('--fold_id',
        default= '',
        type=str, 
        help="folder identification for the files"
    )

    return parser.parse_args()
    
import argparse

from domain_discovery import discover_domains
from preprocess import preprocess
from train import train

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="data directory", type=str, default='./DATA/')
    parser.add_argument("--datasets", help='datasets_name', type=str, default=['TOY1', 'TOY2', 'TOY3'])
    parser.add_argument("--features", help='feature types', type=list, default=['txt','net'])
    parser.add_argument("--random_state", help='random state', type=int, default=0)
    parser.add_argument("--B", help='labelling budget as % of training datapool', type=float, default=0.8)
    parser.add_argument("--h", help='number of hashing functions', type=int, default=10)
    parser.add_argument("--lambda1", help='lambda1 hyperparameter', type=float, default=1)
    parser.add_argument("--lambda2", help='lambda2 hyperparameter', type=float, default=10)
    parser.add_argument("--lambda3", help='lambda3 hyperparameter', type=float, default=5)
    parser.add_argument("--d", help='latent dimension size', type=int, default=512)
    parser.add_argument("--epochs", help='number of epochs', type=int, default=300)
    parser.add_argument("--batch_size", help='batch size', type=int, default=64)
    args = parser.parse_args()

    # run preprocessing script for each dataset
    for dataset in args.datasets:
        args.dataset = dataset
        node_sets = preprocess(args)
        print('Text, Network Embeddings and Labels for {} dataset are stored at {}'.format(dataset, args.path+dataset+'/'))

    # run the pipline for domain discovery
    discover_domains(args)
    # train and evaluate the model
    train(args)

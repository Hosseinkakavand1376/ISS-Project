import argparse
import os

def run_cnn_model(args):
    import cnn_model
    cnn_model.main(args)

def run_autoencoder_model(args):
    import autoencoder
    autoencoder.main(args)

def run_random_forest_model(args):
    import random_forest
    random_forest.main(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or evaluate an ML model for IDS')
    subparsers = parser.add_subparsers(dest='model', required=True, help='Choose which model to use')

    # CNN model arguments
    cnn_parser = subparsers.add_parser('cnn', help='CNN model')
    cnn_parser.add_argument('--dataset_path', type=str, required=True, help='Path to the directory containing the dataset files')
    cnn_parser.add_argument('--test_size', type=float, default=0.2, help='Test set size as a fraction of the total dataset')
    cnn_parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    cnn_parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    cnn_parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    cnn_parser.add_argument('--scaler', type=str, choices=['minmax', 'standard'], default='standard', help='Scaler type: "minmax" or "standard"')
    cnn_parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True, help='Mode: "train" or "eval"')
    cnn_parser.add_argument('--encoded', action='store_true', help='Whether to use encoded labels')
    cnn_parser.add_argument('--num_files', type=int, default=5, help='Number of dataset files to load for training')
    cnn_parser.add_argument('--encode', action='store_true', help='Whether to encode input data using an autoencoder first')

    # Autoencoder model arguments
    autoencoder_parser = subparsers.add_parser('autoencoder', help='Autoencoder model')
    autoencoder_parser.add_argument('--dataset_path', type=str, required=True, help='Path to the directory containing the dataset files')
    autoencoder_parser.add_argument('--test_size', type=float, default=0.2,
                                    help='Test set size as a fraction of the total dataset (required for training)',
                                    nargs='?')
    autoencoder_parser.add_argument('--learning_rate', type=float, default=0.001,
                                    help='Learning rate for the optimizer')
    autoencoder_parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    autoencoder_parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    autoencoder_parser.add_argument('--scaler', type=str, choices=['minmax', 'standard'], default='standard',
                                    help='Scaler type: "minmax" or "standard"')
    autoencoder_parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True,
                                    help='Mode: "train" or "eval"')
    autoencoder_parser.add_argument('--encoded', action='store_true', help='Whether to use encoded labels')
    autoencoder_parser.add_argument('--num_files', type=int, default=5,
                                    help='Number of dataset files to load for training')
    autoencoder_parser.add_argument('--loss', type=str, choices=['mse', 'mae'], default='mse',
                                    help='Loss function: "mse" or "mae"')

    # Random Forest model arguments
    random_forest_parser = subparsers.add_parser('random_forest', help='Random Forest model')
    random_forest_parser.add_argument('--dataset_path', type=str, required=True,
                                      help='Path to the directory containing the dataset files')
    random_forest_parser.add_argument('--test_size', type=float, default=0.2,
                                      help='Test set size as a fraction of the total dataset')
    random_forest_parser.add_argument('--scaler', type=str, choices=['minmax', 'standard'], default='standard',
                                      help='Scaler type: "minmax" or "standard"')
    random_forest_parser.add_argument('--num_files', type=int, default=5,
                                      help='Number of dataset files to load for evaluation')

    args = parser.parse_args()

    if args.model == 'cnn':
        run_cnn_model(args)
    elif args.model == 'autoencoder':
        run_autoencoder_model(args)
    elif args.model == 'random_forest':
        run_random_forest_model(args)

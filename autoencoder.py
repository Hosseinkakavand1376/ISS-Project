import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Cropping1D, Dropout, Conv1D, MaxPool1D, UpSampling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

# Load and preprocess data
def load_data(filename):
    df = pd.read_csv(filename)
    label_replacements = {
        'DDoS-ICMP_Flood': 'DDoS', 'DDoS-UDP_Flood': 'DDoS', 'DDoS-TCP_Flood': 'DDoS', 'DDoS-PSHACK_Flood': 'DDoS',
        'DDoS-SYN_Flood': 'DDoS', 'DDoS-RSTFINFlood': 'DDoS', 'DDoS-SynonymousIP_Flood': 'DDoS',
        'DDoS-ICMP_Fragmentation': 'DDoS', 'DDoS-UDP_Fragmentation': 'DDoS', 'DDoS-ACK_Fragmentation': 'DDoS',
        'DDoS-HTTP_Flood': 'DDoS', 'DDoS-SlowLoris': 'DDoS',
        'DoS-UDP_Flood': 'DoS', 'DoS-TCP_Flood': 'DoS', 'DoS-SYN_Flood': 'DoS', 'DoS-HTTP_Flood': 'DoS',
        'Recon-HostDiscovery': 'Recon', 'Recon-OSScan': 'Recon', 'Recon-PortScan': 'Recon', 'Recon-PingSweep': 'Recon',
        'VulnerabilityScan': 'Recon',
        'Mirai-greeth_flood': 'Mirai', 'Mirai-udpplain': 'Mirai', 'Mirai-greip_flood': 'Mirai',
        'MITM-ArpSpoofing': 'Spoofing', 'DNS_Spoofing': 'Spoofing',
        'DictionaryBruteForce': 'BruteForce',
        'BrowserHijacking': 'Web-based', 'XSS': 'Web-based', 'Uploading_Attack': 'Web-based',
        'SqlInjection': 'Web-based', 'CommandInjection': 'Web-based', 'Backdoor_Malware': 'Web-based',
        'BenignTraffic': 'BENIGN'
    }
    df['label'] = df['label'].replace(label_replacements)
    return df

def plot_training_history(history):
    acc = history.history['val_loss']
    val_acc = history.history['loss']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training loss')
    plt.plot(epochs, val_acc, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def main(args):
    if args.mode == 'eval':
        if args.scaler == 'minmax' and args.loss == 'mse':
            model_path = 'autoencoder_with_MinMaxScaler_linear_MeanSquaredError.keras'
        elif args.scaler == 'minmax' and args.loss == 'mae':
            model_path = 'autoencoder_with_MinMaxScaler_linear.keras'
        elif args.scaler == 'standard' and args.loss == 'mse':
            model_path = 'autoencoder_with_StandardScaler_linear_MeanSquaredError.keras'
        else:
            model_path = 'autoencoder_with_StandardScaler_linear.keras'

        print(f"Loading model from {model_path}...")
        autoencoder = load_model(model_path)
        print(f"Loaded model from {model_path}")

        # Load test data for evaluation
        test = load_data(f"{args.dataset_path}/part-00016-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")
        label_encoder = LabelEncoder()
        test['label'] = label_encoder.fit_transform(test['label'])
        y_test = test['label']

        if args.scaler == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        X_test = scaler.fit_transform(test.drop(columns='label'))
        X_test_expanded = X_test.reshape(-1, X_test.shape[1], 1)

        test_loss = autoencoder.evaluate(X_test_expanded, X_test_expanded)
        print(f"Test Loss: {test_loss}")

        X_test_reconstructed = autoencoder.predict(X_test_expanded)
        print(f"Test expanded:\n {X_test_expanded}")
        print(f"Test reconstructed:\n {X_test_reconstructed}")

        for i in range(5):
            original_value = X_test_expanded[i][0]
            reconstructed_value = X_test_reconstructed[i][0]
            print(f"Original: {original_value}, Reconstructed: {reconstructed_value}")

        print(f"Range of original values: {X_test_expanded.min()} to {X_test_expanded.max()}")
        print(f"Range of reconstructed values: {X_test_reconstructed.min()} to {X_test_reconstructed.max()}")

        mae = mean_absolute_error(X_test_expanded.flatten(), X_test_reconstructed.flatten())
        mse = mean_squared_error(X_test_expanded.flatten(), X_test_reconstructed.flatten())
        r2 = r2_score(X_test_expanded.flatten(), X_test_reconstructed.flatten())

        correlation_matrix = np.corrcoef(X_test_expanded.flatten(), X_test_reconstructed.flatten())
        correlation_coefficient = correlation_matrix[0, 1]

        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R^2 Score: {r2}")
        print(f"Pearson Correlation Coefficient: {correlation_coefficient}")

    else:
        dataset = []
        for i in range(0, args.num_files):
            df = load_data(f"{args.dataset_path}/part-0000{i}-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")
            dataset.append(df)
        dataset = pd.concat(dataset)

        dataset.info()
        dataset.value_counts('label')

        label_encoder = LabelEncoder()
        dataset['label'] = label_encoder.fit_transform(dataset['label'])
        dataset.value_counts('label')

        corr_matrix = dataset.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

        plt.subplots(figsize=(10, 15))
        sns.heatmap(corr_matrix.iloc[:46, 46:])

        X = dataset.drop(columns='label')
        X.describe()

        y = dataset['label']

        # Select the scaler based on user input
        if args.scaler == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        normalized_features = scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(normalized_features, y, test_size=args.test_size,
                                                          random_state=42)

        X_train_encoder = np.expand_dims(X_train, axis=-1)
        X_val_encoder = np.expand_dims(X_val, axis=-1)

        test = load_data(f"{args.dataset_path}/part-00016-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")

        X_test = scaler.fit_transform(test.drop(columns='label'))

        test['label'] = label_encoder.fit_transform(test['label'])
        y_test = test['label']
        test.value_counts('label')

        X_train_expanded = X_train.reshape(-1, X_train.shape[1], 1)
        X_val_expanded = X_val.reshape(-1, X_val.shape[1], 1)
        X_test_expanded = X_test.reshape(-1, X_test.shape[1], 1)

        def build_autoencoder(input_shape):
            input_layer = Input(shape=input_shape)

            # Encoder
            x = Conv1D(128, 3, activation='relu', padding='same')(input_layer)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)

            x = MaxPool1D(2, padding='same')(x)
            x = Conv1D(64, 3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Conv1D(32, 3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            encoded = MaxPool1D(2, padding='same')(x)

            # Decoder
            x = Conv1D(32, 3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Conv1D(64, 3, activation='relu', padding='same')(encoded)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = UpSampling1D(2)(x)
            x = Conv1D(128, 3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = UpSampling1D(2)(x)
            x = Cropping1D((1, 1))(x)
            decoded = Conv1D(input_shape[1], 3, activation='linear', padding='same')(x)

            autoencoder = Model(input_layer, decoded)
            # Select the loss function based on user input
            if args.loss == 'mae':
                loss = MeanAbsoluteError()
            else:
                loss = MeanSquaredError()
            autoencoder.compile(optimizer=Adam(learning_rate=args.learning_rate), loss=loss)
            return autoencoder

        autoencoder = build_autoencoder((X_train_expanded.shape[1], 1))
        autoencoder.summary()

        def lr_schedule(epoch, lr):
            if epoch % 3 == 0 and epoch != 0:
                return lr * 0.9
            return lr

        lr_scheduler = LearningRateScheduler(lr_schedule)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = autoencoder.fit(
            X_train_expanded, X_train_expanded,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            validation_data=(X_val_expanded, X_val_expanded),
            callbacks=[early_stopping, reduce_lr, lr_scheduler]
        )

        plot_training_history(history)

        test_loss = autoencoder.evaluate(X_test_expanded, X_test_expanded)
        print(f"Test Loss: {test_loss}")

        X_test_reconstructed = autoencoder.predict(X_test_expanded)
        print(f"Test expanded:\n {X_test_expanded}")
        print(f"Test reconstructed:\n {X_test_reconstructed}")

        for i in range(5):
            original_value = X_test_expanded[i][0]
            reconstructed_value = X_test_reconstructed[i][0]
            print(f"Original: {original_value}, Reconstructed: {reconstructed_value}")

        print(f"Range of original values: {X_test_expanded.min()} to {X_test_expanded.max()}")
        print(f"Range of reconstructed values: {X_test_reconstructed.min()} to {X_test_reconstructed.max()}")

        mae = mean_absolute_error(X_test_expanded.flatten(), X_test_reconstructed.flatten())
        mse = mean_squared_error(X_test_expanded.flatten(), X_test_reconstructed.flatten())
        r2 = r2_score(X_test_expanded.flatten(), X_test_reconstructed.flatten())

        correlation_matrix = np.corrcoef(X_test_expanded.flatten(), X_test_reconstructed.flatten())
        correlation_coefficient = correlation_matrix[0, 1]

        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R^2 Score: {r2}")
        print(f"Pearson Correlation Coefficient: {correlation_coefficient}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or evaluate an Autoencoder model for IDS')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the directory containing the dataset files')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size as a fraction of the total dataset')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--scaler', type=str, choices=['minmax', 'standard'], default='standard',
                        help='Scaler type: "minmax" or "standard"')
    parser.add_argument('--num_files', type=int, default=5, help='Number of dataset files to load for training')
    parser.add_argument('--loss', type=str, choices=['mse', 'mae'], default='mse', help='Loss function: "mse" or "mae"')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True, help='Mode: "train" or "eval"')

    args = parser.parse_args()
    main(args)
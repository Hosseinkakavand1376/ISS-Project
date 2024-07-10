import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Cropping1D, Dropout, Conv1D, MaxPool1D, UpSampling1D, \
    BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel
from kerastuner.tuners import RandomSearch
from tensorflow.keras.regularizers import l1
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import os

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
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
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
        if args.scaler == 'minmax':
            if args.encoded:
                model_path = 'CNN_Model_MinMaxScaler_encoded.keras'
            else:
                model_path = 'CNN_Model_MinMaxScaler.keras'
        else:
            model_path = 'CNN_Model_StandardScaler.keras'

        # Ensure the model path exists
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} does not exist.")
            return

        print(f"Loading model from {model_path}...")

        # Load the pre-trained model
        try:
            model = load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        print(f"Loaded model from {model_path}")

        # Load and preprocess the test data
        dataset = []
        for i in range(0, args.num_files):
            df = load_data(f"{args.dataset_path}/part-0000{i}-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")
            dataset.append(df)
        dataset = pd.concat(dataset)

        label_encoder = LabelEncoder()
        dataset['label'] = label_encoder.fit_transform(dataset['label'])
        X = dataset.drop(columns='label')
        y = dataset['label']

        # Select the scaler based on user input
        if args.scaler == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # One-hot encode the labels if not already done
        if args.encoded:
            y = to_categorical(y)
        else:
            num_classes = len(np.unique(y))
            y = to_categorical(y, num_classes=num_classes)

        X_expanded = X_scaled.reshape(-1, X_scaled.shape[1], 1)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_expanded, y)
        print(f"Evaluation Loss: {loss}")
        print(f"Evaluation Accuracy: {accuracy}")

    elif args.mode == 'train':
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

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        class_weights_dict = dict(zip(np.unique(y), class_weights))

        print("Class Weights:", class_weights_dict)

        # Select the scaler based on user input
        if args.scaler == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        normalized_features = scaler.fit_transform(X)

        # Encode inputs if the --encode flag is provided
        if args.encode:
            if args.scaler == 'minmax':
                autoencoder_path = 'autoencoder_with_MinMaxScaler_linear_MeanSquaredError.keras'
            else:
                autoencoder_path = 'autoencoder_with_StandardScaler_linear_MeanSquaredError.keras'

            print(f"Loading autoencoder model from {autoencoder_path}...")
            autoencoder = load_model(autoencoder_path)
            normalized_features = autoencoder.predict(normalized_features.reshape(-1, normalized_features.shape[1], 1))
            normalized_features = normalized_features.reshape(-1, normalized_features.shape[1])
            print(f"Loaded and applied autoencoder model from {autoencoder_path}")

        X_train, X_val, y_train, y_val = train_test_split(normalized_features, y, test_size=args.test_size,
                                                          random_state=42)

        test = load_data(f"{args.dataset_path}/part-00016-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")

        X_test = scaler.fit_transform(test.drop(columns='label'))

        test['label'] = label_encoder.fit_transform(test['label'])
        y_test = test['label']
        test.value_counts('label')

        X_test_expanded = X_test.reshape(-1, X_test.shape[1], 1)

        y_train_cat = to_categorical(y_train)
        y_val_cat = to_categorical(y_val)

        def lr_schedule(epoch, lr):
            if epoch % 2 == 0 and epoch != 0:
                return lr * 0.9
            return lr

        lr_scheduler = LearningRateScheduler(lr_schedule)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        class EncodedCNNHyperModel(HyperModel):
            def build(self, hp):
                model = Sequential()
                model.add(Input(shape=(X_train.shape[1], 1)))
                for i in range(hp.Int('num_layers', 1, 3)):
                    model.add(Conv1D(filters=hp.Int('filters_' + str(i), 32, 128, step=32),
                                     kernel_size=3, activation='relu'))
                    model.add(BatchNormalization())
                    model.add(MaxPool1D(pool_size=2))
                    model.add(Dropout(hp.Float('dropout_' + str(i), 0.2, 0.5, step=0.1)))
                model.add(Flatten())
                model.add(
                    Dense(units=hp.Int('units', 64, 128, step=32), activation='relu', kernel_regularizer=l1(0.01)))
                model.add(Dense(len(np.unique(y_train)), activation='softmax'))
                model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss='categorical_crossentropy',
                              metrics=[CategoricalAccuracy()])
                return model

        tuner = RandomSearch(
            EncodedCNNHyperModel(),
            objective='val_categorical_accuracy',
            max_trials=10,
            executions_per_trial=2,
            directory='model_tuning',
            project_name='NetworkTrafficClassification'
        )

        tuner.search(x=np.expand_dims(X_train, axis=-1), y=y_train_cat,
                     validation_data=(np.expand_dims(X_val, axis=-1), y_val_cat),
                     batch_size=args.batch_size,
                     epochs=10,
                     callbacks=[early_stopping, lr_scheduler])

        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.summary()
        history = best_model.fit(
            np.expand_dims(X_train, axis=-1), y_train_cat,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(np.expand_dims(X_val, axis=-1), y_val_cat),
            callbacks=[early_stopping, lr_scheduler],
            class_weight=class_weights_dict
        )

        plot_training_history(history)

        y_val_pred = best_model.predict(np.expand_dims(X_val, axis=-1))
        y_val_pred_classes = np.argmax(y_val_pred, axis=1)
        y_val_true_classes = np.argmax(y_val_cat, axis=1)

        classification_accuracy = accuracy_score(y_val_true_classes, y_val_pred_classes)
        print(f"Classification accuracy on validation set: {classification_accuracy}")

        test_loss, test_accuracy = best_model.evaluate(X_test_expanded, y_test)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

        y_test_pred = best_model.predict(X_test_expanded)
        y_test_pred_classes = np.argmax(y_test_pred, axis=1)
        y_test_true_classes = y_test  # As y_test is already in encoded form

        classification_accuracy = accuracy_score(y_test_true_classes, y_test_pred_classes)
        print(f"Classification accuracy on test set: {classification_accuracy}")

    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Train or evaluate a CNN model for IDS')
        parser.add_argument('--dataset_path', type=str, required=True,
                            help='Path to the directory containing the dataset files')
        parser.add_argument('--test_size', type=float, default=0.2,
                            help='Test set size as a fraction of the total dataset')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
        parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
        parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
        parser.add_argument('--scaler', type=str, choices=['minmax', 'standard'], default='standard',
                            help='Scaler type: "minmax" or "standard"')
        parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True,
                            help='Mode: "train" or "eval"')
        parser.add_argument('--encoded', action='store_true', help='Whether to use encoded labels')
        parser.add_argument('--num_files', type=int, default=5, help='Number of dataset files to load for training')
        parser.add_argument('--encode', action='store_true',
                            help='Whether to encode input data using an autoencoder first')

        args = parser.parse_args()
        main(args)
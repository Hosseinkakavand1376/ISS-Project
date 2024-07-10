# IDS Model Training and Evaluation

This repository provides scripts to train and evaluate various machine learning models for Intrusion Detection Systems (IDS). The available models are:

1. Convolutional Neural Network (CNN)
2. Autoencoder
3. Random Forest

## Prerequisites

Before running any of the scripts, ensure you have the required tools and libraries installed. This guide assumes you are using PyCharm 2024.1, Python 3.12.2, and WSL2 on Windows 11 with TensorFlow 2.16.2 and Keras 3.4.1.

### Tools and Libraries

- **PyCharm 2024.1**
- **Python 3.12.2**
- **WSL2 (Windows Subsystem for Linux)**
- **TensorFlow 2.16.2**
- **Keras 3.4.1**
- **pandas**
- **numpy**
- **seaborn**
- **matplotlib**
- **scikit-learn**
- **keras-tuner**

## Installation Steps

### 1. Install WSL2 on Windows 11

1. Open PowerShell as Administrator and run:
    ```powershell
    wsl --install
    ```

2. Set your default WSL version to 2:
    ```powershell
    wsl --set-default-version 2
    ```

3. Install a Linux distribution (e.g., Ubuntu) from the Microsoft Store.

### 2. Set Up Your Python Environment

1. Open WSL2 terminal (Ubuntu).
2. Update your package list and install Python:
    ```bash
    sudo apt update
    sudo apt install python3.12
    sudo apt install python3.12-venv
    sudo apt install python3-pip
    ```

3. Create a virtual environment:
    ```bash
    python3.12 -m venv ids_env
    source ids_env/bin/activate
    ```

4. Upgrade pip:
    ```bash
    pip install --upgrade pip
    ```

### 3. Install TensorFlow

Follow the [official guide](https://www.tensorflow.org/install/pip#windows-wsl2) to install TensorFlow on WSL2.

1. Install TensorFlow:
    ```bash
    pip install tensorflow==2.16.2
    ```

### 4. Install PyCharm

1. Download and install PyCharm from the [JetBrains website](https://www.jetbrains.com/pycharm/download/).

2. Configure PyCharm to use WSL2:
    - Open PyCharm.
    - Go to `File` > `Settings` > `Project: <project_name>` > `Python Interpreter`.
    - Click the gear icon and select `Add`.
    - Choose `WSL` from the options and select the Python interpreter from your WSL2 installation.

### 5. Install Required Python Libraries

1. Install the required libraries using pip:
    ```bash
    pip install tensorflow==2.16.2 keras==3.4.1 pandas numpy seaborn matplotlib scikit-learn keras-tuner
    ```

## Dataset

Downloading the CIC IoT Dataset 2023
1. Visit the CIC IoT Dataset 2023 download page (http://205.174.165.80/IOTDataset/CIC_IOT_Dataset2023/)
2. Fill out the required form to download the dataset.
3. Unzip the downloaded files into your project folder. The dataset files should be placed in a subdirectory, e.g., CIC_IoT_Dataset2023.


## Running the Models

### CNN Model

#### Training

To train the CNN model, use the following command:

```bash
!python train_model.py cnn --dataset_path /path/to/dataset --test_size 0.2 --scaler minmax --mode train --epochs 20 --batch_size 128 --learning_rate 0.001 --num_files 5
```

If you want to encode the input data using an autoencoder before training the CNN, add the --encode flag:
```bash
!python train_model.py cnn --dataset_path /path/to/dataset --test_size 0.2 --scaler minmax --mode train --epochs 20 --batch_size 128 --learning_rate 0.001 --num_files 5 --encode
```
#### Evaluation

To evaluate the CNN model, use the following command:
```bash
!python train_model.py cnn --dataset_path /path/to/dataset --scaler minmax --mode eval --num_files 5
```
### Autoencoder
#### Training
To train the Autoencoder model, use the following command:
```bash
!python train_model.py autoencoder --dataset_path /path/to/dataset --test_size 0.2 --scaler minmax --mode train --epochs 15 --batch_size 128 --learning_rate 0.001 --num_files 5 --loss mse
```
#### Evaluation
To evaluate the Autoencoder model, use the following command:
```bash
!python train_model.py autoencoder --dataset_path /path/to/dataset --scaler minmax --mode eval --num_files 5 --loss mse
```
### Random Forest
#### Evaluation
To evaluate the Random Forest model, use the following command:
```bash
!python train_model.py random_forest --dataset_path /path/to/dataset --scaler minmax --num_files 5
```
### Arguments
--dataset_path: Path to the directory containing the dataset files.
--test_size: Test set size as a fraction of the total dataset (required for training mode).
--learning_rate: Learning rate for the optimizer.
--epochs: Number of training epochs.
--batch_size: Batch size for training.
--scaler: Scaler type: "minmax" or "standard".
--mode: Mode: "train" or "eval".
--encoded: Whether to use encoded labels (for CNN only).
--num_files: Number of dataset files to load for training.
--loss: Loss function: "mse" or "mae" (for Autoencoder only).
--encode: Whether to encode input data using an autoencoder first (for CNN training only).

### Example Commands
#### Train CNN without encoding:
```bash
!python train_model.py cnn --dataset_path /path/to/dataset --test_size 0.2 --scaler minmax --mode train --epochs 20 --batch_size 128 --learning_rate 0.001 --num_files 5
```
#### Train CNN with encoding:
```bash
!python train_model.py cnn --dataset_path /path/to/dataset --test_size 0.2 --scaler minmax --mode train --epochs 20 --batch_size 128 --learning_rate 0.001 --num_files 5 --encode
```
#### Evaluate CNN:
```bash
!python train_model.py cnn --dataset_path /path/to/dataset --scaler minmax --mode eval --num_files 5
```
#### Train Autoencoder with MSE loss:
```bash
!python train_model.py autoencoder --dataset_path /path/to/dataset --test_size 0.2 --scaler minmax --mode train --epochs 15 --batch_size 128 --learning_rate 0.001 --num_files 5 --loss mse
```
#### Evaluate Autoencoder with MSE loss:
```bash
!python train_model.py autoencoder --dataset_path /path/to/dataset --scaler minmax --mode eval --num_files 5 --loss mse
```
#### Evaluate Random Forest:
```bash
!python train_model.py random_forest --dataset_path /path/to/dataset --scaler minmax --num_files 5
```
### Notes
Ensure that the dataset files follow the naming convention as expected by the scripts.
The models will be saved and loaded from the specified paths during training and evaluation.
This guide should help you get started with training and evaluating IDS models using the provided scripts. If you encounter any issues or have any questions, please refer to the comments within the code or seek further assistance.


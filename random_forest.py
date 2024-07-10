import argparse
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_paths):
    dataset = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataset.append(df)
    return pd.concat(dataset)

def main(args):
    # Load the dataset from multiple files
    file_paths = [f"{args.dataset_path}/part-0000{i}-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv" for i in range(args.num_files)]
    dataset = load_data(file_paths)

    # Set up the matplotlib figure
    plt.figure(figsize=(15, 10))

    # Heatmap of the first 10 features for the first 100 rows to get an overview of the data
    subset = dataset.iloc[:100, :10]
    sns.heatmap(subset.corr(), annot=True, fmt='.2f', cmap='coolwarm')

    plt.title('Heatmap of Correlation Matrix (First 10 Features)')
    plt.show()

    # Splitting the data into features and target
    X = dataset.drop(columns=['label'])
    y = dataset['label']

    # Encoding the labels
    onehot_encoder = OneHotEncoder(sparse_output=False)
    Y_encoded = onehot_encoder.fit_transform(y.values.reshape(-1, 1))

    # Select the scaler based on user input
    if args.scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    # Standardizing the features
    X_scaled = scaler.fit_transform(X)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_encoded, test_size=args.test_size, random_state=42)

    # Training a RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    with tqdm(total=5, desc="Random Forest Training") as pbar:
        rf.fit(X_train, y_train)
        pbar.update(10)
    Y_pred_rf = rf.predict(X_test)
    Y_pred_rf_proba = rf.predict_proba(X_test)

    cm_rf = confusion_matrix(y_test.argmax(axis=1), Y_pred_rf.argmax(axis=1))
    print("Random Forest Accuracy:", accuracy_score(y_test, Y_pred_rf))
    print("Random Forest Classification Report:\n", classification_report(y_test, Y_pred_rf, target_names=onehot_encoder.categories_[0]))

    # Getting feature importances
    importances = rf.feature_importances_
    feature_names = X.columns

    # Creating a DataFrame for visualization
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    # Plotting the feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

    # Applying PCA
    pca = PCA(n_components=0.95)  # Retain 95% of the variance
    X_pca = pca.fit_transform(X_scaled)

    # Checking the number of components
    num_components = pca.n_components_

    # Variance ratio of each component
    variance_ratio = pca.explained_variance_ratio_

    # Plotting the explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(variance_ratio), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by PCA Components')
    plt.grid(True)
    plt.show()

    print('Number of components: ', num_components)
    print('Variance ratio: ', variance_ratio)

    # Plotting the first two principal components to visualize their relationships
    plt.figure(figsize=(12, 8))

    # Scatter plot of the first two PCA components
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dataset['label'], palette='viridis', alpha=0.7, edgecolor='k')

    plt.title('Scatter Plot of First Two Principal Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()

    # Splitting the dataset into train and test sets for comparison
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=args.test_size, random_state=42)

    # Training a RandomForestClassifier on the original features
    rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_original.fit(X_train, y_train)
    original_accuracy = rf_original.score(X_test, y_test)

    # Training a RandomForestClassifier on the PCA-reduced features
    rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_pca.fit(X_train_pca, y_train_pca)
    pca_accuracy = rf_pca.score(X_test_pca, y_test_pca)

    print('Original accuracy: ', original_accuracy)
    print('Pca accuracy: ', pca_accuracy)

    # Setting up the parameter grid for RandomForest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Initializing the GridSearchCV with RandomForestClassifier
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)

    # Fitting the grid search to the data (using PCA-reduced data for faster computation)
    grid_search.fit(X_pca, y)

    # Retrieving the best parameters
    best_params = grid_search.best_params_

    # Retrieving the best score
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

    # Classification Report
    report = classification_report(y_test, Y_pred_rf)
    print("Classification Report:")
    print(report)

    # Precision-Recall Curve
    rf_optimized = RandomForestClassifier(**best_params, random_state=42)
    rf_optimized.fit(X_train_pca, y_train_pca)
    precision_recall_display = PrecisionRecallDisplay.from_estimator(rf_optimized, X_test_pca, y_test_pca)
    precision_recall_display.plot()
    plt.title('Precision-Recall Curve')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a RandomForest model for IDS')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the directory containing the dataset files')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size as a fraction of the total dataset')
    parser.add_argument('--scaler', type=str, choices=['minmax', 'standard'], default='standard', help='Scaler type: "minmax" or "standard"')
    parser.add_argument('--num_files', type=int, default=3, help='Number of dataset files to load for evaluation')

    args = parser.parse_args()
    main(args)

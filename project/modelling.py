import mlflow
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import argparse
import os
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Parse arguments (NO sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors", type=int, default=50)
    parser.add_argument("--leaf_size", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="train_data.csv")
    args = parser.parse_args()

    n_neighbors = args.n_neighbors
    leaf_size = args.leaf_size
    dataset_name = args.dataset

    # Build dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, dataset_name)
    data = pd.read_csv(dataset_path)

    # Load dataset
    data = pd.read_csv(dataset_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Loan_Status", axis=1),
        data["Loan_Status"],
        test_size=0.2,
        random_state=42
    )

    input_example = X_train[:5]

    # MLflow Run
    with mlflow.start_run(nested=True):

        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            leaf_size=leaf_size
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        precision = precision_score(y_test, pred, average="macro")
        recall = recall_score(y_test, pred, average="macro")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=input_example
        )

        print("Run selesai! Metrics sudah dicatat di MLflow.")

import mlflow
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import argparse
import os
import warnings
from dagshub import DAGsHubLogger

# ====== FORCE MLflow to use DAGSHUB ======
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
assert tracking_uri is not None, "ERROR: MLFLOW_TRACKING_URI tidak ditemukan!"
mlflow.set_tracking_uri(tracking_uri)

# Disable autolog (agar tidak membuat folder mlruns/)
mlflow.autolog(disable=True)

# Set experiment
mlflow.set_experiment("Submission Membangun Sistem Machine Learning - Alya Fauzia")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors", type=int, default=50)
    parser.add_argument("--leaf_size", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="train_data.csv")
    args = parser.parse_args()

    n_neighbors = args.n_neighbors
    leaf_size = args.leaf_size
    dataset_name = args.dataset

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, dataset_name)

    data = pd.read_csv(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Loan_Status", axis=1),
        data["Loan_Status"],
        test_size=0.2,
        random_state=42
    )

    # input_example harus float
    input_example = X_train.head(5).astype(float)

    # MLflow Run (remote only)
    with mlflow.start_run() as run:

        # DagsHub logger (optional tapi recommended)
        logger = DAGsHubLogger()
        logger.log_hyperparams({"n_neighbors": n_neighbors, "leaf_size": leaf_size})

        # Log parameter
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("leaf_size", leaf_size)

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

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        logger.log_metrics({
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        })

        # Save model to remote MLflow (DagsHub)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example
        )


        print("Run Done! Metrics ada di DagHub")
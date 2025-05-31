import os
import pickle
import json
import logging
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import wandb


EMBEDDINGS_PATH = "models/embeddings.pkl"
MODEL_PATH = "models/classifier.pt"
METRICS_PATH = "metrics/train_metrics.json"



wandb.init(project="face-recognition", name="training-run-1")

wandb.config.update({
    "model": "SVM",
    "embedding_model": "FaceNet",
    "dataset": "my_dataset",
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
})


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def load_embeddings(path):
    logging.info(f"Loading embeddings from {path}...")
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    X, y = [], []
    for img_path, embedding in data.items():
        label = os.path.normpath(img_path).split(os.sep)[2]  
        X.append(embedding)
        y.append(label)
    return X, y

def train_classifier(X_train, y_train):
    logging.info("Training classifier...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    logging.info("Training completed.")
    return model

def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Precision: {prec:.4f}")
    logging.info(f"Recall: {rec:.4f}")
    return acc, prec, rec

def save_model(model, path):
    logging.info(f"Saving model to {path}...")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logging.info("Model saved.")

def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Saved metrics to {path}.")

def main():
    X, y = load_embeddings(EMBEDDINGS_PATH)
    unique_classes = set(y)
    logging.info(f"Loaded {len(X)} samples with {len(unique_classes)} unique classes: {unique_classes}")

    if len(unique_classes) < 2:
        logging.error("Ошибка: Для обучения необходимо минимум 2 класса в данных!")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    logging.info(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    model = train_classifier(X_train, y_train)
    acc, prec, rec = evaluate_model(model, X_test, y_test)
    save_model(model, MODEL_PATH)
    save_metrics({"accuracy": acc, "precision": prec, "recall": rec}, METRICS_PATH)

if __name__ == "__main__":
    main()

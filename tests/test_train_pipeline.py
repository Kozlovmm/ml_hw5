import os
from src.train import train_classifier

def test_training_pipeline_runs():
    test_model_path = 'models/test_classifier.pt'
    train_classifier(train_dir='data/processed/train', val_dir='data/processed/val', save_path=test_model_path)

    assert os.path.exists(test_model_path), "Файл модели не создан"

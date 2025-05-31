import pytest
from src.utils import FaceDataset
from src.embedding_model import FaceNetEmbedder
import torch

def test_dataset_loading():
    device = torch.device('cpu')
    embedder = FaceNetEmbedder(device)
    dataset = FaceDataset('data/processed/train', embedder, device)
    
    assert len(dataset) > 0, "Датасет пустой"
    
    embedding, label = dataset[0]
    assert embedding.shape[0] == 512, "Неверная размерность эмбеддинга"
    assert isinstance(label, int), "Метка должна быть целым числом"

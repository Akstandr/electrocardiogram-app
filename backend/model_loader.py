"""Загрузка моделей для предсказания ЭКГ"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import joblib
from pathlib import Path


class ImprovedMLP(nn.Module):
    """MLP модель для классификации ЭКГ"""
    def __init__(self, input_dim=256, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def load_models():
    """Загрузка всех моделей"""
    # Пути к моделям
    base_dir = Path(__file__).parent
    model_path = base_dir / "ecg_predict" / "best_model.pt"
    scaler_path = base_dir / "ecg_predict" / "scaler.pkl"
    pca_path = base_dir / "ecg_predict" / "pca.pkl"
    
    # Проверка существования файлов
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler не найден: {scaler_path}")
    if not pca_path.exists():
        raise FileNotFoundError(f"PCA не найден: {pca_path}")
    
    # Определение устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Загрузка Hubert-ECG модели
    print("Загрузка Hubert-ECG модели...")
    hubert_ecg = AutoModel.from_pretrained(
        "Edoardo-BS/hubert-ecg-base",
        trust_remote_code=True
    ).to(device)
    hubert_ecg.eval()
    
    # Загрузка scaler и PCA
    print("Загрузка scaler и PCA...")
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    
    # Загрузка MLP модели
    print("Загрузка MLP модели...")
    model = ImprovedMLP(input_dim=256, num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    
    print("Все модели загружены!")
    
    return {
        "hubert_ecg": hubert_ecg,
        "scaler": scaler,
        "pca": pca,
        "model": model,
        "device": device
    }


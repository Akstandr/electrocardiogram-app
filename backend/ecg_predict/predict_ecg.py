import json
import numpy as np
import torch
from transformers import AutoModel
import joblib
from scipy.signal import resample
from biosppy.signals.tools import filter_signal
import torch.nn as nn
import torch.nn.functional as F
import argparse
import warnings

# 1. Класс MLP модели
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
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

# 2. Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 3. Загрузка моделей
hubert_ecg = AutoModel.from_pretrained("Edoardo-BS/hubert-ecg-base", trust_remote_code=True).to(device)
hubert_ecg.eval()

scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

model = ImprovedMLP(input_dim=256, num_classes=5)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval().to(device)

# 4. Предобработка
def apply_filter(signal, filter_bandwidth, fs=100):
    order = int(0.3 * fs)
    s, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                            order=order, frequency=filter_bandwidth,
                            sampling_rate=fs)
    return s

def scale_minus1_1(seq, eps=1e-8):
    minv = np.min(seq, axis=1, keepdims=True)
    maxv = np.max(seq, axis=1, keepdims=True)
    return 2 * (seq - minv) / (maxv - minv + eps) - 1

def preprocess_one(sig, orig_fs=500, target_fs=100, duration_sec=5):
    channels, length = sig.shape
    n_take = int(duration_sec * orig_fs)
    sig_cut = sig[:, :n_take]
    n_target = int(duration_sec * target_fs)
    sig_rs = resample(sig_cut, n_target, axis=1)
    sig_f = apply_filter(sig_rs, filter_bandwidth=[0.05, 47], fs=target_fs)
    sig_s = scale_minus1_1(sig_f)
    return sig_s

# 5. Вспомогательные функции
def parse_ecg_json(ecg_json: dict):
    """Парсинг JSON в numpy-массив 12xN"""
    leads_order = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    X_12leads = []
    for lead_name in leads_order:
        lead_data = next(l for l in ecg_json["Leads"] if l["Name"] == lead_name)
        samples = [s["Y"] for s in lead_data["Samples"]]
        X_12leads.append(samples)
    return np.array(X_12leads)

def load_ecg_from_json(json_path: str):
    """Загрузка сигнала из файла (для тестов в терминале)"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return parse_ecg_json(data)

# 6. Эмбеддинги и предсказание
def get_embeddings(sig_pre):
    lead_embeddings = []
    for lead_index in range(12):
        x = torch.tensor(sig_pre[lead_index], dtype=torch.float32)[None, :].to(device)
        with torch.no_grad():
            output = hubert_ecg(input_values=x)
            emb = output.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        lead_embeddings.append(emb)
    return np.stack(lead_embeddings)

class_names = ["NORM", "IMI", "ASMI", "LVH", "NDT"]
class_names_ru = {
    "NORM": "Норма",
    "IMI": "Инфаркт миокарда",
    "ASMI": "Острый инфаркт миокарда",
    "LVH": "Гипертрофия левого желудочка",
    "NDT": "Другие патологии"
}

def predict_ecg_from_dict(ecg_json: dict):
    """Основная функция для веба"""
    sig = parse_ecg_json(ecg_json)
    sig_pre = preprocess_one(sig)
    embeddings_12 = get_embeddings(sig_pre)
    emb_flat = embeddings_12.flatten()[None, :]
    emb_scaled = scaler.transform(emb_flat)
    if not np.isfinite(emb_scaled).all():
        print("⚠️ В данных после масштабирования есть NaN или Inf!")
        print("Минимум:", np.nanmin(emb_scaled), "Максимум:", np.nanmax(emb_scaled))
    # Подавляем предупреждения PCA
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        emb_pca = pca.transform(emb_scaled)


    x = torch.tensor(emb_pca, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = probs.argmax()
        pred_prob = probs[pred_class]

    return class_names_ru[class_names[pred_class]], round(float(pred_prob) * 100, 1)

# 7. Локальный запуск через терминал
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG pathology prediction")
    parser.add_argument("--json", required=True, help="Path to ECG JSON file")
    args = parser.parse_args()

    sig_json = load_ecg_from_json(args.json)
    diagnosis, prob = predict_ecg_from_dict({"Leads": [
        {"Name": name, "Samples": [{"Y": float(y)} for y in sig_json[i]]}
        for i, name in enumerate(["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"])
    ]})
    print(f"Предсказанная патология: {diagnosis} (вероятность {prob}%)")

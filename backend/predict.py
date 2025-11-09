"""Использование модели для предсказания ЭКГ"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample
from biosppy.signals.tools import filter_signal
import warnings


CLASS_NAMES = ["NORM", "IMI", "ASMI", "LVH", "NDT"]
CLASS_NAMES_RU = {
    "NORM": "Норма",
    "IMI": "Инфаркт миокарда",
    "ASMI": "Острый инфаркт миокарда",
    "LVH": "Гипертрофия левого желудочка",
    "NDT": "Другие патологии"
}


def apply_filter(signal, filter_bandwidth, fs=100):
    """Применение фильтра к сигналу"""
    order = int(0.3 * fs)
    s, _, _ = filter_signal(
        signal=signal,
        ftype='FIR',
        band='bandpass',
        order=order,
        frequency=filter_bandwidth,
        sampling_rate=fs
    )
    return s


def scale_minus1_1(seq, eps=1e-8):
    """Масштабирование сигнала в диапазон [-1, 1]"""
    minv = np.min(seq, axis=1, keepdims=True)
    maxv = np.max(seq, axis=1, keepdims=True)
    return 2 * (seq - minv) / (maxv - minv + eps) - 1


def preprocess_signal(sig, orig_fs=500, target_fs=100, duration_sec=5):
    """Предобработка сигнала ЭКГ"""
    channels, length = sig.shape
    n_take = int(duration_sec * orig_fs)
    sig_cut = sig[:, :n_take]
    n_target = int(duration_sec * target_fs)
    sig_rs = resample(sig_cut, n_target, axis=1)
    sig_f = apply_filter(sig_rs, filter_bandwidth=[0.05, 47], fs=target_fs)
    sig_s = scale_minus1_1(sig_f)
    return sig_s


def parse_ecg_json(ecg_json):
    """Парсинг JSON в numpy-массив 12xN"""
    leads_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    X_12leads = []
    for lead_name in leads_order:
        lead_data = next(l for l in ecg_json["Leads"] if l["Name"] == lead_name)
        samples = [s["Y"] for s in lead_data["Samples"]]
        X_12leads.append(samples)
    return np.array(X_12leads)


def get_embeddings(sig_pre, hubert_ecg, device):
    """Получение эмбеддингов для всех отведений"""
    lead_embeddings = []
    for lead_index in range(12):
        x = torch.tensor(sig_pre[lead_index], dtype=torch.float32)[None, :].to(device)
        with torch.no_grad():
            output = hubert_ecg(input_values=x)
            emb = output.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        lead_embeddings.append(emb)
    return np.stack(lead_embeddings)


def predict_ecg(ecg_json, models):
    """
    Предсказание патологии по данным ЭКГ
    
    Args:
        ecg_json: Словарь с данными ЭКГ в формате {"Leads": [...]}
        models: Словарь с загруженными моделями
    
    Returns:
        dict: {"diagnosis": str, "probability": float, "diagnosis_code": str}
    """
    hubert_ecg = models["hubert_ecg"]
    scaler = models["scaler"]
    pca = models["pca"]
    model = models["model"]
    device = models["device"]
    
    # Парсинг и предобработка
    sig = parse_ecg_json(ecg_json)
    sig_pre = preprocess_signal(sig)
    
    # Получение эмбеддингов
    embeddings_12 = get_embeddings(sig_pre, hubert_ecg, device)
    emb_flat = embeddings_12.flatten()[None, :]
    
    # Масштабирование и PCA
    emb_scaled = scaler.transform(emb_flat)
    if not np.isfinite(emb_scaled).all():
        warnings.warn("⚠️ В данных после масштабирования есть NaN или Inf!")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        emb_pca = pca.transform(emb_scaled)
    
    # Предсказание
    x = torch.tensor(emb_pca, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = probs.argmax()
        pred_prob = probs[pred_class]
    
    diagnosis_code = CLASS_NAMES[pred_class]
    diagnosis_ru = CLASS_NAMES_RU[diagnosis_code]
    probability = round(float(pred_prob) * 100, 1)
    
    return {
        "diagnosis": diagnosis_ru,
        "probability": probability,
        "diagnosis_code": diagnosis_code
    }


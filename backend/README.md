# Простой ECG Prediction API

Минималистичный backend для предсказания патологий ЭКГ.

## Структура

```
backend/
├── model_loader.py    # Загрузка моделей
├── predict.py         # Использование модели для предсказания
├── app.py             # FastAPI веб-сервер
├── ecg_predict/       # Модели ML (best_model.pt, scaler.pkl, pca.pkl)
└── requirements.txt   # Зависимости
```

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

### Локально
```bash
python app.py
```

### Docker
```bash
docker-compose up --build
```

## Использование API

### POST `/predict`

**Запрос:**
```json
{
  "Leads": [
    {"Name": "I", "Samples": [{"Y": 0.1}, {"Y": 0.2}, ...]},
    {"Name": "II", "Samples": [{"Y": 0.2}, {"Y": 0.3}, ...]},
    ...
  ]
}
```

**Ответ:**
```json
{
  "diagnosis": "Норма",
  "probability": 95.5,
  "diagnosis_code": "NORM"
}
```

### GET `/health`
Проверка работоспособности

### GET `/docs`
Swagger документация

## Файлы

- **model_loader.py** - загружает все модели (Hubert-ECG, scaler, PCA, MLP)
- **predict.py** - использует модели для предсказания по данным ЭКГ
- **app.py** - FastAPI сервер с endpoint `/predict`


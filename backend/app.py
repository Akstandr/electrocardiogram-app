"""Простой FastAPI сервер для предсказания ЭКГ"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from model_loader import load_models
from predict import predict_ecg


# Загрузка моделей при старте
print("Инициализация моделей...")
models = load_models()

# Создание FastAPI приложения
app = FastAPI(title="ECG Prediction API", version="1.0.0")


# Схемы данных
class ECGSample(BaseModel):
    Y: float


class ECGLead(BaseModel):
    Name: str
    Samples: List[ECGSample]


class ECGRequest(BaseModel):
    Leads: List[ECGLead] = Field(..., min_length=12, max_length=12)


class ECGResponse(BaseModel):
    diagnosis: str
    probability: float
    diagnosis_code: str


# API endpoints
@app.get("/")
def root():
    return {"message": "ECG Prediction API", "docs": "/docs"}


@app.post("/predict", response_model=ECGResponse)
def predict(ecg_request: ECGRequest):
    """Предсказание патологии по данным ЭКГ"""
    try:
        ecg_data = ecg_request.model_dump()
        result = predict_ecg(ecg_data, models)
        return ECGResponse(**result)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Отсутствует отведение: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


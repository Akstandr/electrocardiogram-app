import json
from predict_ecg import predict_ecg_from_dict

with open("00001_hr.json") as f:
    data = json.load(f)

diagnosis, prob = predict_ecg_from_dict(data)
print("Патология:", diagnosis)
print("Вероятность:", prob, "%")

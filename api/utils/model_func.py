import cv2
import numpy as np
from ultralytics import YOLO
import torch
import logging
from transformers import BertTokenizer, BertForSequenceClassification

logger = logging.getLogger(__name__)

# Модель YOLO
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

def transform_image(image):
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_bgr

def blur_faces(img, model, blur_factor=15):
    results = model.predict(img)
    for result in results:
        boxes = result.boxes.data
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            face_roi = img[y1:y2, x1:x2]
            ksize = int(max(face_roi.shape) / blur_factor)
            if ksize % 2 == 0:
                ksize += 1
            blurred_face = cv2.GaussianBlur(face_roi, (ksize, ksize), 0)
            img[y1:y2, x1:x2] = blurred_face
    return img

# Модель BERT
def load_bert_model(tokenizer_path, model_path):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}

def predict_bert(model_data, text):
    tokenizer = model_data["tokenizer"]
    model = model_data["model"]
    inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).item()
    return ["Хороший", "Нейтральный", "Плохой"][preds]
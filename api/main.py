import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import io
import cv2
from utils.model_func import load_yolo_model, transform_image, blur_faces, load_bert_model, predict_bert

logger = logging.getLogger('uvicorn.info')

yolo_model = None
bert_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model, bert_model
    try:
        yolo_model = load_yolo_model("weights/face_yolo11m.pt")
        bert_model = load_bert_model("weights/bert_tokenizer", "weights/bert_classifier")
        logger.info('Модели загружены')
        yield
    except Exception as e:
        logger.error(f"Ошибка при загрузке моделей: {e}")
        yield
        raise

app = FastAPI(lifespan=lifespan)

@app.get('/')
def return_info():
    return 'Привет, FastAPI!'

@app.post('/detect_face', responses={200: {"content": {"image/jpeg": {}}}})
async def detect_face(file: UploadFile):
    loop = asyncio.get_event_loop()
    try:
        if yolo_model is None:
            raise HTTPException(status_code=500, detail="Модель YOLO не загружена")
        image = Image.open(file.file)
        adapted_image = transform_image(image)
        blurred_img = blur_faces(adapted_image, yolo_model)
        blurred_img_rgb = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(blurred_img_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}")

@app.post('/sentiment')
async def get_sentiment(text: str = Form(...)):
    try:
        logger.info(f"Получен запрос на классификацию: {text}")
        if bert_model is None:
            raise HTTPException(status_code=500, detail="Модель классификации отзывов не загружена")
        sentiment = predict_bert(bert_model, text)
        logger.info(f"Результат классификации: {sentiment}")
        return JSONResponse({"sentiment": sentiment})
    except Exception as e:
        logger.error(f"Ошибка при классификации отзыва: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при классификации отзыва: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)
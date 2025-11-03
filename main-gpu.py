from fastapi import FastAPI, UploadFile, File, Depends, Request
from contextlib import asynccontextmanager
from PIL import Image
import io
import numpy as np
from paddleocr import TextDetection, PaddleOCR
import time


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    app.state.text_detection_model = TextDetection(
        model_name="PP-OCRv5_mobile_det",
        device="gpu",
        precision="fp16",
    )
    app.state.text_recognition_model = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu",
        precision="fp16",
    )
    print("Models loaded.")
    yield
    del app.state.text_detection_model
    del app.state.text_recognition_model
    print("Application Shutdown.")


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    print(f"{request.method} {request.url.path} completed in {process_time:.4f}s")
    return response


def get_text_detection_model():
    return app.state.text_detection_model


def get_text_recognition_model():
    return app.state.text_recognition_model


@app.post("/text-detection")
async def text_detection(
    file: UploadFile = File(...), model=Depends(get_text_detection_model)
):
    image_bytes = await file.read()
    file_name = file.filename
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    output = model.predict(np.asarray(image))
    bboxes = output[0]["dt_polys"].tolist()
    confidences = output[0]["dt_scores"]
    return {
        "filename": file_name,
        "bboxes": bboxes,
        "confidences": confidences,
    }


@app.post("/text-recognition")
async def text_recognition(
    file: UploadFile = File(...), model=Depends(get_text_recognition_model)
):
    image_bytes = await file.read()
    file_name = file.filename
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    output = model.predict(np.asarray(image))
    return {
        "filename": file_name,
        "text": output[0]["rec_texts"],
        "confidence": output[0]["rec_scores"],
    }

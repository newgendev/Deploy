from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import uuid
import os

app = FastAPI()

# Static + Templates
app.mount("/static", StaticFiles(directory="Deployment/static_files"), name="static")
templates = Jinja2Templates(directory="Deployment/templates")


# TFLite interpreters 
interpreter = tf.lite.Interpreter(model_path="Deployment/SavedModels/model.tflite")
interpreter.allocate_tensors()

cl_interpreter = tf.lite.Interpreter(model_path="Deployment/SavedModels/good_model.tflite")
cl_interpreter.allocate_tensors()

# Class labels
class_names = ["Non-CT", "Normal", "Stroke"]
clf_class_names = ["Haemorrhgic Stroke", "Ischemic Stroke"]


def preprocess_image(img_bytes):
    """Decode, resize, and preprocess image for ResNet input"""
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    img = tf.keras.applications.resnet50.preprocess_input(img)  # normalization
    img = np.expand_dims(img, axis=0)
    return img


def tflite_predict(interpreter, img):
    """Run inference with TFLite model"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = img.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])



@app.get("/get", response_class=HTMLResponse)
async def get_started(request: Request):
    return templates.TemplateResponse("getStarted.html", {"request": request})


@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("new.html", {"request": request})


@app.get("/submit", response_class=HTMLResponse)
async def submit(request: Request):
    return templates.TemplateResponse("new2nd.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, scan: UploadFile = File(...)):
    # Preprocess image
    img_bytes = await scan.read()
    img = preprocess_image(img_bytes)

    # Inference with TFLite
    prediction = tflite_predict(interpreter, img)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_names[predicted_class_index]

    # Convert to tensor for explainability models
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    if predicted_class_label == "Stroke":
        clf_pred = tflite_predict(cl_interpreter, img)[0][0]

        if clf_pred > 0.5:
            clf_result = f"Prediction: {clf_class_names[1]} (Confidence: {clf_pred * 100:.2f}%)"
        else:
            clf_result = f"Prediction: {clf_class_names[0]} (Confidence: {(1 - clf_pred) * 100:.2f}%)"

        heatmap = compute_gradcam(gradcam_model, img_tensor, predicted_class_index)
        output_img_path = overlay_heatmap(img, heatmap)
        saliency_img_path = saliency_map(gradcam_model, img_tensor, predicted_class_index)

        return templates.TemplateResponse("results.html", {
            "request": request,
            "result": clf_result,
            "img_path": output_img_path,
            "sal_path": saliency_img_path
        })

    elif predicted_class_label == "Non-CT":
        result = f"The Image Provided is not a CT image of the Brain. (Confidence: {prediction[predicted_class_index] * 100:.2f}%)"

    else:
        result = f"Prediction: {class_names[1]} (Confidence: {prediction[predicted_class_index] * 100:.2f}%)"

    heatmap = compute_gradcam(gradcam_model, img_tensor, predicted_class_index)
    output_img_path = overlay_heatmap(img, heatmap)
    saliency_img_path = saliency_map(gradcam_model, img_tensor, predicted_class_index)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "result": result,
        "img_path": output_img_path,
        "sal_path": saliency_img_path
    })

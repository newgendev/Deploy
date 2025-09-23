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

# Keras models 
gradcam_model = tf.keras.models.load_model("Deployment/SavedModels/Best.keras")
cl_gradcam_model = tf.keras.models.load_model("Deployment/SavedModels/good_model.keras")

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


def compute_gradcam(model, img, predicted_class_index):
    """Compute Grad-CAM heatmap"""
    resnet_model = model.get_layer("resnet50")
    last_conv_layer = resnet_model.get_layer("conv5_block3_out")

    grad_model = tf.keras.models.Model(
        inputs=resnet_model.input,
        outputs=[last_conv_layer.output, resnet_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, int(predicted_class_index)]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap


def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image"""
    original_img = img[0].astype(np.uint8)
    heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, alpha, heatmap, 1 - alpha, 0)

    filename = f"gradcam_{uuid.uuid4().hex}.png"
    output_path = os.path.join("Deployment/static_files/outputs", filename)
    cv2.imwrite(output_path, superimposed_img)

    return f"/static/outputs/{filename}"


def saliency_map(model, img_tensor, predicted_class_index):
    """Generate saliency map"""
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        loss = predictions[:, predicted_class_index]

    grads = tape.gradient(loss, img_tensor)[0]
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)

    filename = f"saliency_{uuid.uuid4().hex}.png"
    output_path = os.path.join("Deployment/static_files/saliency", filename)
    plt.imshow(saliency, cmap='coolwarm')
    plt.axis("off")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return f"/static/saliency/{filename}"


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

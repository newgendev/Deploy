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

app.mount("/static", StaticFiles(directory="Deployment/static_files"), name="static")
templates = Jinja2Templates(directory="Deployment/templates")

model = tf.keras.models.load_model("Deployment/SavedModels/model.tflite")
clmodel = tf.keras.models.load_model("Deployment/SavedModels/good_model.tflite")
class_names = ["Non-CT","Normal", "Stroke"]
clf_class_names = ["Haemorrhgic Stroke", "Ischemic Stroke"]

@app.get("/get", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("getStarted.html", {"request": request})
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("new.html", {"request": request})
@app.get("/submit", response_class=HTMLResponse)
async def next(request: Request):
    return templates.TemplateResponse("new2nd.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, scan: UploadFile = File(...)):
    img_bytes = await scan.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class_labels= class_names[predicted_class_index]
    home = "/home"
    if predicted_class_labels == "Stroke":
        clf_pred = clmodel.predict(img)

        if clf_pred > 0.5:
              img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
              clf_result = f"Prediction: {clf_class_names[1]} (Confidence: {float(clf_pred)  * 100}%)"     
              heatmap = compute_gradcam(model, img_tensor, predicted_class_index)
              output_img_path = overlay_heatmap(img, heatmap)
              saliency_img_path = saliency_map(model, img_tensor,predicted_class_index)
 
        else:
             img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
             clf_result = f"Prediction: {clf_class_names[0]} (Confidence: {float(1  - clf_pred)  * 100}%)"
             heatmap = compute_gradcam(model, img_tensor, predicted_class_index)
             output_img_path = overlay_heatmap(img, heatmap)
             saliency_img_path = saliency_map(model, img_tensor,predicted_class_index)

        #return HTMLResponse(content=f"<h2>{clf_result}</h2><br><a href='/home'>Back to Home</a>")
        return templates.TemplateResponse("results.html", {
        "request": request,
        "result": clf_result,
        "img_path": output_img_path,
        "sal_path": saliency_img_path,
})


    elif predicted_class_labels == "Non-CT":
         img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
         result = f"The Image Provided is not a CT image of the Brain: {class_names[0]}. (Confidence:{float(prediction[predicted_class_index])}%)"
         heatmap = compute_gradcam(model, img_tensor, predicted_class_index)
         output_img_path = overlay_heatmap(img, heatmap)
         saliency_img_path = saliency_map(model, img_tensor,predicted_class_index)

        
    else:
          img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
          result = f"Prediction: {class_names[1]} (Confidence: {prediction[predicted_class_index]})"
          heatmap = compute_gradcam(model, img_tensor, predicted_class_index)
          output_img_path = overlay_heatmap(img, heatmap)
          saliency_img_path = saliency_map(model, img_tensor,predicted_class_index)

                
           
    #return HTMLResponse(content=f"<h2>{result}</h2><br><a href='/home'>Back to Home</a>")
    return templates.TemplateResponse("results.html", {
        "request": request,
        "result": result,
        "img_path": output_img_path,
        "sal_path": saliency_img_path,

    })

def compute_gradcam(model, img, predicted_class_index):
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


def overlay_heatmap(img, heatmap, alpha = 0.4):
    original_img = img[0].astype(np.uint8) 
    heatmap = cv2.resize(heatmap,(img.shape[2], img.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, alpha, heatmap, 1 - alpha, 0)

    filename = f"gradcam_{uuid.uuid4().hex}.png"
    output_path = os.path.join("Deployment/static_files/outputs", filename)
    cv2.imwrite(output_path, superimposed_img)

    relative_path = f"/static/outputs/{filename}"
    return relative_path

def saliency_map(model,img_tensor,predicted_class_index):
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)  # Output shape: [1, num_classes]
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class_index]

    grads = tape.gradient(loss, img_tensor)[0]
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)
    pred_index = tf.argmax(predictions[0])
    loss = predictions[0, pred_index]
    
    file_name  = f"gradcam_{uuid.uuid4().hex}.png"
    outputpath = os.path.join("Deployment/static_files/saliency", file_name)
    plt.imshow(saliency, cmap='coolwarm')
    plt.axis("off")
    plt.savefig(outputpath, bbox_inches='tight', pad_inches=0)
    plt.close()
    relative_path = f"/static/saliency/{file_name}"
    return relative_path



     
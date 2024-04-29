import json
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from mltu.tensorflow.losses import CTCloss

app = FastAPI()

# Load the model
model = load_model('model.h5', custom_objects={'CTCloss': CTCloss}, compile=False)

# Define your CTC decoder function
def ctc_decoder(preds, alphabet):
    alphabet = alphabet + ' '  # Add space to the end for blank label
    decoded = ''
    for pred in preds:
        int_preds = np.argmax(pred, axis=1)
        int_preds = [i for i, group in itertools.groupby(int_preds)]
        decoded += ''.join([alphabet[int_pred] for int_pred in int_preds if int_pred < len(alphabet)])
    return decoded

# Define preprocessing function
def preprocess_image(image, width, height):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    # Further preprocessing if needed
    return image

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

@app.get("/test/")
async def test():
    return JSONResponse(content={"text": "API online"})

@app.post("/upload/")
async def upload_image(file: UploadFile):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = preprocess_image(img, 128, 32)
        res = model.predict(np.expand_dims(image, axis=0))

        # Decode the output using CTC decoder
        decoded_text = ctc_decoder(res, alphabet)
        
        # Return the decoded text as a response
        return JSONResponse(content={"text": decoded_text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

import cv2


app = FastAPI()

model = load_model('model.h5')


def preprocess_image(image, width, height):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    # Further preprocessing if needed
    return image



@app.get("/test/")
async def test():
    return JSONResponse(content={"text": "test back form api"})

@app.post("/upload/")
async def upload_image(file: UploadFile):
    try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = preprocess_image(img,128,32)
            model.predict(np.expand_dims(image, axis=0))

             # Return the text as a response
            return JSONResponse(content={"text": "this is the response"})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from PIL import Image
from tensorflow.keras.models import load_model

app = FastAPI()

#load_model('/model.h5')

@app.post("/upload/")
async def upload_image(file: UploadFile):
    try:
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as temp_image:
            temp_image.write(file.file.read())
            temp_image.seek(0)

            # Process the image and get the text
            text = image_to_text(Image.open(temp_image))

            # Return the text as a response
            return JSONResponse(content={"text": text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

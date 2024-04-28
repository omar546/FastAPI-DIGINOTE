from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from PIL import Image
from tensorflow.keras.models import load_model

import cv2
from mltu.tensorflow.dataProvider import DataProvider
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

app = FastAPI()

model = load_model('model.h5')


from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = None
        self.vocab = ""
        self.height = 32
        self.width = 128
        self.max_text_length = 0
        self.batch_size = 16
        self.learning_rate = 0.0005
        self.train_epochs = 20
        self.train_workers = 20

configs = ModelConfigs()


data_provider = DataProvider(
    dataset=None,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
)

@app.get("/test/")
async def test():
    return JSONResponse(content={"text": "test back form api"})

@app.post("/upload/")
async def upload_image(file: UploadFile):
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as temp_image:
            temp_image.write(file.file.read())
            temp_image.seek(0)
            image = cv2.imread(temp_image)

            preprocessed_image = image.copy()  # Make a copy to avoid modifying the original image
            for preprocessor in data_provider.data_preprocessors:
                preprocessed_image = preprocessor(preprocessed_image)
        


             # Return the text as a response
            return JSONResponse(content={"text": "this is the response"})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

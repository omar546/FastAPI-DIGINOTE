import json
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from keras.models import load_model
from mltu.utils.text_utils import ctc_decoder
from mltu.configs import BaseModelConfigs
from mltu.tensorflow.losses import CTCloss

app = FastAPI()

# Load the fine-tuned YOLO model
checkpoint_path = 'best.pt'
yolo_model = YOLO(checkpoint_path)

# Load the OCR model
configs = BaseModelConfigs.load("configs.yaml")
ocr_model = load_model('model1.h5', custom_objects={'CTCloss': CTCloss}, compile=False)

def preprocess_image(image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    return gray

def threshold_image(image):
    # Apply Otsu's method to get the initial threshold
    _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply Gaussian Adaptive Thresholding using the Otsu threshold as the base
    gaussian_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 91, 2)

    # Combine the Otsu and Gaussian Adaptive Thresholding results
    combined_thresh = cv2.bitwise_and(otsu_thresh, gaussian_thresh)

    return combined_thresh

def perform_morphological_operations(image):
    # Define a structuring element (kernel) for the morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

    # Perform opening (erosion followed by dilation) to remove small noise
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    return opening

def process_image(image):
    # Preprocess the adjusted image
    preprocessed = preprocess_image(image)

    # Threshold the preprocessed image
    thresholded = threshold_image(preprocessed)

    # Perform morphological operations on the thresholded image
    morph_processed = perform_morphological_operations(thresholded)

    # Invert the image to get a white background with black text
    result = cv2.bitwise_not(morph_processed)

    return result

def adjust_images(image, brightness=20, contrast=1.2):
    # Adjust the brightness and contrast
    adjusted = np.clip(image * contrast + brightness, 0, 255).astype(np.uint8)
    return adjusted

def resize_image(image, target_height=96, target_width=1408):
    resized = cv2.resize(image, (target_width, target_height))
    return resized

def extract_ocr_text(image):
    # Prepare the image for OCR
    image = resize_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)

    # Perform OCR using the loaded OCR model
    prediction_text = ocr_model.predict(image)[0]
    prediction_text = np.array([prediction_text])
    text = ctc_decoder(prediction_text, configs.vocab)

    return text

@app.get("/test/")
async def test():
    return JSONResponse(content={"text": "API online"})

@app.post("/upload/")
async def upload_image(file: UploadFile):
    try:
        # Read the image file uploaded
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        
        # Convert numpy array to image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the image using YOLO model
        results = yolo_model(img)
        
        save_dir = '/var/www/html/saved_yolo'
        os.makedirs(save_dir, exist_ok=True)

        # Initialize a list to hold the HTML content
        html_content = []

        # Iterate over results to extract bounding boxes and apply OCR
        for idx, result in enumerate(results):
            # Save the detected image
            output_image_path = os.path.join(save_dir, f'detected_image_{idx}.jpg')
            result.save(output_image_path)
            print(f"Detected image saved to: {output_image_path}")
            
            boxes = result.boxes
            sorted_indices = torch.argsort(boxes.xyxy[:, 1])
            xyxy_sorted = boxes.xyxy[sorted_indices]
            cls_sorted = boxes.cls[sorted_indices]
            
            for idx, detection in enumerate(xyxy_sorted):
                x1, y1, x2, y2 = map(int, detection.tolist())
                cropped_img = img[y1:y2, x1:x2]
                class_name = result.names[cls_sorted[idx].item()]
                
                if class_name == 'Headline' or class_name == 'Subtitle':
                    # Extract OCR text
                    ocr_text = extract_ocr_text(cropped_img)
                    
                    # Determine HTML tag
                    html_tag = 'h1' if class_name == 'Headline' else 'u'
                    html_content.append(f'<{html_tag}>{ocr_text}</{html_tag}>')
                
                elif class_name == 'Figure':
                    # Save the cropped image and create an img tag
                    figure_path = os.path.join(save_dir, f'figure_{x1}_{y1}_{x2}_{y2}.jpg')
                    cv2.imwrite(figure_path, cropped_img)
                    html_content.append(f'<img src="{figure_path}" alt="Figure"/>')
                else:
                    # text_image=process_image(cropped_img)
                    img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                    ret, thresh2 = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150,2))
                    mask = cv2.morphologyEx(thresh2, cv2.MORPH_DILATE, kernel)
                    bboxes = []
                    bboxes_img = cropped_img.copy()
                    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] if len(contours) == 2 else contours[1]
                    for cntr in contours:
                        x,y,w,h = cv2.boundingRect(cntr)
                        cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), (0,0,255), 1)
                        bboxes.append((x,y,w,h))

                    # cv2_imshow(bboxes_img)
                    # # print(bboxes)
                    for bbox in reversed(bboxes):
                        x, y, w, h = bbox
                        roi = cropped_img[y:y+h, x:x+w]  # Extract region of interest (ROI)

                        # cv2_imshow(roi)
                        image = cv2.resize(roi, (1408, 96))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = np.expand_dims(image, axis=0)
                        prediction_text = ocr_model.predict(image)[0]
                        prediction_text = np.array([prediction_text])
                        text = ctc_decoder(prediction_text,configs.vocab)
                        html_content.append(f'<p>{text}</p>')


        # Combine the HTML content into a single string
        html_output = '\n'.join(html_content)

        return JSONResponse(content={"text": html_output})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

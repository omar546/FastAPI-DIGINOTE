import json
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

app = FastAPI()

def preprocess_image(image):
    # image=cv2.imread(image)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (7,7), 0)

    #Convert the image to grayscale
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))

    # Perform opening (erosion followed by dilation) to remove small noise
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening

def adjust_images(images_path, brightness=20, contrast=1.2):

    # Read the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adjust the brightness and contrast
    adjusted = np.clip(image * contrast + brightness, 0, 255).astype(np.uint8)
    return adjusted

def process_image(image):
    # Read the image
    # image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is None:
        print("Error: Failed to load image from", image)
        return

    # Adjust the brightness and contrast
    #adjusted = adjust_images(image)

    # Preprocess the adjusted image
    preprocessed = preprocess_image(image)

    # Threshold the preprocessed image
    thresholded = threshold_image(preprocessed)

    # Perform morphological operations on the thresholded image
    morph_processed = perform_morphological_operations(thresholded)

    # Invert the image to get a white background with black text
    result = cv2.bitwise_not(morph_processed)

    return result

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
# image_path = "/content/photo_2024-06-02_15-30-35.jpg"  # Replace with your image path
# img=process_image(image_path)
# cv2.imwrite("image2.jpg",img)

# Load the fine-tuned YOLO model
checkpoint_path = 'best.pt'
yolo_model = YOLO(checkpoint_path)


@app.get("/test/")
async def test():
    return JSONResponse(content={"text": "API online"})

@app.post("/upload/")
async def upload_image(file: UploadFile):
    try:

        # Test the model on an image (assuming 'model' and 'img' are defined)
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        # Convert numpy array to image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = yolo_model(img)
        save_dir = '/var/www/html/saved_yolo'

        # Initialize TrOCR model and processor
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

        # Initialize a list to hold the HTML content
        html_content = []

        for idx, result in enumerate(results):
            # Save the detected image
            output_image_path = os.path.join(save_dir, f'detected_image.jpg')
            result.save(output_image_path)
            print(f"Detected image saved to: {output_image_path}")
            boxes = result.boxes

            sorted_indices = torch.argsort(boxes.xyxy[:, 1])

            # Sort both xyxy and cls tensors
            xyxy_sorted = boxes.xyxy[sorted_indices]
            cls_sorted = boxes.cls[sorted_indices]
            class_sort=[]
            for element in cls_sorted:
                class_sort.append(element)
            for idx,detection in enumerate(xyxy_sorted):
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, detection.tolist())

                # Crop the image based on bounding box coordinates
                cropped_img = img[y1:y2, x1:x2]

                # Extract the label (class name)
                label = class_sort[idx].item()
                class_name = result.names[label]

                # Determine the HTML tag based on the class name
                if class_name == 'Headline':
                    # Apply OCR to the cropped image using the model
                    pixel_values = processor(images=cropped_img, return_tensors="pt").pixel_values
                    generated_ids = model.generate(pixel_values)
                    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    html_tag = 'h1'
                    html_content.append(f'<{html_tag}>{text}</{html_tag}>')
                elif class_name == 'Subtitle':
                    # Apply OCR to the cropped image using the model
                    pixel_values = processor(images=cropped_img, return_tensors="pt").pixel_values
                    generated_ids = model.generate(pixel_values)
                    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    # Use underline tag for subtitle
                    html_content.append(f'<u>{text}</u>')
                elif class_name == 'Figure':
                    # Save the cropped image and create an img tag
                    figure_path = os.path.join(save_dir, f'figure_{x1}_{y1}_{x2}_{y2}.jpg')
                    cv2.imwrite(figure_path, cropped_img)
                    html_content.append(f'<img src="{figure_path}" alt="Figure"/>')
                else:
                    img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                    ret, thresh2 = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 2))
                    mask = cv2.morphologyEx(thresh2, cv2.MORPH_DILATE, kernel)
                    bboxes = []
                    bboxes_img = cropped_img.copy()
                    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] if len(contours) == 2 else contours[1]
                    for cntr in contours:
                        x, y, w, h = cv2.boundingRect(cntr)
                        cv2.rectangle(bboxes_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        bboxes.append((x, y, w, h))

                    for bbox in reversed(bboxes):
                        x, y, w, h = bbox
                        roi = cropped_img[y:y + h, x:x + w]  # Extract region of interest (ROI)
                        image = roi
                        image = cv2.resize(image, (1408, 96))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = np.expand_dims(image, axis=0)

                        pixel_values = processor(images=image, return_tensors="pt").pixel_values

                        generated_ids = model.generate(pixel_values)
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        html_content.append(f'<p>{generated_text}</p>')

        # Filter out lines that start with '0'
        filtered_html_content = [line for line in html_content if not line.startswith('0')]

        # Combine the HTML content into a single string
        html_output = '\n'.join(filtered_html_content)

        return JSONResponse(content={"text": html_output})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

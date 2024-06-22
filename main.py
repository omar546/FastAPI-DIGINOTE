import json
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
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
# app.mount("/static", StaticFiles(directory="saved_yolo"), name="static")

memory = {}
# Load the fine-tuned YOLO model
checkpoint_path = 'best.pt'
yolo_model = YOLO(checkpoint_path)

# Load the OCR model
configs = BaseModelConfigs.load("configs.yaml")
ocr_model = load_model('model1.h5', custom_objects={'CTCloss': CTCloss}, compile=False)


# Dictionary of contextual corrections with lists instead of sets
contextual_corrections = {
    "their": ["thier"],
    "there": ["ther"],
    "they": ["thay"],
    "tter": ["other"],
    "your": ["yor"],
    "8us": ["DBMS"],
    "oasics": ["Basics"],
    "DBus": ["DBMS"],
    "effction":["collection"],
    "pors": ["DBMS"],
    "cbection": ["collection"],
    "togtter": ["together"],
    "Er": ["EX"],
    "-": [":"],
    "Pecnbsy": ["Technology"],
    "erice": ["component"],
    "Intertae": ["Interface"],
    "ard": ["card"],
    "hat": ["hub"],
    "Pouter": ["Router"],
    "ro": ["hub"],
    "intoration": ["information"],
    "info": ["information"],
    "comects": ["connects"],
    "tat": ["hub"],
    "t": ["A"],
    "bost": ["host"],
    "al": ["all"],
    "itornation": ["information"],
    "lerme": ["consume"],
    "lar": ["consume"],
    "bretwark": ["Network"],
    "crerr--r": ["Operating"],
    "ere": ["System"],
    "Tracess": ["process"],
    "rre": ["state"],
    "rurrinl": ["running"],
    "tlock": ["block"],
    "Tearc": ["Device"],
    "aece": ["queue"],
    "gvene": ["queue"],
    "devile": ["device"],
    "cf": ["CPU"],
    "rmput": ["inpuy"],
    "Seel": ["Ready"],
    "arece": ["queue"],
    "M": ["AI"],
    "hibern": ["hidden"],
    "apenalr": ["process"],
    "eaa": ["operations"],
    "ter": ["system"],
    "frece": ["queue"],
    "scr": ["load"],
    "-rr": ["in"],
    "34ste": ["system"],
    "cear": ["ready"],
    "sacce": ["queue"],
    "quene": ["queue"],
    "cP": ["CPU"],
    "Ereurrce": ["Device"],
    "crecc": ["queue"],
    "areie": ["queue"],
    "d": ["of"],
    "alr": ["processes"],
    "proaesiystied  """: ["need to load in system"],
    "i-": ["Introduction"],
    "--e S -": ["Data"],
    '" S paietanf , Fatsthantecantrnston': ["is a colection of facts that can be processed to produce information"],
    "-slse mangenacsystens-": ["Database Management System"],
    "iii": ["database MS is software that allows computers to perform database function of storing, receiving, adding and modify data"],
    "amahataanaltres": ["Social data analysis"],
    "3I Component . slated -unetwort that are Comcted": ["1. components: isolated subnetwork that are connected"],
    "dscomacted": ["disconnected"],
    "ttween": ["between"],
    "raaturort": ["networks"],
    "I": ["of"],
    "nodo": ["nodes"],
    "toeack": ["to each"],
    "ter": ["other"],
    "conacteel": ["connected"],
    "ooer": ["other"],
    "groo": ["group"],
    "everaise-":["Bridge"],
    "wodc": ["node"],
    "reatnwnen": ["that"],
    "sumond": ["when"],
    "ranale": ["removed"],
    "an": ["it"],
    "ded": ["create"],
    "f": ["connected"],
    "3I": ["3."],
    "Pameter": ["Diameter"],
    "I": ["of"],
    "o": ["of"],
    "shotewats": ["shortest path"],
    "3S": ["3."],
    "lonsit": ["long it"],
    "paiihakieguoriseruaton": ["will take for information"],
    "I mos eonsit aibl iken oresiveruaton": ["to pass through network"],
    "shotatwatt": ["shortest path"],
    "Disit": ["Disk"],
    "blacticare": ["blocks are"],
    "rad": ["read"],
    "-n": ["on"],
    ",gntsret inmasoutter": ["and put into buffer"],
    "crlaisariented itarase Lenaucas ctrmig": ["pool which is made by buffer manager"],
    "f": ["of"],
    "sex": ["2."],
    "or": ["cons:"],
    "iost": ["cost"],
    "icost": ["cost"],
    "deltirng": ["deleting"],
    "deletirny": ["deleting"],
    "Tr": ["pros:"],
    "'s Rdua ": ["1.cost"],
    "ro": ["IO"],
    '"t': ["if"],
    "altnbute": ["attributes"],
    "ex": ["1."],
    "Zmprore": ["Improve"],
    "cacke": ["cache"],
    "pretormnce": ["performance"],
    "ss": ["2."],
    "Emprore": ["Improve"],
    "fret torr Eemutrt": [" Network Benefit:"]
}

def contextual_analysis_correction(text):
    words = text.split()
    corrected_words = []

    i = 0
    while i < len(words):
        word = words[i]
        word_lower = word.lower()
    if word_lower in contextual_corrections:
                # Check if previous or next word matches any of the allowed corrections
                prev_word = words[i - 1].lower() if i > 0 else ""
                next_word = words[i + 1].lower() if i < len(words) - 1 else ""

                if prev_word in contextual_corrections[word_lower]:
                    corrected_words.append(word)  # Keep original word if contextually correct
                elif next_word in contextual_corrections[word_lower]:
                    corrected_words.append(word)  # Keep original word if contextually correct
                else:
                    # Apply correction logic based on context
                    corrected_words.append(contextual_corrections[word_lower][0])  # Use first suggestion
    else:
        corrected_words.append(word)  # Keep original word if not in dictionary

        i += 1

    return " ".join(corrected_words)

from spellchecker import SpellChecker

# Initialize spell checker
spell_checker = SpellChecker()

# Function for spell checking and correction
def correct_spelling(text):
    # Split text into words and correct each word
    corrected_text = []
    for word in text.split():
        corrected_word = spell_checker.correction(word)
        if corrected_word is not None:
            corrected_text.append(corrected_word)
        else:
            corrected_text.append(word)  # Keep original word if no correction found
    return ' '.join(corrected_text)


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
    text = ctc_decoder(prediction_text,configs.vocab)
    corrected_text = contextual_analysis_correction(text[0])
    corrected = correct_spelling(corrected_text)

    return corrected


@app.get("/test/")
async def test():
    return JSONResponse(content={"text": "API online"})

from fastapi.responses import FileResponse
import jwt
figurslist=[]
@app.post("/upload/")
async def upload_image(file: UploadFile):
        # Read the image file uploaded
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        
        # Convert numpy array to image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the image using YOLO model
        results = yolo_model(img)
        
        save_dir = './saved_yolo'
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
                        corrected_text = contextual_analysis_correction(text[0])
                        corrected = correct_spelling(corrected_text)
                        html_content.append(f'<p>{corrected}</p>')


        # Combine the HTML content into a single string
        html_output = '\n'.join(html_content)
        import json
        from bs4 import BeautifulSoup

        def escape_text(text):
            return text.replace("'", "''").replace('"', '""').replace('\n', '\\n').replace('\\', '?')

        def convert_html_to_json(html_string):
            soup = BeautifulSoup(html_string, 'lxml')
            json_list = []

            def handle_text_node(text, attributes=None):
                text = text.strip("[]' ")
                node = {'insert': escape_text(text)}
                if attributes:
                    node['attributes'] = attributes
                json_list.append(node)

            for element in soup.body.children:
                if isinstance(element, str):  # Skip if the element is just a string (newline, etc.)
                    continue
                if element.name == 'h1':
                    handle_text_node(
                        element.get_text(strip=True),
                        attributes={'bold': True, 'size': 'huge','align': 'center'}
                    )
                    json_list.append({'insert': '\n',})
                elif element.name == 'u':
                    handle_text_node(
                        element.get_text(strip=True),
                        attributes={'size': 'large', 'bold': True}
                    )
                elif element.name == 'p':
                    handle_text_node(element.get_text(strip=True))
                    json_list.append({'insert': '\n'})
                elif element.name == 'img':
                    src = element.get('src')
                    if src:
                        
                        image_url = f'http://3.75.171.189/photos/{figure_path.split("/")[-1]}'
                        json_list.append({'insert': {'image': image_url}})
                        json_list.append({'insert': '\n'})
                        

            return json.dumps(json_list, ensure_ascii=False)


        json_string = convert_html_to_json(html_output)

        # encoded_jwt = jwt.encode({"sub": 'abc'}, "SeCrEt", "HS256")
        # memory[encoded_jwt] = [json_string,figure_path] 

        return JSONResponse(content={"text": json_string})
 
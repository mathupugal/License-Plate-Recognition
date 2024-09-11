"""
Flask web application for license plate recognition and text extraction
using a trained CNN model and EasyOCR.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import easyocr
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
# Set the upload folder path
UPLOAD_FOLDER = 'static/uploads/'
TEXT_FOLDER = 'static/txt/'  # Folder to save text files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEXT_FOLDER'] = TEXT_FOLDER

# Create the folders if they do not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEXT_FOLDER, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('license_plate_recognition_model.h5')  # pylint: disable=no-member

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def process_image(img_path):
    """
    Process the image for license plate recognition and text extraction.

    Args:
        img_path (str): The path to the image file.

    Returns:
        str: The extracted text from the image.
    """
    # Load the image
    image = cv2.imread(img_path)  # pylint: disable=no-member

    # Preprocess the image for the model
    img_height, img_width = 150, 150
    img_array = cv2.resize(image, (img_width, img_height))  # pylint: disable=no-member
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict with the model
    _ = model.predict(img_array)

    # Extract text using EasyOCR
    result = reader.readtext(img_path)
    detected_text = " ".join([res[1] for res in result])

    # Save the extracted text to a single file
    text_file_path = os.path.join(app.config['TEXT_FOLDER'], 'all_extracted_texts.txt')
    with open(text_file_path, 'a', encoding='utf-8') as txt_file:  # Open in append mode
        txt_file.write(f"{os.path.basename(img_path)}: {detected_text}\n")

    return detected_text

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handle file uploads and text extraction.

    Returns:
        str: The rendered template for the index page with extracted texts.
    """
    if request.method == 'POST':
        # Check if files are uploaded
        if 'files' not in request.files:
            return redirect(request.url)

        files = request.files.getlist('files')

        # If no files are selected, redirect
        if not files:
            return redirect(request.url)

        image_paths = []
        extracted_texts = []

        for file in files:
            if file and file.filename:
                # Save the file to the upload folder
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                # Process the image and extract the text
                extracted_text = process_image(file_path)

                image_paths.append(file_path)
                extracted_texts.append(extracted_text)

        return render_template('index.html', images=image_paths, texts=extracted_texts)

    return render_template('index.html')

@app.route('/workflow')
def workflow():
    """
    Render the workflow page.

    Returns:
        str: The rendered template for the workflow page.
    """
    return render_template('workflow.html')

if __name__ == "__main__":
    app.run(debug=True)

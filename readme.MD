<h1 align="center">License Plate Recognition</h1>

This project implements a License Plate Recognition (LPR) using image processing and machine learning techniques. It is a Flask-based web application designed to detect and extract text from license plate images using a trained Convolutional Neural Network (CNN) model and EasyOCR.

This repository contains the code and resources for a License Plate Recognition system. The project is divided into three main parts: 
 1) Training and Evaluation
 2) Flask Web Application 
 3) Single Image Processing

**1. Training and Evaluation**
The "Training_and_Evaluation" directory contains all the resources and code required to train and evaluate the license plate recognition model. The workflow involves:

Image Preprocessing: Images are processed and prepared for training using various techniques such as gray scale conversion, edge detection, and contouring.

Model Training: The processed images are used to train a Convolutional Neural Network (CNN) model to recognize and extract text from license plates.

Evaluation: The trained model is evaluated on separate datasets (training, validation, and test) to measure its performance.

Prediction: Predictions for each dataset are stored in corresponding text files.

**2. Flask Web Application**
The "flask_webpage" directory contains the code for deploying the trained model as a web application using Flask. The application allows users to upload images, process them using the pre-trained model, and return the extracted text. Key components include:

Main Flask Application

HTML Templates

Static Files

**3. Single Image Processing**
The "working_process_for_1_image" directory provides a separate notebook for processing a single image. This is useful for testing the workflow on individual images before scaling it up.


**Project Structure:**

    LicensePlateRecognition/
        1)training_and_evaluation/
            dataset_1/                          # Input folder with raw images
            dataset_2/                          # Input folder with raw images       
            model.ipynb                         # Notebook for image preprocessing, training, testing, and evaluation  
            new_folder_new/                     # Folder for images after initial processing        
            annotated_result/                   # Folder for images with annotations        
            unannotated_result/                 # Folder for images without annotations       
            train_folder/                       # Folder for training images
            val_folder/                         # Folder for validation images       
            test_folder/                        # Folder for test images
            train_predictions.txt               # File containing predictions for training images
            val_predictions.txt                 # File containing predictions for validation images
            test_predictions.txt                # File containing predictions for test images
            license_plate_recognition_model.h5  # Trained model file


        2)flask_webpage/
            main.py                             # Main Flask application file
            license_plate_recognition_model.h5  # Pre-trained CNN model
            templates/
                index.html                      # Main page for uploading images
                workflow.html                   # Workflow description page
            static/
                uploads/                        # Directory for storing uploaded images
                txt/                            # Directory for storing extracted text
                    1.jpg                       # Background image used in the HTML
                    2.jpg                       # Background image used in the HTML
        
        3)working_process_for_1_image/
            dataset/                            # Folder containing the image to be processed
            working_process.ipynb               # Notebook for single image processing
    
    
        readme.md                               # Project overview and instructions    
        .gitignore                              # Files and directories to ignore in Git    
        requirements.txt                        # List of dependencies for the project
        dockerfile                              

**How to Run**
1. Training the Model (model.ipynb) 
2. Running the Flask Web Application (main.py)



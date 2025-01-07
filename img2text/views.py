from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from PIL import Image
import pytesseract
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load and preprocess the MNIST dataset (as per your code)
def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    return (train_images, train_labels), (test_images, test_labels)

# Build the CNN model
def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Train the model
def train_model(model, train_images, train_labels, test_images, test_labels):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, batch_size=64,
              validation_data=(test_images, test_labels), verbose=0)

# Evaluate the model
def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    return test_acc

# Extract text from the image
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text.strip()

# Save the extracted text to a file
def save_text_to_file(text, classification, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n\n\n\n')
        f.write(f"The extracted text is classified as: {classification}\n")

# Classify the type of text
def classify_text_type(text):
    if any(char.isdigit() for char in text) or len(text) > 50:  
        return "Typed"
    else:
        return "Handwritten"

def image_upload_view(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        if uploaded_file:
            # Ensure the uploads directory exists
            uploads_dir = 'uploads'
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)
            
            # Save the uploaded file to the uploads directory
            file_path = os.path.join(uploads_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            # Extract text using OCR (example with pytesseract)
            from PIL import Image
            import pytesseract

            try:
                extracted_text = pytesseract.image_to_string(Image.open(file_path))
            except Exception as e:
                extracted_text = "Error extracting text: " + str(e)

            # Append the extracted text to image.txt
            output_file_path = os.path.join(uploads_dir, 'image.txt')
            with open(output_file_path, 'w') as output_file:
                # output_file.write(f"Extracted from {uploaded_file.name}:\n")
                output_file.write(extracted_text)
                output_file.write("\n" + "-"*50 + "\n")

            # Read the full contents of image.txt to display in the results
            with open(output_file_path, 'r') as output_file:
                full_text = output_file.read()

            # Render result.html with the full content of image.txt
            return render(request, 'result.html', {
                'extracted_text': extracted_text,
                'full_text': full_text
            })
        else:
            return render(request, 'upload.html', {'error': 'No file uploaded.'})
    return render(request, 'upload.html')





from flask import Flask, request, render_template,url_for
import numpy as np
import tensorflow as tf
import cv2
import os
from werkzeug.utils import secure_filename

# Load the Keras model
model = tf.keras.models.load_model('model/trained_model.keras')

# List of class names (adjust according to your model's output classes)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Create Flask app
app = Flask(__name__,static_url_path='/static')

#Folder to upload image 
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_PATH'] = 16 * 1024 * 1024 

# Route to render index.html
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle image upload and prediction
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return "No selected file"
        
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Read image
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Preprocess image for prediction
            img = cv2.resize(img, (128, 128))  # Resize image to match model's expected sizing
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size used for training

            # Predict class
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions)
            predicted_class = class_names[predicted_class_idx]

            # Render result in HTML
            return render_template("index.html", prediction=predicted_class, image_url=url_for('static', filename='uploads/' + filename))

    return "Prediction failed"

if __name__ == '__main__':
    app.run(debug=True)

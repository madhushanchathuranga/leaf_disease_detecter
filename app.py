from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Load the model
model = load_model('model\densenet121_corn_leaf_disease.h5')
class_names = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = preprocess_image(filepath)
            predictions = model.predict(img)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = round(np.max(predictions) * 100, 2)
            return render_template('result.html', 
                                   prediction=predicted_class, 
                                   confidence=confidence, 
                                   image_url=url_for('static', filename='uploads/' + filename))
    return render_template('uploads.html')

if __name__ == '__main__':
    app.run(debug=True)

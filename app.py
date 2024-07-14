from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
import tensorflow as tf
import base64

app = Flask(__name__)

model = load_model('unet_model1.keras')

def process_image(image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img, (512, 512))  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.uint8)

    pred_mask = model.predict(np.expand_dims(img, axis=0))[0]
    pred_mask = cv2.convertScaleAbs(pred_mask, alpha=380, beta=0)

    retval, buffer = cv2.imencode('.png', pred_mask)
    pred_mask_base64 = base64.b64encode(buffer).decode('utf-8')

    return pred_mask_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            pred_mask_base64 = process_image(file_path)
            return jsonify({'result': pred_mask_base64})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.run(debug=True)
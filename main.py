import os, sys
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf
from google.cloud import storage
import io
import requests

app = Flask(__name__)
#################################
# TODO:                         #
# Set GCS_BUCKET and MODEL_NAME #
#################################
GCS_BUCKET = ''
MODEL_NAME = ''

MODEL_DOWNLOAD_FOLDER = '/tmp'
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

model_path = os.path.join(MODEL_DOWNLOAD_FOLDER, MODEL_NAME)
storage_client = storage.Client()
bucket = storage_client.get_bucket(GCS_BUCKET)
blob = bucket.get_blob(MODEL_NAME)
blob.download_to_filename(model_path)
model = load_model(model_path)

graph = tf.get_default_graph()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return redirect(url_for('predict'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file.')
            return redirect(url_for('predict'))
        file = request.files.get('file')
        if file.filename == '':
            flash('No file.')
            return redirect(url_for('predict'))
        if file and is_allowed_file(file.filename):
            gcs = storage.Client()
            bucket = gcs.get_bucket(GCS_BUCKET)
            blob = bucket.blob(file.filename)
            blob.upload_from_string(
                file.read(),
                content_type=file.content_type
            )

            file_url = blob.public_url
            img = Image.open(io.BytesIO(requests.get(file_url).content)).convert('RGB')
            img = img.resize((150, 150))
            x = np.array(img, dtype=np.float32)
            x = x / 255.
            x = x.reshape((1,) + x.shape)
            
            global graph
            with graph.as_default():
                pred = model.predict(x, batch_size=1, verbose=0)
                score = pred[0][0]
                if(score >= 0.5):
                    person = 'Smalling'
                else:
                    person = 'Obama'
                    score = 1 - score

            resultmsg = '[{}] {:.4%} Sure.'.format(person, score)
            
            return render_template('result.html', resultmsg=resultmsg, filepath=file_url)
    return render_template('predict.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()
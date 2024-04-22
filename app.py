from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from TextExtraction import TextExtraction

#from TextExtraction import TextExtraction

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder

textExtraction = TextExtraction()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)
        print(f"Uploaded Image: {img}")

        textExtraction.processImage(img)
        return render_template('image_render.html', img=img)
    return render_template('image_render.html')
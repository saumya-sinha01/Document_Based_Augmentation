from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from TextExtraction import TextExtraction

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
        input_image = os.path.join(app.config['UPLOAD'], filename)

        textExtraction.processImage(input_image)
        return render_template('image_render.html', img=input_image)

    return render_template('image_render.html')

@app.route('/searchtext', methods=['POST'])
def search():
    searchText = request.form['searchText']
    input_image = textExtraction.processed_image

    if textExtraction.search(searchText):
        return render_template('image_render.html', foundText='Text Found!', img=input_image, searchText=searchText)
    else:
        return render_template('image_render.html', foundText='Not Found!', img=input_image, searchText=searchText)



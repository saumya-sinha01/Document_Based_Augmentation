from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from TextExtraction import TextExtraction
from Data_Augmentation import Data_Augmentation
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder
textExtraction = TextExtraction()
data_Augmentation = Data_Augmentation()

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
    input_image = textExtraction.input_image

    if textExtraction.search(searchText):
        return render_template('image_render.html', foundText='Text Found!', img=input_image, searchText=searchText)
    else:
        return render_template('image_render.html', foundText='Not Found!', img=input_image, searchText=searchText)

@app.route('/removeText', methods=['POST'])
def remove():
    text_to_be_removed = request.form['removeText']
    full_json_data = textExtraction.full_image_json
    full_image_path = textExtraction.input_image

    augmented_image_filepath = data_Augmentation.removeTextFromFullImage(text_to_be_removed, full_json_data, full_image_path)
    return render_template('image_render.html', deletedText='Deleted All Occurences!',
                           removedText=text_to_be_removed, img=full_image_path, editedImg=augmented_image_filepath)

    # extracted_json_list = textExtraction.extracted_json_list
    # data_Augmentation.removeText(removeText, extracted_json_list, textExtraction.cropped_image_files)




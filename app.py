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
    input_image = textExtraction.processed_image

    if textExtraction.search(searchText):
        return render_template('image_render.html', foundText='Text Found!', img=input_image, searchText=searchText)
    else:
        return render_template('image_render.html', foundText='Not Found!', img=input_image, searchText=searchText)

@app.route('/removeText', methods=['POST'])
def remove():
    removeText = request.form['removeText']
    extracted_json_list = textExtraction.extracted_json_list
    data_Augmentation.removeText(removeText, extracted_json_list, textExtraction.cropped_image_files)
    # data_Augmentation.load_JSON(extracted_json_list, removeText)
    # text_coordinates = data_Augmentation.get_coordinates()
    # text_to_be_removed = "saumya"
    #
    # # Loop through each cropped image file path
    # for image_file_path in textExtraction.cropped_image_files:
    #     # Open the image file
    #     image_file = image_file_path
    #
    #     # Call the reconstruct_table method for each image file
    #     data_Augmentation.reconstruct_table(image_file, text_to_be_removed)
    # # image_file_path = textExtraction.cropped_image_files
    # # image_file = Image.open(image_file_path)
    # # data_Augmentation.reconstruct_table(image_file_path, text_to_be_removed)






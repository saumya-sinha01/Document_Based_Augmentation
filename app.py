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

        # Process the image with your extraction logic
        textExtraction.processImageV2(input_image)

        # Render the template with the uploaded image
        return render_template('image_render.html', img=input_image)

    # On GET request, just render the template without any image
    return render_template('image_render.html')

# @app.route('/displayKeyValuePairs', methods=['POST'])
# def displayKeyValuePairs():
#     input_image = textExtraction.input_image
#     extracted_kv_pairs = textExtraction.extracted_key_value_pairs[0][0]
#     return render_template('image_render.html', img=input_image, kv_tuples=extracted_kv_pairs)

@app.route('/displayKeyValuePairs', methods=['POST'])
def displayKeyValuePairs():
    input_image = textExtraction.input_image
    extracted_kv_pairs = textExtraction.extracted_key_value_pairs[0]
    labeled_images = textExtraction.labeled_images
    return render_template('image_render.html', img=input_image, labeled_images=labeled_images, kv_tuples=extracted_kv_pairs, currentSection='kvpSection')


@app.route('/searchSection', methods=['POST'])
def search():
    searchText = request.form['searchText']
    input_image = textExtraction.input_image
    image_found, searched_image = textExtraction.search(searchText)

    if image_found:
        return render_template('image_render.html', foundText='Text Found!', img=searched_image, searchText=searchText, currentSection='searchSection')
    else:
        return render_template('image_render.html', foundText='Not Found!', img=input_image, searchText=searchText, currentSection='searchSection')


@app.route('/replaceText', methods=['POST'])
def replace():
    input_image = textExtraction.input_image
    input_text = request.form['replaceText']
    replacement_text = request.form['replacementText']

    isReplaced, orig_img_visualized_filename, orig_img_replaced_text_filename = textExtraction.replace(input_text, replacement_text, False)

    if isReplaced:
        return render_template('image_render.html', replaceText='Text Replaced!', img=orig_img_visualized_filename,
                               replaced_img=orig_img_replaced_text_filename, currentSection='replaceSection')
    else:
        return render_template('image_render.html', replaceText='Input Text Not Found!', img=input_image,
                               replaced_img=None, currentSection='replaceSection')

@app.route('/replaceAllText', methods=['POST'])
def replaceAllText():
    input_image = textExtraction.input_image
    input_text = request.form['replaceText']
    replacement_text = request.form['replacementText']

    isReplaced, orig_img_visualized_filename, augmented_img_visualized_filename = textExtraction.replace(input_text, replacement_text, True)

    if isReplaced:
        return render_template('image_render.html', replaceText='Text Replaced!', img=orig_img_visualized_filename,
                               replaced_img=augmented_img_visualized_filename, currentSection='replaceSection')
    else:
        return render_template('image_render.html', replaceText='Input Text Not Found!', img=input_image,
                               replaced_img=None, currentSection='replaceSection')


@app.route('/removeText', methods=['POST'])
def remove():
    input_text = request.form['removeText']
    input_image = textExtraction.input_image
    isDeleted, orig_img_visualized_filename, augmented_img_visualized_filename = textExtraction.deleteText(input_text)

    if isDeleted:
        return render_template('image_render.html', deletedText='Text Deleted!', img=orig_img_visualized_filename,
                               deleted_img=augmented_img_visualized_filename, currentSection='deleteSection')
    else:
        return render_template('image_render.html', deletedText='Text to delete not found!', img=input_image,
                               deleted_img=None, currentSection='deleteSection')

@app.route('/removeAllText', methods=['POST'])
def removeAllText():
    input_text = request.form['removeText']
    input_image = textExtraction.input_image
    isDeleted, orig_img_visualized_filename, augmented_img_visualized_filename = textExtraction.deleteText(input_text, True)

    if isDeleted:
        return render_template('image_render.html', deletedText='Text Deleted!', img=orig_img_visualized_filename,
                               deleted_img=augmented_img_visualized_filename, currentSection='deleteSection')
    else:
        return render_template('image_render.html', deletedText='Text to delete not found!', img=input_image,
                               deleted_img=None, currentSection='deleteSection')

@app.route('/getLatestImage', methods=['GET'])
def getLatestImage():
    input_image = textExtraction.input_image
    # Render the template with the uploaded image
    return render_template('image_render.html', img=input_image)



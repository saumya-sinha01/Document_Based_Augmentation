from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from TextExtraction import TextExtraction
from Data_Augmentation import Data_Augmentation

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

        # Process the image with your extraction logic
        textExtraction.processImage(input_image)

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

    if textExtraction.search(searchText):
        return render_template('image_render.html', foundText='Text Found!', img=input_image, searchText=searchText, currentSection='searchSection')
    else:
        return render_template('image_render.html', foundText='Not Found!', img=input_image, searchText=searchText, currentSection='searchSection')


# @app.route('/searchtext', methods=['POST'])
# def search():
#     searchText = request.form['searchText']
#     input_image = textExtraction.input_image
#
#     if textExtraction.search(searchText):
#         return render_template('image_render.html', foundText='Text Found!', img=input_image, searchText=searchText)
#     else:
#         return render_template('image_render.html', foundText='Not Found!', img=input_image, searchText=searchText)

@app.route('/removeText', methods=['POST'])
def remove():
    text_to_be_removed = request.form['removeText']
    full_json_data = textExtraction.full_image_json
    full_image_path = textExtraction.input_image

    augmented_image_filepath = data_Augmentation.removeTextFromFullImage(text_to_be_removed, full_json_data, full_image_path)
    return render_template('image_render.html', deletedText='Deleted All Occurrences!',
                           removedText=text_to_be_removed, img=full_image_path, editedImg=augmented_image_filepath)

@app.route('/replaceText', methods=['POST'])
def replace():
    text_to_be_replaced = request.form['replaceText']
    full_json_data = textExtraction.full_image_json
    full_image_path = textExtraction.input_image
    replacement_text = request.form['replacementText']
    augmented_image_path_replacement = data_Augmentation.replaceText(text_to_be_replaced, full_json_data, full_image_path, replacement_text)
    return render_template('image_render.html', replacedText_Msg='Replaced Occurrence!', replacementText=replacement_text,
                           replaceText=text_to_be_replaced, img=full_image_path, editedImg=augmented_image_path_replacement)

@app.route('/replaceAllText', methods=['POST'])
def replaceAll():
    text_to_be_replacedAll = request.form['replaceAllText']
    full_json_data = textExtraction.full_image_json
    full_image_path = textExtraction.input_image
    replacementAll_text = request.form['replacementAllText']
    augmented_image_path_replacementAll = data_Augmentation.replaceAllText(text_to_be_replacedAll, full_json_data,
                                                                           full_image_path, replacementAll_text)
    return render_template('image_render.html', replacedAllText_Msg='Replaced All Occurrences!',
                           replacementAllText=replacementAll_text,
                           replaceAllText=text_to_be_replacedAll, img=full_image_path,
                           editedImg=augmented_image_path_replacementAll)

# @app.route('/KVPSearch', methods=['POST'])
# def kvp_search():
#     key_value_pairs = textExtraction.display_key_value_pairs()
#     return render_template('image_render.html', key_value_pairs=key_value_pairs, img=textExtraction.input_image)
@app.route('/KVPSearch', methods=['POST'])
def kvp_search():
    key_value_pairs = textExtraction.display_key_value_pairs()
    return render_template('image_render.html', key_value_pairs=key_value_pairs, img=textExtraction.input_image, currentSection='kvpSection')

if __name__ == '__main__':
    app.run(debug=True)

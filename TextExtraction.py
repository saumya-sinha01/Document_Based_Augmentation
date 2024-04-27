import os, cv2, json
from collections import defaultdict

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from ultralyticsplus import YOLO, render_result


class TextExtraction:

    processed_image = None
    def __init__(self):
        self.image_upload_folder = 'static\\uploads\\'
        os.environ['USE_TORCH'] = '1'

        self.extracted_json_list = []
        self.cropped_image_files = []
        self.full_image_json = None
        self.image_file_name = None

        #Initialize and Set Model Parameters...
        self.yolo_model = YOLO('keremberke/yolov8m-table-extraction')
        self.yolo_model.overrides['conf'] = 0.25  # NMS confidence threshold
        self.yolo_model.overrides['iou'] = 0.45  # NMS IoU threshold
        self.yolo_model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        self.yolo_model.overrides['max_det'] = 1000  # maximum number of detections per image

        #Initialize OCR Predictor
        self.predictor = ocr_predictor(pretrained=True)

    def search(self, searchText):
        if not searchText:
            return False

        searchText = searchText.lower()
        if searchText in self.dict:
            return True

        return False

    def processImage(self, input_image):
        self.processed_image = input_image


        # perform inference
        results = self.yolo_model.predict(input_image)

        #Load the image:
        original_image = cv2.imread(input_image)

        #Extract the filename..
        image_file_name = input_image.replace(self.image_upload_folder, '').replace('.png', '')

        parsed_json_list = []
        result_count = 0
        box_count = 0
        for result in results:
            result_count += 1
            for captured_box in result.boxes:
                box_count += 1
                box = captured_box.xyxy

                # Extract the coordinates of the top-left and bottom-right corners
                x1, y1, x2, y2 = map(int, box[0])
                cropped_image = original_image[y1:y2, x1:x2]

                # Save or display the cropped image
                cropped_image_filename = self.image_upload_folder + image_file_name + '_' + str(result_count) + '_' + str(box_count) + '.png'
                cv2.imwrite(cropped_image_filename, cropped_image)

                extracted_json = self.extractText(cropped_image_filename)
                self.extracted_json_list.append(extracted_json)
                self.cropped_image_files.append(cropped_image_filename)
                self.parse_extracted_json(extracted_json)

    def parse_extracted_json(self, extracted_json):
        self.dict = defaultdict()
        dict = json.loads(extracted_json)
        value_dict = dict['pages'][0]
        block_list = value_dict['blocks']
        for block_value in block_list:
            word_dict_list = block_value['lines'][0]
            word_list = word_dict_list['words']
            for word_dict in word_list:
                word_value = word_dict['value'].lower()
                if word_value in self.dict:
                    self.dict[word_value].append(word_dict)
                else:
                    self.dict[word_value] = [word_dict]

    def extractText(self, cropped_table_image):
        # Read the file
        doc = DocumentFile.from_images(cropped_table_image)

        # Perform OCR on the document
        result = self.predictor(doc)

        # JSON export
        json_export = result.export()

        # Fields to remove
        fields_to_remove = ['confidence', 'page_idx', 'orientation', 'language', 'artefacts']

        # Remove the specified fields
        self.remove_fields(json_export, fields_to_remove)

        # Remove 'geometry' from 'blocks' and 'lines'
        for page in json_export['pages']:
            for block in page['blocks']:
                if 'geometry' in block:
                    del block['geometry']
                for line in block.get('lines', []):
                    if 'geometry' in line:
                        del line['geometry']

        # Convert the modified data back to JSON
        modified_json_full = json.dumps(json_export, separators=(',', ':'))
        return modified_json_full

    # Define a function to remove fields recursively
    def remove_fields(self, obj, fields):
        if isinstance(obj, list):
            for item in obj:
                self.remove_fields(item, fields)
        elif isinstance(obj, dict):
            for key in list(obj.keys()):
                if key in fields:
                    del obj[key]
                else:
                    self.remove_fields(obj[key], fields)

    # Function to remove 'geometry' key from 'blocks' and 'lines'
    def remove_geometry(self, data):
        if isinstance(data, list):
            for item in data:
                self.remove_geometry(item)
        elif isinstance(data, dict):
            if 'geometry' in data:
                del data['geometry']
            for key, value in data.items():
                self.remove_geometry(value)



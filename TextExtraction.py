import os, cv2, json, re
from collections import defaultdict
import pytesseract
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from ultralyticsplus import YOLO, render_result

from KeyValuePairExtractor import KeyValuePairExtractor


class TextExtraction:

    input_image = None
    def __init__(self):
        self.image_upload_folder = 'static\\uploads\\'
        os.environ['USE_TORCH'] = '1'

        self.extracted_json_list = []
        self.extracted_key_value_pairs = []
        self.cropped_image_files = []
        self.full_image_json = None

        #Initialize and Set Model Parameters...
        self.yolo_model = YOLO('keremberke/yolov8m-table-extraction')
        self.yolo_model.overrides['conf'] = 0.25  # NMS confidence threshold
        self.yolo_model.overrides['iou'] = 0.45  # NMS IoU threshold
        self.yolo_model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        self.yolo_model.overrides['max_det'] = 1000  # maximum number of detections per image

        #Initialize OCR Predictor
        self.predictor = ocr_predictor(pretrained=True)
        self.kvExtractor = KeyValuePairExtractor()

    def search(self, searchText):
        if not searchText:
            return False

        searchText = searchText.lower()
        if searchText in self.dict:
            return True

        return False

    def processImage(self, input_image):
        self.input_image = input_image

        #Extract json text for the full image..
        self.full_image_json = self.extractText(input_image)

        #Crop Images to predict the table..
        # perform inference
        results = self.yolo_model.predict(input_image)

        #Load the image:
        original_image = cv2.imread(input_image)

        #Extract the filename..
        image_file_name = input_image.replace(self.image_upload_folder, '').replace('.png', '')

        result_count = 0
        box_count = 0

        #Extract text from cropped images...
        for result in results:
            result_count += 1

            # render = render_result(model=self.yolo_model, image=input_image, result=result)
            # render.show()

            for captured_box in result.boxes:
                box_count += 1
                box = captured_box.xyxy

                # Extract the coordinates of the top-left and bottom-right corners
                x1, y1, x2, y2 = map(int, box[0])
                cropped_image = original_image[y1:y2, x1:x2]

                # Save or display the cropped image
                cropped_image_filename = self.image_upload_folder + image_file_name + '_' + str(result_count) + '_' + str(box_count) + '.png'
                cv2.imwrite(cropped_image_filename, cropped_image)
                render_result(model=self.yolo_model, image=cropped_image, result=result).show()

                #Extract the key value pairs for cropped image...
                extractedKVPairs = self.kvExtractor.extract_key_value_pair(cropped_image_filename)
                self.extracted_key_value_pairs.append(extractedKVPairs)

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

    # def extract_key_value_pairs(self):
    #     key_value_pairs = []
    #     for json_data in self.extracted_json_list:
    #         data = json.loads(json_data)
    #         for page in data['pages']:
    #             for block in page['blocks']:
    #                 for line in block['lines']:
    #                     text = ' '.join(word['value'] for word in line['words'])
    #                     pairs = re.findall(r'(\w+):\s*(\w+)', text)
    #                     key_value_pairs.extend(pairs)
    #     return key_value_pairs

    def extract_key_value_pairs(self, text):
        lines = text.split('\n')
        kvp = {}
        key = None

        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip()
                value = parts[1].strip()
                if key and value:
                    kvp[key] = value
            elif key:
                kvp[key] += f' {line.strip()}'

        self.kvp = kvp
        return kvp

    def display_key_value_pairs(self):
        text = pytesseract.image_to_string(self.input_image)
        kvp = self.extract_key_value_pairs(text)
        return kvp

    # def extract_key_value_pairs(self):
    #     key_value_pairs = []
    #     for extracted_json in self.extracted_json_list:
    #         dict = json.loads(extracted_json)
    #         value_dict = dict['pages'][0]
    #         block_list = value_dict['blocks']
    #         for block_value in block_list:
    #             line_list = block_value['lines']
    #             for line in line_list:
    #                 text = ' '.join([word['value'] for word in line['words']])
    #                 pairs = self._find_key_value_pairs(text)
    #                 key_value_pairs.extend(pairs)
    #     return key_value_pairs
    #
    # def _find_key_value_pairs(self, text):
    #     pattern = re.compile(r'(\w+):\s*(\w+)')
    #     key_value_pairs = pattern.findall(text)
    #     return key_value_pairs









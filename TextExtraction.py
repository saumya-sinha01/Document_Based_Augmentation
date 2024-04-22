import os
import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from ultralyticsplus import YOLO, render_result


class TextExtraction:
    def __init__(self):
        self.yolo_model = YOLO('keremberke/yolov8m-table-extraction')
        #self.yolo_model = YOLO('yolov8n.pt')

        # set model parameters
        self.yolo_model.overrides['conf'] = 0.25  # NMS confidence threshold
        self.yolo_model.overrides['iou'] = 0.45  # NMS IoU threshold
        self.yolo_model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        self.yolo_model.overrides['max_det'] = 1000  # maximum number of detections per image

    def processImage(self, image):


        # perform inference
        results = self.yolo_model.predict(image)

        # observe results
        print('Boxes: ', results[0].boxes)
        render = render_result(model=self.yolo_model, image=image, result=results[0])
        render.show()

        # parsed_json_list = []
        # for result in results:
        #     json = self.extractText(result)
        #     print(f'Parse Json: {json}')
        #     parsed_json_list.append(json)


    def extractText(self, result):
        # Instantiate a pretrained model
        predictor = ocr_predictor(pretrained=True)

        # Perform OCR on the document
        result = predictor(result)

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



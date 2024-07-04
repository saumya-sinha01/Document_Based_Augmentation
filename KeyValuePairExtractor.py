import pandas as pd
import numpy as np
import pytesseract
import torch
from itertools import groupby
import transformers
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, LayoutLMv3FeatureExtractor
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont

class KeyValuePairExtractor:

    def load_tags(self):
        datasets = load_dataset("nielsr/funsd")
        labels = datasets['train'].features['ner_tags'].feature.names

        return labels

    def load_models(self):
        feature_extractor = LayoutLMv3FeatureExtractor("microsoft/layoutlmv3-base", apply_ocr=True)
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

        return feature_extractor, processor, model

    def load_image(self, path):
        image = Image.open(path).convert("RGB")
        return image

    def unnormalize_box(self, bbox, width, height):
        return [
            width * (bbox[0] / 1000),
            height * (bbox[1] / 1000),
            width * (bbox[2] / 1000),
            height * (bbox[3] / 1000),
        ]


    def iob_to_label(self, label):
        label = label[2:]
        if not label:
            return 'other'
        return label


    def process_image(self, image, feature_extractor, processor, model, labels):
        id2label = {v: k for v, k in enumerate(labels)}
        label2id = {k: v for v, k in enumerate(labels)}
        width, height = image.size

        print("Prior Testing.......................................")
        encods = feature_extractor(image, return_tensors="pt")
        print("Testing.......................................")
        encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
        offset_mapping = encoding.pop("offset_mapping")

        outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        token_boxes = encoding.bbox.squeeze().tolist()

        is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0
        true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
        true_boxes = [self.unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]

        words = encods.words
        n = len(true_predictions) - len(words[0])
        k = int(n / 2)
        true_predictions = true_predictions[k:-k]
        true_boxes = true_boxes[k:-k]

        l_words = []
        preds = []
        bboxes = []
        key_pairs = []

        for i in range(0, len(words[0])):
            json_dict = {}
            if true_predictions[i] not in ["O"]:
                if true_predictions[i] in ["B-HEADER", "I-HEADER"]:
                    json_dict["label"] = "TITLE"
                elif true_predictions[i] in ["B-QUESTION", "I-QUESTION"]:
                    json_dict["label"] = "KEY"
                else:
                    json_dict["label"] = "VALUE"
                json_dict["value"] = words[0][i]
                key_pairs.append(json_dict)
                bboxes.append(true_boxes[i])

        return key_pairs, bboxes


    def visualize_image(self, image, key_pairs, bboxes):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        label2color = {'KEY': 'blue', 'VALUE': 'green', 'TITLE': 'orange'}

        for kp, box in enumerate(zip(key_pairs, bboxes)):
            draw.rectangle(box, outline=label2color[kp['label']])
            draw.text((box[0] + 10, box[1] - 10), text=kp['label'], fill=label2color[predicted_label], font=font)

        return image

    def extract_key_value_pair(self, input_path_image):
        # loads the pretrained dataset also
        # loads the image and labels
        image = self.load_image(input_path_image)
        labels = self.load_tags()

        # loads the model
        feature_extractor, processor, model = self.load_models()

        # gets the bounding boxes, predictions, extracted words and image processed
        kp = self.process_image(image, feature_extractor, processor, model, labels)

        return kp


# kvExtractor = KeyValuePairExtractor()
# input_path_image = 'D:/DocumentGen/test_images/test_image_1.jpg'
# kp = kvExtractor.extract_key_value_pair(input_path_image)
# # print(kp)
# tuples_list = kp[0]
# for tuple in tuples_list:
#     label = tuple['label']
#     value = tuple['value']
#     print(f"Label: {label}, Value: {value}")
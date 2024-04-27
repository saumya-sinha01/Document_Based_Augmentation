import json
import math
import os

import self

from TextExtraction import TextExtraction
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

class Data_Augmentation:
    textExtraction = TextExtraction()
    def removeText(self, text_to_be_removed, extracted_json_list, cropped_image_files):

        for index in range(len(extracted_json_list)):
            cropped_image_file = cropped_image_files[index]
            original_image = Image.open(cropped_image_file)

            extracted_json = extracted_json_list[index]
            extracted_json_dict = json.loads(extracted_json)
            # Get OCR text coordinates
            text_coordinates = self.get_coordinates(extracted_json_dict)
            # Reconstruct the table and overlay the edited text
            self.reconstruct_table(text_coordinates, original_image, text_to_be_removed)
            # Show the image
            original_image.show()
            print("Displaying the modified image..")

    def overlay_text(self, image, text, position, font=None, fill="black"):
        self.draw = ImageDraw.Draw(image)
        if font is None:
            font = ImageFont.load_default()  # Default font
        self.draw.text(position, text, font=font, fill=fill)

    def convert_coordinates(self, geometry, page_dim):
        len_x = page_dim[1]
        len_y = page_dim[0]
        (x_min, y_min) = geometry[0]
        (x_max, y_max) = geometry[1]
        x_min = math.floor(x_min * len_x)
        x_max = math.ceil(x_max * len_x)
        y_min = math.floor(y_min * len_y)
        y_max = math.ceil(y_max * len_y)
        return [x_min, x_max, y_min, y_max]

    def get_coordinates(self, extracted_json_dict):
        page_dim = extracted_json_dict['pages'][0]["dimensions"]
        text_coordinates = []
        for obj1 in extracted_json_dict['pages'][0]["blocks"]:
            for obj2 in obj1["lines"]:
                for obj3 in obj2["words"]:
                    converted_coordinates = self.convert_coordinates(obj3["geometry"], page_dim)
                    print("{}: {}".format(converted_coordinates, obj3["value"]))
                    text_coordinates.append((obj3["value"], converted_coordinates))  #Store text and its coordinates

        return text_coordinates

    def reconstruct_table(self, text_coordinates_dict, original_image, text_to_be_removed):
        for text, coordinates in text_coordinates_dict:
            if str(text).lower() == str(text_to_be_removed).lower():
                # Extract bounding box coordinates
                left, right, top, bottom = coordinates
                # Calculate the width and height of the text box
                width = right - left
                height = bottom - top
                # Create a white rectangle to cover the existing text
                white_box = Image.new("RGB", (width, height), color="white")
                original_image.paste(white_box, (left, top))

# Load the original table image

import json
import math

from PIL import Image, ImageDraw, ImageFont

class Data_Augmentation:

    def __init__(self):
        self.image_upload_folder = 'static\\uploads\\'

    def removeTextFromFullImage(self, text_to_be_removed, extracted_json, original_image_file):
        original_image = Image.open(original_image_file)
        extracted_json_dict = json.loads(extracted_json)
        # Get OCR text coordinates
        text_coordinates = self.get_coordinates(extracted_json_dict)
        # Reconstruct the table and overlay the edited text
        self.reconstruct_table(text_coordinates, original_image, text_to_be_removed)
        # Show the image
        # original_image.show()
        # print("Displaying the modified image...")
        original_image_filename = self.getFileName(original_image_file)
        augmented_image_filepath = self.image_upload_folder + original_image_filename + '_deleted.png'
        #Create the deleted image.
        original_image.save(fp=augmented_image_filepath)
        return augmented_image_filepath

    #method to replace text based on searched text
    def replaceText(self, text_to_be_replaced, extracted_json_list, original_image_file, replacement_Text):
            original_image = Image.open(original_image_file)
            extracted_json_dict = json.loads(extracted_json_list)
            # Get OCR text coordinates
            text_coordinates = self.get_coordinates(extracted_json_dict)
            # Reconstruct the table and overlay the edited text
            self.reconstruct_table(text_coordinates, original_image, text_to_be_replaced)
            # Overlay the edited text (Saumya) on top of the white box with Times New Roman font and blue color
            font = ImageFont.truetype("times.ttf", size=14)  # Times New Roman font, adjust size as needed
            self.overlay_text(original_image, replacement_Text, (self.left, self.top), font=font, fill="#000000")  # Black color
            # Show the image
            #original_image.show()
            #print("Displaying the modified image..")
            original_image_filename = self.getFileName(original_image_file)
            augemented_image_path_replacement = self.image_upload_folder + original_image_filename + '_replace.png'
            original_image.save(fp=augemented_image_path_replacement)
            return augemented_image_path_replacement

    def replaceAllText(self, text_to_be_replaced, extracted_json_list, original_image_file, replacement_Text):
        original_image = Image.open(original_image_file)
        extracted_json_dict = json.loads(extracted_json_list)
        # Get OCR text coordinates
        text_coordinates = self.get_coordinates(extracted_json_dict)
        # Reconstruct the table and overlay the edited text
        coordinates_List = self.reconstruct_table_for_ReplaceAll(text_coordinates, original_image, text_to_be_replaced)
        # Overlay the edited text (Saumya) on top of the white box with Times New Roman font and blue color
        font = ImageFont.truetype("times.ttf", size=16)  # Times New Roman font, adjust size as needed
        # self.overlay_text(original_image, replacement_Text, (self.left, self.top), font=font,fill="#000000")  # Black color
        self.overlay_text_MutlpleText(original_image, replacement_Text, coordinates_List, font=font, fill="#000000")
        # Show the image
        # original_image.show()
        # print("Displaying the modified image..")
        original_image_filename = self.getFileName(original_image_file)
        augemented_image_path_replacement = self.image_upload_folder + original_image_filename + '_replace.png'
        original_image.save(fp=augemented_image_path_replacement)
        return augemented_image_path_replacement

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

    def overlay_text_MutlpleText(self, image, text, coordinates_list, font=None, fill="black"):
        draw = ImageDraw.Draw(image)
        if font is None:
            font = ImageFont.load_default()  # Default font
        for coord in coordinates_list:
            position = coord["position"]
            draw.text(position, text, font=font, fill=fill)

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
                self.left, self.right, self.top, self.bottom = coordinates
                # Calculate the width and height of the text box
                width = self.right - self.left
                height = self.bottom - self.top
                # Create a white rectangle to cover the existing text
                white_box = Image.new("RGB", (width, height), color="white")
                original_image.paste(white_box, (self.left, self.top))

    def reconstruct_table_for_ReplaceAll(self, text_coordinates_dict, original_image, text_to_be_removed):
        # List to store coordinates of the text to be removed and replacement text
        coordinates_list = []
        for text, coordinates in text_coordinates_dict:
            if str(text).lower() == str(text_to_be_removed).lower():
                # Extract bounding box coordinates
                self.left, self.right, self.top, self.bottom = coordinates
                # Calculate the width and height of the text box
                width = self.right - self.left
                height = self.bottom - self.top
                # Create a white rectangle to cover the existing text
                white_box = Image.new("RGB", (width, height), color="white")
                original_image.paste(white_box, (self.left, self.top))
                coordinates_list.append({
                    "original_text": text,
                    "original_coordinates": coordinates,
                    "position": (self.left, self.top)
                })
                #self.bottom,self.right
        print("Coordinates of replaced texts:", coordinates_list)
        return coordinates_list
    def getFileName(self, input_image):
        return input_image.replace(self.image_upload_folder, '').replace('.png', '')


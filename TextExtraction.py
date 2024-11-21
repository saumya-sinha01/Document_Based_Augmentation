import os, cv2, json, re
import time
import math
from pathlib import Path
from collections import defaultdict
import pytesseract
from doctr.models import ocr_predictor
from ultralyticsplus import YOLO
from bisect import bisect_left
from PIL import Image, ImageDraw, ImageFont

from KeyValuePairExtractor import KeyValuePairExtractor

class TextExtraction:

    input_image = None
    orig_width = None
    orig_height = None

    def __init__(self):
        self.image_upload_folder  = 'static\\uploads\\'
        self.image_search_folder  = 'static\\search\\'
        self.image_replace_folder = 'static\\replace\\'
        os.environ['USE_TORCH'] = '1'

        self.extracted_json_list = []
        self.extracted_key_value_pairs = []
        self.labeled_images = []
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

    '''
    This function parses the uploaded image and extracts text from it. It saves the extracted key-value pairs into a dictionary: self.dict
    '''
    def processImageV2(self, input_image):
        self.input_image = input_image
        self.input_image_extension = os.path.splitext(input_image)[1].lower()

        #Reset State:
        self.extracted_json_list = []
        self.extracted_key_value_pairs = []
        self.labeled_images = []
        self.cropped_image_files = []
        self.full_image_json = None

        #Load the image:
        original_image = cv2.imread(input_image)

        #Extract the filename..
        image_file_name = input_image.replace(self.image_upload_folder, '').replace('.png', '')
        image_file_name = image_file_name.replace(self.image_upload_folder, '').replace('.jpg', '')
        image_file_name = image_file_name.replace(self.image_upload_folder, '').replace('.jpeg', '')

        #Save the image file name too for later use.
        self.image_file_name = image_file_name

        #Get the original width and height.
        orig_width, orig_height, channels = original_image.shape
        self.orig_width = orig_width
        self.orig_height = orig_height

        # Extract the key value pairs for cropped image...
        extractedKVPairs, bboxes, labeled_image = self.kvExtractor.extract_key_value_pair_from_image_path(input_image)
        self.extracted_key_value_pairs.append(extractedKVPairs)

        labeled_image_filename = self.image_upload_folder + image_file_name + '_labeled.png'

        # Resize the image.
        resized_image = labeled_image.resize((orig_height, orig_width))
        resized_image.save(labeled_image_filename)
        self.labeled_images.append(labeled_image_filename)

        #Update the dictionary...
        self.parse_extracted_json(extractedKVPairs, bboxes)

    '''
    This function searches for the input text into the dictionary and if present creates a new image with the search text highlighted within boundaries.
    '''
    def search(self, searchText):
        if not searchText:
            return False

        lowercaseSearchText = searchText.lower()
        searched_image = Image.open(self.input_image).convert("RGB").copy()
        if lowercaseSearchText in self.dict:
            #Remove all the old search files...
            self.delete_old_images(self.image_search_folder)
            #Perform search...
            coordinatesList = self.dict[lowercaseSearchText]
            for coordinates in coordinatesList:
                searched_image = self.highlight_text_within_image_for_single_coordinate(searched_image, coordinates)

            filename = self.save_searched_image(searched_image)
            return True, filename
        return False, None

    '''
    This function searches for the input text into the dictionary and if present replaces with the replacement_text. 
    The argument 'replace_all decides' whether all instances should be replaced or just one. 
    '''
    def replace(self, input_text, replacement_text, replace_all):
        if not input_text:
            return False

        #Get the position of replacement_text in the self.dictionary for later use in highlighting.
        replacement_text_coordinate_list = []
        if replacement_text.lower() in self.dict:
            replacement_text_coordinate_list = self.dict[replacement_text.lower()]

        lowercaseInputText = input_text.lower()
        if lowercaseInputText in self.dict:
            self.delete_old_images(self.image_replace_folder)
            # Get the co-ordinates list of the test found so far.
            coordinatesList = self.dict[lowercaseInputText]
            if replace_all:
                original_image_copy = Image.open(self.input_image).convert("RGB").copy()
                augmented_image = original_image_copy.copy()
                for text_coordinate in coordinatesList:
                    # Highlight input_text for every text co-ordinate in the original image.
                    original_image_copy = self.highlight_text_within_image_for_single_coordinate(original_image_copy, text_coordinate)
                    # Perform replace operation for every text co-ordinate. Returns a new image
                    augmented_image = self.replaceText(replacement_text, text_coordinate, augmented_image)

                #Once updated, perform the following steps:
                # Update the dictionary and the input image.
                self.update_image(augmented_image)

                # Highlight the replaced text in augmented image.
                augmented_image_visualized = self.highlight_text_within_image(augmented_image, replacement_text, True)

                # Save the augmented image and original highlighted image to be returned.
                orig_img_visualized_filename = self.save_replaced_text_image(original_image_copy)
                augmented_img_visualized_filename = self.save_replaced_text_image(augmented_image_visualized)
                return True, orig_img_visualized_filename, augmented_img_visualized_filename
            else:
                # Pick the first co-ordinate from the list.
                first_coordinate = coordinatesList[0]

                #Find the position which will be helpful for insertion.
                coordinate_position = self.find_insert_position(replacement_text_coordinate_list, first_coordinate)

                #Create a copy of the original image to be used for replace operation.
                original_image_copy = Image.open(self.input_image).convert("RGB").copy()
                orig_img_visualized = self.highlight_text_within_image_for_single_coordinate(original_image_copy, first_coordinate)

                #Perform replace operation. Returns a new image.
                augmented_image = Image.open(self.input_image).convert("RGB").copy()
                augmented_image = self.replaceText(replacement_text, first_coordinate, augmented_image)

                # Update the dictionary and the input image.
                self.update_image(augmented_image)

                #Highlight the input and replaced text.
                augmented_image_visualized = self.highlight_text_within_image(augmented_image, replacement_text, False, coordinate_position)

                #Save the augmented image and original highlighted image to be returned.
                orig_img_visualized_filename = self.save_replaced_text_image(orig_img_visualized)
                augmented_img_visualized_filename = self.save_replaced_text_image(augmented_image_visualized)

                return True, orig_img_visualized_filename, augmented_img_visualized_filename

        return False, None, None

    def deleteText(self, input_text, delete_all=False):
        if not input_text:
            return False

        input_text = input_text.lower()
        if input_text in self.dict:

            if delete_all:
                #Perform some logic here.
                coordinatesList = self.dict[input_text]
                original_image_copy = Image.open(self.input_image).convert("RGB").copy()
                augmented_image = Image.open(self.input_image).convert("RGB").copy()

                for text_coordinate in coordinatesList:
                    # Highlight input_text for every text co-ordinate in the original image.
                    original_image_copy = self.highlight_text_within_image_for_single_coordinate(original_image_copy, text_coordinate)
                    # Perform replace operation for every text co-ordinate. Returns a new image
                    augmented_image = self.reconstruct_table(text_coordinate, augmented_image)

                # Once updated, perform the following steps:
                # Update the dictionary and the input image.
                self.update_image(augmented_image)

                # Save the augmented image and original highlighted image to be returned.
                orig_img_visualized_filename = self.save_replaced_text_image(original_image_copy)
                augmented_img_visualized_filename = self.save_replaced_text_image(augmented_image)
                return True, orig_img_visualized_filename, augmented_img_visualized_filename

            else:
                # Pick the first co-ordinate from the list.
                coordinatesList = self.dict[input_text]
                first_coordinate = coordinatesList[0]

                # Create a copy of the original image to be used for replace operation.
                original_image_copy = Image.open(self.input_image).convert("RGB").copy()
                orig_img_visualized = self.highlight_text_within_image_for_single_coordinate(original_image_copy, first_coordinate)

                # Perform replace operation. Returns a new image.
                augmented_image = Image.open(self.input_image).convert("RGB").copy()
                augmented_image = self.reconstruct_table(first_coordinate, augmented_image)

                # Update the dictionary and the input image.
                self.update_image(augmented_image)

                # Save the augmented image and original highlighted image to be returned.
                orig_img_visualized_filename = self.save_replaced_text_image(orig_img_visualized)
                augmented_img_visualized_filename = self.save_replaced_text_image(augmented_image)

                return True, orig_img_visualized_filename, augmented_img_visualized_filename

        return False, None, None

    #Function to replace text based on searched text. Returns a new image.
    def replaceText(self, replacement_text, text_coordinates, input_image):
            # Reconstruct the table and overlay the edited text
            left, bottom, right, top = text_coordinates
            input_image_copy = self.reconstruct_table(text_coordinates, input_image)
            # Overlay the edited text on top of the white box with Times New Roman font and blue color
            font = ImageFont.truetype("times.ttf", size=15)  # Times New Roman font, adjust size as needed
            self.overlay_text(input_image_copy, replacement_text, (left - 0.5, bottom - 0.5), font=font, fill="#000000")  # Black color
            return input_image_copy

    def overlay_text(self, image, text, position, font=None, fill="black"):
        self.draw = ImageDraw.Draw(image)
        if font is None:
            font = ImageFont.load_default()  # Default font
        self.draw.text(position, text, font=font, fill=fill)

    def reconstruct_table(self, text_coordinates, input_image):
        # Extract bounding box coordinates
        left, bottom, right, top = text_coordinates
        # Calculate the width and height of the text box
        width = math.ceil(right - left) + 1
        height = math.ceil(top - bottom) + 1
        # Create a white rectangle to cover the existing text
        white_box = Image.new("RGB", (width, height), color="white")

        # original_image_copy = Image.open(self.input_image).convert("RGB").copy()
        # original_image_copy.paste(white_box, (left, bottom))

        input_image_copy = input_image.convert("RGB").copy()
        input_image_copy.paste(white_box, (math.floor(left), math.floor(bottom)))

        return input_image_copy

    def parse_extracted_json(self, extractedKVPairs, bboxes):
        self.dict = defaultdict()
        for idx in range(0, len(extractedKVPairs)):
            tuple = extractedKVPairs[idx]
            word_value = tuple['value']
            word_value = str(word_value).lower()

            coord = bboxes[idx]
            if word_value in self.dict:
                self.dict[word_value].append(coord)
            else:
                self.dict[word_value] = list()
                self.dict[word_value].append(coord)

    def highlight_text_within_image(self, image, text, highlight_all = False, coordinate_position = 0):
        highlighted_image = image
        if text.lower() in self.dict:
            coordinatesList = self.dict[text.lower()]
            if highlight_all:
                for coordinates in coordinatesList:
                    highlighted_image = self.highlight_text_within_image_for_single_coordinate(highlighted_image, coordinates)
            else:
                if len(coordinatesList) > coordinate_position:
                    highlighted_image = self.highlight_text_within_image_for_single_coordinate(highlighted_image, coordinatesList[coordinate_position])
                else:
                    highlighted_image = self.highlight_text_within_image_for_single_coordinate(highlighted_image, coordinatesList[0])

        return highlighted_image

    def highlight_text_within_image_for_single_coordinate(self, image, coordinates):
        try:
            draw = ImageDraw.Draw(image)

            # Ensure coordinates are valid
            if len(coordinates) != 4:
                raise ValueError("Coordinates must contain exactly 4 values (x1, y1, x2, y2).")

            coord = [(coordinates[0], coordinates[1]), (coordinates[2], coordinates[3])]

            # Drawing rectangle and text
            draw.rectangle(coord, outline='green')

            return image

        except ValueError as ve:
            print(f"ValueError: {ve}")
            return None

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def save_searched_image(self, searched_image):
        timestamp = time.time()
        image_file_name = "search_" + str(timestamp)
        labeled_image_filename = self.image_search_folder + image_file_name + '.png'
        resized_searched_image = searched_image.resize((self.orig_height, self.orig_width))
        resized_searched_image.save(labeled_image_filename)
        return labeled_image_filename

    def save_replaced_text_image(self, replaced_image):
        timestamp = time.time()
        image_file_name = "replaced_" + str(timestamp)
        labeled_image_filename = self.image_replace_folder + image_file_name + '.png'
        resized_replaced_image = replaced_image.resize((self.orig_height, self.orig_width))
        resized_replaced_image.save(labeled_image_filename)
        return labeled_image_filename

    def delete_old_images(self, folder_name):
        # Iterate and delete all files in the directory
        directory = Path(folder_name)

        for file_path in directory.glob('*'):
            if file_path.is_file():
                file_path.unlink()
                print(f"Deleted: {file_path}")

    def update_image(self, updated_image):
        #Delete the old images.
        self.delete_old_images(self.image_upload_folder)

        # Create a copy of the updated image for saving.
        updated_image_copy = updated_image.convert("RGB").copy()

        # Overwrite the original image with updated image...
        original_image_filename = self.image_upload_folder + self.image_file_name + self.input_image_extension
        updated_image_copy.save(original_image_filename)

        # Extract the key value pairs for the updated image...
        extractedKVPairs, bboxes, labeled_image = self.kvExtractor.extract_key_value_pair_from_image_path(original_image_filename)
        self.extracted_key_value_pairs = []
        self.extracted_key_value_pairs.append(extractedKVPairs)

        # Resize and overwrite the labeled image.
        labeled_image_filename =  self.image_upload_folder + self.image_file_name + '_labeled.png'
        updated_image_copy = labeled_image.resize((self.orig_height, self.orig_width))
        updated_image_copy.save(labeled_image_filename)

        #Update the labeled image to be used in key value pairs functionality.
        self.labeled_images = []
        self.labeled_images.append(labeled_image_filename)

        # Re-extract the json...
        self.parse_extracted_json(extractedKVPairs, bboxes)

    def find_insert_position(self, coords, new_coord):
        # Extract only 'left' values for binary search
        left_values = [coord[0] for coord in coords]
        # Find the insertion index
        index = bisect_left(left_values, new_coord[0])
        return index

#FileNotFoundError: [Errno 2] No such file or directory:
#'D:\\PycharmProjects\\Document_Based_Augmentation\\Document_Based_Augmentation\\static\\uploads\\test_image_4.jpg'
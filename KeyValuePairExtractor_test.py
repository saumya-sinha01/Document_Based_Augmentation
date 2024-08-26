from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image, ImageDraw
import torch

# Load the pre-trained LayoutLMv3 model and processor
model_name = "microsoft/layoutlmv3-base"
processor = LayoutLMv3Processor.from_pretrained(model_name)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

# Load and preprocess the document image
image_path = "D:\\PycharmProjects\\Document_Based_Augmentation\\Document_Based_Augmentation\\static\\uploads\\test_image_1.jpg"  # Replace with your image path
image = Image.open(image_path)

# Preprocess the image for the model
inputs = processor(images=image, return_tensors="pt", padding="max_length", truncation=True)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Decode the model's output to extract key-value pairs
token_predictions = outputs.logits.argmax(-1).squeeze().tolist()
tokens = inputs["input_ids"].squeeze().tolist()

# Get the words and bounding boxes from the processor
words, boxes = processor.tokenizer.convert_ids_to_tokens(tokens), inputs.boxes.squeeze().tolist()

key_value_pairs = []
current_key = ""
current_value = ""

for word, box, label in zip(words, boxes, token_predictions):
    label = model.config.id2label[label]

    if label == "B-KEY":
        if current_key:
            key_value_pairs.append((current_key.strip(), current_value.strip()))
        current_key = word
        current_value = ""
    elif label == "I-KEY":
        current_key += f" {word}"
    elif label == "B-VALUE" or label == "I-VALUE":
        current_value += f" {word}"

if current_key:
    key_value_pairs.append((current_key.strip(), current_value.strip()))

# Draw the key-value pairs on the image using ImageDraw
draw = ImageDraw.Draw(image)

for key, value in key_value_pairs:
    # Simple logic to find the corresponding bounding boxes for key and value
    key_index = tokens.index(processor.tokenizer.encode(key)[1])
    value_index = tokens.index(processor.tokenizer.encode(value)[1])

    key_box = boxes[key_index]
    value_box = boxes[value_index]

    # Draw rectangles around the key and value
    draw.rectangle(key_box, outline="red", width=2)
    draw.rectangle(value_box, outline="blue", width=2)

    # Draw text for key and value
    draw.text((key_box[0], key_box[1]), key, fill="red")
    draw.text((value_box[0], value_box[1]), value, fill="blue")

# Save or display the annotated image
# image.save("D:\\PycharmProjects\\Document_Based_Augmentation\\Document_Based_Augmentation\\static\\uploads\\test_image_1.jpg")
image.show()

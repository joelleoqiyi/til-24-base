print('updated9')
import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import io

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# NEW: get example.jpg as bytes
# Define the path to the image
image_path = 'example.jpg'
# Open the file in binary mode ('rb')
with open(image_path, 'rb') as file:
    image_bytes = file.read()  # Read the entire file as a bytes object
image_stream = io.BytesIO(image_bytes)
image = Image.open(image_stream).convert('RGB')

print('s ', image.size)

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# Check for cats and remote controls
# VERY important: text queries need to be lowercased + end with a dot
text = "a cat. a remote control."
text = "a yellow helicopter."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

print('OK')
print(results)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
def save_image(result):
    # Create a figure and axis to plot on
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    # Plot each bounding box
    for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
        # Extract coordinates
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        # Add label and score
        plt.text(x_min, y_min, f'{label} ({score:.2f})', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    # Show the image
    plt.axis('off')  # Hide axes
    fig.canvas.draw()  # Draw the canvas
    image_with_boxes = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    image_with_boxes.save('output_image2.jpg')
save_image(results[0])

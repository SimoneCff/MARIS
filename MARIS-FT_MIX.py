import os
import numpy as np
import PIL
import argparse
import re
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import big_vision.big_vision.evaluators.proj.paligemma.transfers.segmentation as segeval

"""                  
MARIS Main Script:
    Corals Automated Recognition and Identification System
    ------------------------------------------------------
    Model used:
        "google/paligemma-3b-mix" 
        "CNN - 4 Base Layer & 2 Fully Connected Layers"
    ------------------------------------------------------
"""

reconstruct_masks = segeval.get_reconstruct_masks('oi')
dtype = torch.bfloat16
local_model_path = "./model"
target_size = (448, 448)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device Used : {device}")
model = PaliGemmaForConditionalGeneration.from_pretrained(local_model_path, torch_dtype=dtype, device_map=device, revision="bfloat16").eval()
processor = AutoProcessor.from_pretrained(local_model_path)

def request(image, text_input):
    try:  
        model_inputs = processor(text=text_input, images=image, return_tensors="pt").to(model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode(): 
            generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)
            return decoded
    except Exception as e:
        print(f"Error in processing the request: {e}")

def parse_segments(detokenized_output: str) -> tuple[np.ndarray, np.ndarray]:
    matches = re.finditer(
        '<loc(?P<y0>\d\d\d\d)><loc(?P<x0>\d\d\d\d)><loc(?P<y1>\d\d\d\d)><loc(?P<x1>\d\d\d\d)>'
        + ''.join(f'<seg(?P<s{i}>\d\d\d)>' for i in range(16)),
        detokenized_output,
    )
    boxes, segs = [], []
    fmt_box = lambda x: float(x) / 1024.0
    for m in matches:
        d = m.groupdict()
        boxes.append([fmt_box(d['y0']), fmt_box(d['x0']), fmt_box(d['y1']), fmt_box(d['x1'])])
        segs.append([int(d[f's{i}']) for i in range(16)])
    return np.array(boxes), np.array(reconstruct_masks(np.array(segs)))

def crop_and_resize(image, target_size):
    width, height = image.size
    source_size = min(image.size)
    left = width // 2 - source_size // 2
    top = height // 2 - source_size // 2
    right, bottom = left + source_size, top + source_size
    return image.resize(target_size, box=(left, top, right, bottom))

def read_image(image_path, target_size):
    image = PIL.Image.open(image_path)
    image = crop_and_resize(image, target_size)
    image = np.array(image)
    # Remove alpha channel if necessary.
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image

def parse_bbox_and_labels(detokenized_output: str):
    matches = re.finditer(
        '<loc(?P<y0>\d\d\d\d)><loc(?P<x0>\d\d\d\d)><loc(?P<y1>\d\d\d\d)><loc(?P<x1>\d\d\d\d)>'
        ' (?P<label>.+?)( ;|$)',
        detokenized_output,
    )
    labels, boxes = [], []
    fmt = lambda x: float(x) / 1024.0
    for m in matches:
        d = m.groupdict()
        boxes.append([fmt(d['y0']), fmt(d['x0']), fmt(d['y1']), fmt(d['x1'])])
        labels.append(d['label'])
    return np.array(boxes), np.array(labels)

def display_boxes(ax, image, boxes, labels, target_image_size):
    h, l = target_image_size
    ax.imshow(image)
    for i in range(boxes.shape[0]):
        y, x, y2, x2 = (boxes[i]*h)
        width = x2 - x
        height = y2 - y
        # Create a Rectangle patch
        rect = patches.Rectangle((x, y),
                                 width,
                                 height,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        # Add label
        ax.text(x, y, labels[i], color='red', fontsize=12)
        # Add the patch to the Axes
        ax.add_patch(rect)

def display_segment_output(ax, image, segment_mask, target_image_size):
    # Calculate scaling factors
    h, w = target_image_size
    x_scale = w / 64
    y_scale = h / 64

    # Create coordinate grids for the new image
    x_coords = np.arange(w)
    y_coords = np.arange(h)
    x_coords = (x_coords / x_scale).astype(int)
    y_coords = (y_coords / y_scale).astype(int)
    resized_array = segment_mask[y_coords[:, np.newaxis], x_coords]
    # Display the image
    ax.imshow(image)

    # Overlay the mask with transparency
    ax.imshow(resized_array, cmap='jet', alpha=0.5)

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--object', type=str)
args = parser.parse_args()

image = read_image(args.image, target_size)
name = os.path.splitext(os.path.basename(args.image))[0]

# Create a figure with 2 subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
fig.canvas.manager.set_window_title("MARIS Result")

if args.object == "" or args.object is None:
    prompt = 'answer en can you Write down all the object that you detect into the image?'
    output = request(image=image,text_input=prompt)
    output = output.replace(prompt,"")
    object_DT = output.strip()
else:
    object_DT = args.object
prompt = f'detect {object_DT}\n'
output = request(image=image,text_input=prompt)
boxes, labels = parse_bbox_and_labels(output)

prompt = f'segment {object_DT}\n'
output = request(image=image,text_input=prompt)
try:
    _, seg_output = parse_segments(output)
    display_segment_output(axs[1], image, seg_output[0], target_size)
    axs[1].set_title("Segmentation")
except:
    print("SEGMENTATION ERROR")
    axs[1].set_title("Segmentation")

# Display detection results
display_boxes(axs[0], image, boxes, labels, target_size)
axs[0].set_title("Detection")

if not os.path.exists('./Out'):
        os.makedirs('./Out')

# Save the combined figure
plt.savefig(f'Out/detection_and_segmentation_{name}.png')
plt.show()
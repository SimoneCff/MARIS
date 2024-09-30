import numpy as np
import PIL
import argparse
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch

"""                  
MARIS Main Script:
    Marine Automated Recognition and Identification System
    ------------------------------------------------------
    Model used:
        "google/paligemma-3b-mix" 
        "CNN - 4 Base Layer & 2 Fully Connected Layers"
    ------------------------------------------------------
"""

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

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
args = parser.parse_args()
image = read_image(args.image, target_size)

while True:
    prompt = input("Chiedi quello che vuoi (premi 'q' per uscire): ")
    
    if prompt.lower() == 'q':
        break
    output = request(image=image, text_input=prompt)
    print(f"<MARIS> : {output}")
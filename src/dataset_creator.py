import cv2
import urllib
import numpy as np
import json
import requests
import io
from PIL import Image
import json
from unidecode import unidecode
from tqdm.auto import tqdm

index = 0

with open("export.json") as f:  
    data = json.load(f)

all_data = []
url_data = []
class_labels = {}

for idx, item in enumerate(tqdm(data)):    
    response = requests.get(item['asset'])
    if response.status_code == 200:
        image_bytes = io.BytesIO(response.content)
        img = Image.open(image_bytes)
        for task in item['tasks']:
            for obj in task['objects']:
                meta_data = {"file_name": "", "label": ""}
                url_bbox_data = {"file_name": "", "url": "", "bbox" : []}
                box = [obj['bounding-box']['x'], obj['bounding-box']['y'], obj['bounding-box']['width'], obj['bounding-box']['height']]
                left = box[0]
                right = box[0]+box[2]
                top = box[1] 
                bottom = box[1]+box[3]
                cropped_img = img.crop((min(left, right), min(top, bottom), max(left, right), max(top, bottom)))
                cropped_img.save(f"images/test/{index:05d}.png")
                meta_data['file_name'] = f"{index:05d}.png" 
                url_bbox_data['file_name'] = f"{index:05d}.png" 
                
                # Add labels to the dictionary
                if unidecode(obj["title"]) not in class_labels:
                    new_index = len(class_labels)
                    class_labels[unidecode(obj["title"])] = new_index

                meta_data['label'] = class_labels[unidecode(obj["title"])]
                url_bbox_data['url'] = item['asset']
                url_bbox_data['bbox'] = box
                index += 1
                all_data.append(meta_data)
                url_data.append(url_bbox_data)
                
with open("images/test/metadata.jsonl", "a") as f:
    for data in all_data:
        f.write(json.dumps(data) + "\n")


with open("data/url_bbox_meta.jsonl", "a") as f:
    for data in url_data:
        f.write(json.dumps(data)+ "\n")

print(class_labels)


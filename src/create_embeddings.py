from model import extract_embeddings, model, transformation_chain
from datasets import load_dataset
import torch
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import re 
import json

dataset = load_dataset("imagefolder", data_dir="images")
# {'Bebek giyim': 0, 'Ic Giyim': 1, 'Tek parca': 2, 'Ust Giyim': 3, 'Alt Giyim': 4, 'Aksesuar giyim': 5, 'Aksesuar': 6, 'Ayakkabi': 7, 'Dis giyim': 8}

time_started = time.time()
extract_fn = extract_embeddings(model.to("mps"))
dataset_emb = dataset["test"].map(extract_fn, batched=True, batch_size=41727)
print("Embedding extraction done in " + str(time.time() - time_started) + " sec.")

candidate_ids = []

input_jsonl = "data/url_bbox_meta.jsonl"
with open(input_jsonl, "r") as f:
    # Read JSONL entries
    entries = [json.loads(line) for line in f]

file_name_url_bbox_pairs = [{"file_name": entry["file_name"], "url": entry["url"], "bbox": entry["bbox"]} for entry in entries]

for id in tqdm(range(len(dataset_emb))):
    label = dataset_emb[id]["label"]
    url = file_name_url_bbox_pairs[id]["url"]
    bbox = file_name_url_bbox_pairs[id]["bbox"]

    # Create a unique indentifier.
    entry = str(id) + "_" + str(label) + "_" +  str(url) + "_" + str(bbox)

    candidate_ids.append(entry)


all_candidate_embeddings = np.array(dataset_emb["embeddings"])
all_candidate_embeddings = torch.from_numpy(all_candidate_embeddings)


with open("embeddings.pickle", "wb") as handle:
    pickle.dump(all_candidate_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("candidate_ids.pickle", "wb") as handle:
    pickle.dump(candidate_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

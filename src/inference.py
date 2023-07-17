import pickle
from src.plot import plot_images, create_table
from src.model import model, transformation_chain, device
from datasets import load_dataset
import numpy as np
from src.similarity_check import fetch_similar
import time
import json
import re
from PIL import Image

def batch_inferce():
    with open("embeddings.pickle", "rb") as handle:
        all_candidate_embeddings = pickle.load(handle)

    with open("candidate_ids.pickle", "rb") as handle:
        candidate_ids = pickle.load(handle)

    dataset = load_dataset("imagefolder", data_dir="images")

    for i in range(10):
        test_idx = 36723
        test_sample = dataset["test"][test_idx]["image"]
        test_label = dataset["test"][test_idx]["label"]

        time_started = time.time()
        sim_ids, sim_labels = fetch_similar(test_sample, transformation_chain, device, model, all_candidate_embeddings, candidate_ids)
        print(f"Top 10 candidate labels: {sim_labels}")
        print("Similar image found done in " + str(time.time() - time_started) + " sec.")

        images = []
        labels = []

        for id, label in zip(sim_ids, sim_labels):
            images.append(dataset['test'][id]["image"])
            labels.append(dataset['test'][id]["label"])

        images.insert(0, test_sample)
        labels.insert(0, test_label)
        plot_images(images, labels, i)

def single_inferece(image_path, all_candidate_embeddings, candidate_ids):
    image = Image.open(image_path).convert('RGB')
    time_started = time.time()
    sim_ids, sim_labels, sim_urls, sim_boxes = fetch_similar(image, transformation_chain, device, model, all_candidate_embeddings, candidate_ids)
    print(f"Finding similar images by emdedding comparison done in " + format(time.time() - time_started) + " sec.", )
    image_path = create_table(image, sim_urls, sim_boxes)
    return image_path

# batch_inferce()

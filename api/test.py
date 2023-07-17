from  src.inference import single_inferece
import pickle
from PIL import Image

with open("embeddings.pickle", "rb") as handle:
    all_candidate_embeddings = pickle.load(handle)

with open("candidate_ids.pickle", "rb") as handle:
    candidate_ids = pickle.load(handle)

image_path = ""
single_inferece(image_path, all_candidate_embeddings, candidate_ids)

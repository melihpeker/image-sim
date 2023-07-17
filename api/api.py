from src.inference import single_inferece

import os
from flask import Flask, jsonify, request, send_from_directory, abort, send_file
from werkzeug.utils import secure_filename
import pickle
import tempfile
from flask_cors import CORS


app = Flask(__name__)
app.config['IMAGE_PATH'] = "/Users/melihpeker/Desktop/image-sim/frontend"
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})

@app.before_first_request
def load_embeddings():
    with open("/Users/melihpeker/Desktop/image-sim/data/embeddings.pickle", "rb") as handle:
        all_candidate_embeddings = pickle.load(handle)
        app.config['all_candidate_embeddings'] = all_candidate_embeddings
    
    with open("/Users/melihpeker/Desktop/image-sim/data/candidate_ids.pickle", "rb") as handle:
        candidate_ids = pickle.load(handle)
        app.config['candidate_ids'] = candidate_ids


@app.route("/", methods=["GET"])
def index():
    return "Image API"

@app.route("/find-similar", methods=["POST"])
def find_similar():
    """
        Converts DWG file to DXF file
    """
    if len(request.files) != 0:
        if 'image' in request.files:
            f = request.files['image']
            filename = secure_filename(f.filename)
            if filename != '':
                file_ext = os.path.splitext(filename)[1]
                if not file_ext == ".png":
                    abort(400)
            filename = "tmp.png"
            f.save(os.path.join(app.config['IMAGE_PATH'], filename))
        image_path = single_inferece(os.path.join(app.config['IMAGE_PATH'], filename), app.config['all_candidate_embeddings'], app.config['candidate_ids'])
        return send_from_directory(app.config['IMAGE_PATH'],  "result.png")




if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8088)


import matplotlib.pyplot as plt
import numpy as np
import urllib
import numpy as np
import requests
import io
from matplotlib.table import Table
import math
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import time


def plot_images(images, labels, index):
    if not isinstance(labels, list):
        labels = labels.tolist()

    plt.figure(figsize=(20, 10))
    columns = 10
    for (i, image) in enumerate(images):
        label_id = int(labels[i])
        ax = plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        if i == 0:
            ax.set_title("Query Image\n" + "Label: {}".format(label_id))
        else:
            ax.set_title(
                "Similar Image # " + str(i) + "\nLabel: {}".format(label_id)
            )
        plt.imshow(np.array(image).astype("int"))
        plt.axis("off")
        plt.savefig("similars_" + str(index) + ".png")


def create_table(original_image, urls, boxes):
    # Create a list of 10 images (replace with your image paths or data)
    similar_images = []
    max_tries = 5
    time_started = time.time()
    for url, box in zip(urls, boxes):
        # print(url)
        timeout = 0
        for i in range(max_tries):
            try: 
                response = requests.get(url, timeout=3)
                break
            except:
                timeout += 1
        if timeout == 3:
            print("image skipped")
            continue

        web_image = Image.open(BytesIO(response.content))
        # web_image = Image.open(urllib.request.urlopen(url))
        # web_image.save("img.png")
        left = box[0]
        right = box[0]+box[2]
        top = box[1] 
        bottom = box[1]+box[3]
        cropped_img = web_image.crop((min(left, right), min(top, bottom), max(left, right), max(top, bottom)))
        similar_images.append(cropped_img)

    print("Downloading and cropping images done in " +  format(time.time() - time_started) + " sec.",)
   # Calculate the size of the grid and spacing
    num_images = len(similar_images)
    num_cols = 10
    num_rows = 1
    spacing = 100  # Adjust the spacing between images (in pixels)

    # Create a blank canvas for the grid
    canvas_width = sum(image.width for image in similar_images) + spacing * (num_cols - 1) + 200
    canvas_height = max(image.height for image in similar_images) + 512 + original_image.height + spacing * (num_rows - 1) 
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    # Prepare font for text
    font_size =64  # Adjust the font size
    font = ImageFont.truetype("Arial", font_size)

    # Paste the images and add text on top
    draw = ImageDraw.Draw(canvas)
    x, y = int(canvas_width/2) - int(original_image.width/2), 100
    canvas.paste(original_image, (x, y))
    text = "Original Image"
    text_width, text_height = draw.textsize(text, font=font)
    text_x = x + (original_image.width - text_width) // 2
    text_y = y  - text_height
    draw.text((text_x, text_y), text, fill="black", font=font)

    x, y = 100, 512+original_image.height
    for i, image in enumerate(similar_images):
        canvas.paste(image, (x, y))
        text = f"Image {i+1}"
        text_width, text_height = draw.textsize(text, font=font)
        text_x = x + (image.width - text_width) // 2
        text_y = y - text_height 
        draw.text((text_x, text_y), text, fill="black", font=font)
        x += image.width + spacing
        if x >= canvas_width:
            x = 0
            y += image.height + spacing
    # Save the final grid image
    name = "/Users/melihpeker/Desktop/image-sim/frontend/result.png"
    canvas.save(name)
    return name

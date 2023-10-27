import io
from operator import truediv
import os
from PIL import Image
import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect

app = Flask(__name__)

RESULT_FOLDER = os.path.join("static")
app.config["RESULT_FOLDER"] = RESULT_FOLDER

model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'best.pt')
model.eval()

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  
    results = model(imgs, size=640) 
    return results

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return redirect("/")

        img_bytes = file.read()
        results = get_prediction(img_bytes)
        results.save(save_dir="static")
        filename = "image0.jpg"

        return render_template(
            "result.html", result_image=filename, model_name='best.pt'
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.debug = True
    app.run()

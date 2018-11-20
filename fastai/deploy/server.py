import os 

clear = lambda: os.system('clear')
clear()

from flask import Flask, request, jsonify
from fastai import accuracy

from fastai.vision import (
    ImageDataBunch, 
    get_transforms, 
    imagenet_stats, 
    create_cnn,
    models,
    open_image
)

app = Flask(__name__)

UPLOAD = "uploads"
path = "tmp/"

data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.load("model")

@app.route('/predict', methods=['POST'])
def predict():
    file_remote = request.files['image']
    filename = file_remote.filename.lower()
    file_local = os.path.join(UPLOAD, filename)
    file_remote.save(file_local)

    prediction = predict_catdog(file_local)
    return jsonify(prediction)


def predict_catdog(file_local):
    img = open_image(file_local)
    prediction = learn.predict(img)

    prediction = {
        "cat" : round(prediction[2][0].item() * 100, 2),
        "dog" : round(prediction[2][1].item() * 100, 2)
    }
    
    return prediction


if __name__ == "__main__":     
    app.run()
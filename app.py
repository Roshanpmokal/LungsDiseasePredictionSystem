from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
from tkinter import Y
import numpy as np
import imghdr
from skimage import transform

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,abort,send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('model.h5')

def predict_label(img_path):
  i = image.load_img(img_path)
  i = np.array(i).astype('float32')/255
  i = transform.resize(i,(150,150,3))
  i = np.expand_dims(i, axis=0)
  # Make prediction
  preds = model.predict(i)
  pred_class=preds.argmax()
  # Arrange the correct return according to the model. 
  return pred_class

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
  if request.method == 'POST':
    img = request.files['my_image']
    img_path = "static/" + img.filename	
    img.save(img_path)

    p = predict_label(img_path)
    str1 = 'This is Normal Case'
    str2 = 'This is COVID-19 Case'
    str3 = 'This is Lungs Opacity Case'
    str4 = 'This is Viral Pnemonia Case'
    if p == 0:
      result = str1
    elif p == 1:
      result = str2
    elif p == 2:
      result = str3
    elif p == 3:
      result = str4

  return render_template("index.html", prediction = result, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
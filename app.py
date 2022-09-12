import json
from textwrap import indent
from typing import Dict
import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
import inference 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import detect_mask_image
from json import load
# Setting custom Page Title and Icon with changed layout and sidebar state
v_name = inference.video_name


def mask_image(image):
    global RGB_img
    # load our serialized face detector model from disk
    #print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    #print("[INFO] loading face mask detector model...")
    model = load_model("mask_detector.model")

    # load the input image from disk and grab the image spatial
    # dimensions

    # image = cv2.imread("./images/out.jpg")

    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    # print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()
    Dictionary = {}
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = True if mask > withoutMask else False

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            str = '{}, {}, {}, {}'.format(startX, startY, endX, endY)
            Dictionary[str] = label
            # display the label and bounding box rectangle on the output
            # frame
            # cv2.putText(image, label, (startX, startY - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            # cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            # RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB
        json_dict = json.dumps(Dictionary, indent=4)
    return json_dict


folder = f'output/out_{v_name}'


def load_images_from_folder(folder):
    k = 0
    images = []
    for filename in os.listdir(folder):

        img = cv2.imread(os.path.join(folder, filename))

        if img is not None:
            k += 1
            print(f"Image id: {k}...............")
            print(filename)
            print(mask_image(img))
    # return images


load_images_from_folder(folder)



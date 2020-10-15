# USAGE
# python human_activity_recognition.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt --input example_activities.mp4
# python human_activity_recognition.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt

import numpy as np
import argparse
import imutils
import sys
import cv2

import torch

import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained human activity recognition model")
ap.add_argument("-c", "--classes", required=True,
                help="path to class labels file")
ap.add_argument("-i", "--input", type=str, default="",
                help="optional path to video file")
args = vars(ap.parse_args())

CLASSES = open(args['classes']).read().strip().split("\n")
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

# load the human activity recognition model

logging.info("Loading human activity recognition model...")

net = cv2.dnn.readNetFromONNX(args["model"])

# grab a pointer to the input video stream

logging.info("Accessing video stream...")

vs = cv2.VideoCapture(args["input"] if args['input'] else 0)

while True:

    frames = []

    for i in range(0, SAMPLE_DURATION):

        (grabbed, frame) = vs.read()

        # if not the video has ended
        if not grabbed:
            logging.info("No frame read from stream - exiting")
            sys.exit(0)

        # otherwise, the frame was read so resize it and add it to frame list
        frame = imutils.resize(frame, width=400)
        frames.append(frame)

    blob = cv2.dnn.blobFromImages(
        frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE),
        (114.7748, 107.7354, 99.4750),
        swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)

    net.setInput(blob)

    outputs = net.forward()

    label = CLASSES[np.argmax(outputs)]

    # loop over our frames
    for frame in frames:

        # draw the predicted activity on the frame

        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        # diplay the frame to our screen
        cv2.imshow("Activity Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key was pressed, break from the loop
        if key == ord("q"):
            break

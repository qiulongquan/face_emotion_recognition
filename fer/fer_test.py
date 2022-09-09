# We install the FER() library to perform facial recognition
# This installation will also take care of any of the above dependencies if they are missing

import json

import cv2
from fer import FER
from PIL import Image, ImageDraw, ImageFont

def scale_to_width(img, width):
    """幅が指定した値になるように、アスペクト比を固定して、リサイズする。
    """
    h, w = img.shape[:2]
    height = round(h * (width / w))
    dst = cv2.resize(img, dsize=(width, height))

    return dst

test_image_one = cv2.imread("./img/3.jpg")
emo_detector = FER(mtcnn=True)
# Capture all the emotions on the image
h,w,_=test_image_one.shape
if w>1000:
    test_image_one = scale_to_width(test_image_one, 1000)
captured_emotions = emo_detector.detect_emotions(test_image_one)
# Print all captured emotions with the image
print(captured_emotions)

# Use the top Emotion() function to call for the dominant emotion in the image
dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
print(dominant_emotion, emotion_score)

# We install the FER() library to perform facial recognition
# This installation will also take care of any of the above dependencies if they are missing
import json
import os
import sys

import cv2
import pandas as pd
from fer import FER, Video
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

location_videofile = "./test_img_video/test3.mp4"

face_detector = FER(mtcnn=True)
# Input the video for processing
input_video = Video(location_videofile)
processing_data = input_video.analyze(face_detector, display=False)
vid_df = input_video.to_pandas(processing_data)
vid_df = input_video.get_first_face(vid_df)
vid_df = input_video.get_emotions(vid_df)

# Plotting the emotions against time in the video
pltfig = vid_df.plot(figsize=(20, 8), fontsize=16).get_figure()
print(type(pltfig))
# <class 'matplotlib.figure.Figure'>
pltfig.show()
pltfig.savefig("img.png")

# We will now work on the dataframe to extract which emotion was prominent in the video
angry = sum(vid_df.angry)
disgust = sum(vid_df.disgust)
fear = sum(vid_df.fear)
happy = sum(vid_df.happy)
sad = sum(vid_df.sad)
surprise = sum(vid_df.surprise)
neutral = sum(vid_df.neutral)

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotions_values = [angry, disgust, fear, happy, sad, surprise, neutral]

score_comparisons = pd.DataFrame(emotions, columns = ['Human Emotions'])
score_comparisons['Emotion Value from the Video'] = emotions_values
print(score_comparisons)

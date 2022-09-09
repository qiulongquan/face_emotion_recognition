import glob
import os

import av  # strealing video library
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from feat import Detector
from PIL import Image
from streamlit_webrtc import webrtc_streamer

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model="svm",
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.cvtColor(cv2.Canny(img,threshold1, threshold2), cv2.COLOR_GRAY2BGR) if canny else img
    return av.VideoFrame.from_ndarray(img, format="bgr24")

if __name__=='__main__':
    test_image = './test_img_video/2.jpg'
    st.title("Hello World.")
    # 画像の確認
    # st.image(test_image)
    # st.markdown("识别结果输出：")
    # image_prediction = detector.detect_image(test_image)
    # image_prediction = image_prediction[["frame","FaceRectX","FaceRectY","FaceRectWidth","FaceRectHeight","FaceScore","anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]]
    # for idx in range(len(image_prediction)):
    #     st.write(image_prediction.iloc[idx])
    # st.video('./test_img_video/test1.mp4', format="video/mp4", start_time=0)
    # 摄像头视频流input处理
    threshold1 = st.slider("Threshold1", min_value=0, max_value=1000, step=1, value=100)
    threshold2 = st.slider("Threshold2", min_value=0, max_value=1000, step=1, value=200)
    canny = st.checkbox("canny")
    webrtc_streamer(key='example',video_frame_callback=callback)
    # if ctx.video_processor:
    #     ctx.video_processor.test_state = st.checkbox('Gray Scale -> Color ')

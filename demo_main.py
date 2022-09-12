import json
import os
from pathlib import Path
from demo_fer import face_emotion
from demo_plate import plate
from demo_license import license
import av  # strealing video library
import cv2
import numpy as np
import streamlit as st
from feat import Detector
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import (RTCConfiguration, WebRtcMode,WebRtcStreamerContext, webrtc_streamer)
import logging

HERE = Path(__file__).parent

def logset():
  logging.basicConfig(filename="demo_log.log",format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
    "%(message)s",force=True,)

def main():
  logset()
  logging.info("main process start...")
  st.header("SEIKOIST Demo")
  pages = {
        "表情認識": face_emotion,
        "免許識別": license,
        "ナンバープレート識別": plate,
  }
  page_titles = pages.keys()
  page_title = st.sidebar.selectbox("Choose the app mode",page_titles,)
  st.subheader(page_title)
  page_func = pages[page_title]
  page_func()
  st.sidebar.markdown("---")


if __name__=='__main__':
  main()

import json
import os
from pathlib import Path

import av  # strealing video library
import cv2
# import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from feat import Detector
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import (RTCConfiguration, WebRtcMode,
                              WebRtcStreamerContext, webrtc_streamer)
import logging

HERE = Path(__file__).parent

def logset():
  logging.basicConfig(filename="demo_log.log",format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
    "%(message)s",force=True,)

def plate():
  logging.info("Starting plate...")
  st.title("Plate app")

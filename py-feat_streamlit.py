import json
import os
import re
from io import StringIO
from pathlib import Path

import av  # strealing video library
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from feat import Detector
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import (RTCConfiguration, WebRtcMode,
                              WebRtcStreamerContext, webrtc_streamer)

HERE = Path(__file__).parent

#py-featでモデルを構築する。色々いじる余地があるので公式ドキュメント参照
detector = Detector(
    face_model="retinaface",
    landmark_model="mobilenet",
    # landmark_model = "mobilefacenet",
    au_model="svm",
    # emotion_model="fer",
    emotion_model="resmasknet",
    # facepose_model="img2pose",
)
MAX_IMAGE_WIDTH=1000
IMG_BGR_PATH='stream_img_bgr.jpg'
emotion_list=["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

def non_negative_for_list(list1):
  list_shape=np.array(list1).shape
  for m in range(list_shape[0]):
      for n in range(list_shape[1]):
          for i in range(list_shape[2]):
              if list1[m][n][i]<0:
                  list1[m][n][i]=1.0
  return list1

def callback(frame):
    try:
      frame_bgr = frame.to_ndarray(format = 'bgr24')
      # 入力frame格式转换变成4维（1,480,640,3）
      frame_bgr=frame_bgr[np.newaxis,:]
      # Return type feat.data.Fex (dataframe)
      detected_faces = detector.detect_faces(frame_bgr)
      # 针对face boxes坐标为负的处理
      # detected_faces=non_negative_for_list(detected_faces)
      detected_landmarks = detector.detect_landmarks(frame_bgr, detected_faces)
      results=detector.detect_emotions(frame_bgr, detected_faces, detected_landmarks)
    except:
      return frame
    else:
    # 从4D变换成3D
      frame_bgr=frame_bgr[0]
      frame_bgr=draw_img_for_frame(frame_bgr,detected_faces,results)
      return av.VideoFrame.from_ndarray(frame_bgr,format = 'bgr24')

def scale_to_width(img, width):
    """幅が指定した値になるように、アスペクト比を固定して、リサイズする。
    """
    w,h = img.size
    height = round(h * (width / w))
    dst = img.resize((width, height))
    return dst

def draw_img(img_rgb,image_prediction):
  draw = ImageDraw.Draw(img_rgb)
  rectcolor = (0, 255, 0)
  linewidth = 2
  textcolor = (255, 0, 0)
  textsize = 14
  font = ImageFont.truetype("arial.ttf", size=textsize)
  labels_list=list(image_prediction.columns.values)
  for idx in range(len(image_prediction)):
    sub_dict={}
    print(image_prediction.iloc[idx])
    x1 = image_prediction.iloc[idx,1]
    y1 = image_prediction.iloc[idx,2]
    w = image_prediction.iloc[idx,3]
    h = image_prediction.iloc[idx,4]
    for i in range(6,13,1):
      sub_dict[labels_list[i]]=image_prediction.iloc[idx,i]
    draw.rectangle([(x1, y1), (x1 + w, y1 + h)], outline=rectcolor, width=linewidth)
    dict = sorted(sub_dict.items(), key=lambda x: x[1], reverse=True)
    # text = "emotions:" +dict[0][0]+'\n'+ json.dumps(dict[0])+'\n'+json.dumps(dict[1])
    text = "emotions:\n" +dict[0][0]+' '+str(dict[0][1])+'\n'+dict[1][0]+' '+str(dict[1][1])
    # x座標はleftと同じ。
    # y座標はtopよりテキストの大きさと矩形の線の太さの半分だけ上にする。
    # テキストの大きさ(=textsize)。矩形の線の太さの半分(=linewidth//2)。
    txpos = (x1, y1 - textsize - linewidth // 2)
    # 文字列"text"が占める領域のサイズを取得
    txw, txh = draw.textsize(text, font=font)

    # テキストを描画する領域を"rectcolor"で塗りつぶし。
    # 左上座標をtxpos、右下座標を (left+txw, top)とする矩形をrectcolor(=赤色)で塗りつぶし。
    draw.rectangle([txpos, (x1 + txw, y1)], outline=rectcolor, width=linewidth)
    # テキストをtextcolor(=白色)で描画
    draw.text(txpos, text, font=font, fill=textcolor)
  return img_rgb

def draw_img_for_frame(frame_bgr,detected_faces,results):
  frame_bgr=cv2pil(frame_bgr)
  draw = ImageDraw.Draw(frame_bgr)
  rectcolor = (0, 255, 0)
  linewidth = 2
  textcolor = (255, 0, 0)
  textsize = 14
  font = ImageFont.truetype("arial.ttf", size=textsize)
  # for idx,result in enumerate(results):
  #       result = dict(zip(emotion_list,result))
  #       result = sorted(result.items(), key=lambda x: x[1], reverse=True)
  #       print("======{0}:{1}\n".format(idx,result))
  for idx,detected_face in enumerate(detected_faces[0]):
    x1 = detected_face[0]
    y1 = detected_face[1]
    x2 = detected_face[2]
    y2 = detected_face[3]
    score=detected_face[4]
    draw.rectangle([(x1, y1), (x2, y2)], outline=rectcolor, width=linewidth)
    result = dict(zip(emotion_list,results[idx]))
    result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    text = "emotions:\n" +result[0][0]+' '+str(result[0][1])+'\n'+result[1][0]+' '+str(result[1][1])
    # x座標はleftと同じ。
    # y座標はtopよりテキストの大きさと矩形の線の太さの半分だけ上にする。
    # テキストの大きさ(=textsize)。矩形の線の太さの半分(=linewidth//2)。
    txpos = (x1, y1 - textsize - linewidth // 2)
    # 文字列"text"が占める領域のサイズを取得
    txw, txh = draw.textsize(text, font=font)
    # テキストを描画する領域を"rectcolor"で塗りつぶし。
    # 左上座標をtxpos、右下座標を (left+txw, top)とする矩形をrectcolor(=赤色)で塗りつぶし。
    draw.rectangle([txpos, (x1 + txw, y1)], outline=rectcolor, width=linewidth)
    # テキストをtextcolor(=白色)で描画
    draw.text(txpos, text, font=font, fill=textcolor)
  return pil2cv(frame_bgr)

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image
  
@st.experimental_memo(show_spinner=False)
def load_image(img_path):
    image = Image.open(img_path)
    w,h=image.size
    if w>MAX_IMAGE_WIDTH:
      image_resized = scale_to_width(image, MAX_IMAGE_WIDTH)
    # image_resized = image_resized[:, :, [2, 1, 0]] # BGR -> RGB
      return image_resized
    else:
      return image

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():
  st.title("Face emotion app")
  #画像アップロードかカメラを選択するようにする。
  # img_source = st.radio("画像のソースを選択してください。",
  #                               ("画像をアップロード", "カメラで撮影"))
  # if img_source == "カメラで撮影":
  #   img_file_buffer = st.camera_input("カメラで撮影")
  # elif img_source == "画像をアップロード":
  #   img_file_buffer = st.file_uploader("ファイルを選択")
  # else:
  #     pass
  # #どちらを選択しても後続の処理は同じ
  # if img_file_buffer is not None:
  #   img_rgb=load_image(img_file_buffer)
  #   st.image(img_rgb, use_column_width=True)
  #   cv2.imwrite(IMG_BGR_PATH,pil2cv(img_rgb))
  #   #py-featの表情解析結果をデータフレーム形式でimage_predictionとする
  #   image_prediction = detector.detect_image(IMG_BGR_PATH)
  #   #感情に関するカラムだけを残す
  #   image_prediction = image_prediction[["frame","FaceRectX","FaceRectY","FaceRectWidth","FaceRectHeight","FaceScore","anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]]
  #   # image_prediction = image_prediction[["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]]
  #   st.write(image_prediction)
  #   img_rgb_draw=draw_img(img_rgb,image_prediction)
  #   st.image(img_rgb_draw,caption=f"Processed image", use_column_width=True)
  #   st.markdown("#### 表情認識完了")
  # 摄像头视频流input处理
  st.title("WebRTC demo")
  webrtc_streamer(key='face emotion recognition',mode=WebRtcMode.SENDRECV,rtc_configuration=RTC_CONFIGURATION,video_frame_callback=callback,media_stream_constraints={"video": True, "audio": False},async_processing=True)


if __name__=='__main__':
  main()

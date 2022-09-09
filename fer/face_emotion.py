# We install the FER() library to perform facial recognition
# This installation will also take care of any of the above dependencies if they are missing
import json

import cv2
from fer import FER
from PIL import Image, ImageDraw, ImageFont

# import matplotlib.pyplot as plt
# %matplotlib inline

img = cv2.imread("./test_img_video/3.jpg")
emo_detector = FER(mtcnn=True)
# Capture all the emotions on the image
captured_emotions = emo_detector.detect_emotions(img)
# Print all captured emotions with the image
print(captured_emotions)
img_tmp_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
img_tmp_rgb = Image.fromarray(img_tmp_rgb)

draw = ImageDraw.Draw(img_tmp_rgb)
rectcolor = (0, 255, 0)
linewidth = 2
textcolor = (255, 255, 255)
textsize = 14
font = ImageFont.truetype("arial.ttf", size=textsize)

for sub_dict in captured_emotions:
    x1 = sub_dict["box"][0]
    y1 = sub_dict["box"][1]
    w = sub_dict["box"][2]
    h = sub_dict["box"][3]
    draw.rectangle([(x1, y1), (x1 + w, y1 + h)], outline=rectcolor, width=linewidth)
    dict = sorted(sub_dict["emotions"].items(), key=lambda x: x[1], reverse=True)
    text = "emotions:" +dict[0][0]+'\n'+ json.dumps(dict[0])+'\n'+json.dumps(dict[1])
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

# img_resize = img_tmp_rgb.resize((img_tmp_rgb.width // 3, img_tmp_rgb.height // 3))
img_tmp_rgb.show()

# Use the top Emotion() function to call for the dominant emotion in the image
dominant_emotion, emotion_score = emo_detector.top_emotion(img)
print(dominant_emotion, emotion_score)

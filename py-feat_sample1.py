import cv2
import numpy as np
from feat import Detector
from PIL import Image
from feat import Detector
from feat.utils import read_pictures
    
detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model="svm",
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

def py_feat_test():
    frame = read_pictures(['./test_img_video/2.jpg'])
    # detector = Detector()
    detected_faces = detector.detect_faces(frame)
    # detected_landmarks = detector.detect_landmarks(frame, detected_faces)
    results=detector.detect_emotions(frame, detected_faces, "detected_landmarks")
    emotion_list=["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    for idx,result in enumerate(results):
        result = dict(zip(emotion_list,result))
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        print("======{0}:{1}\n".format(idx,result))


if __name__=='__main__':
    py_feat_test()

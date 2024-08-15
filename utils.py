from __future__ import annotations
from typing import List, BinaryIO

import cv2
import numpy as np
from config import config


def decode_image(request_photo: np.ndarray) -> np.ndarray:
    nparr = np.frombuffer(request_photo.file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def create_face_dict(age: list, bbox: np.ndarray, conf: float):
    return {'age': age[1], 'gender': age[0], 'bbox': bbox.tolist(), 'conf': conf}


def make_response(age_gender: List[List[str, int]], bboxes: List[np.ndarray], confs: List[np.ndarray]):
    response = [create_face_dict(age, bbox, conf ) for (age, bbox, conf) in zip(age_gender, bboxes, confs)]
    return response

def process_video(request_video: BinaryIO):
    video = cv2.VideoCapture(request_video)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def response_video(age_genders, bboxes, confs):
    response = []
    for age_gender, bbox, conf in zip(age_genders, bboxes, confs):
        response.append(create_face_dict(age_gender[0], bbox[0], conf[0]))
    return response
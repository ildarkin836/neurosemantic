import logging
from distutils.util import strtobool
from typing import List, Union

import cv2
import fastapi
import numpy as np

import uvicorn
from fastapi import FastAPI, File, Header, Request, UploadFile, Response, status, HTTPException
from config import config
from utils import *

import time

from wrappers import Ensemble
from tempfile import NamedTemporaryFile
import os

from models import *


triton_ip = config.triton_ip
triton_port = config.triton_port
fastapi_port = config.fastapi_port


ensemble_model = Ensemble(
    url=f"{triton_ip}:{triton_port}",
    detector_model_name="face_detection",
    classifier_model_name="classifier",
    detector_conf_threshold=config.det_threshold,
    detector_iou_threshold=config.iou_threshold,
)


logger = logging.getLogger("NeuroSemantic")
logger.setLevel("INFO")
app = FastAPI()


@app.post("/recognize_image", response_model=list[BaseResponse])
async def recognize_image(img: UploadFile = File(...)):
    decoded_img = decode_image(img)
    if decoded_img is None:
        raise HTTPException(status_code=400, detail="its not image")
    age_gender, bboxes, confs = ensemble_model(decoded_img)
    if age_gender is None:
        return []
    return make_response(age_gender, bboxes, confs)


@app.post("/recognize_video", response_model=list[BaseResponse])
async def recognize_video(video: UploadFile = File(...)):
    temp = NamedTemporaryFile(delete=False)
    # try:
    try:
        contents = video.file.read()
        with temp as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="There was an error uploading the file")
    finally:
        video.file.close()

    frames = process_video(temp.name)
    age_genders = []
    bboxs = []
    confs = []
    for frame in frames:
        age_gender, bboxes, conf = ensemble_model(frame)
        age_genders.append(age_gender)
        bboxs.append(bboxes)
        confs.append(conf)
    response = response_video(age_genders, bboxs, confs)
    # except Exception:
    #     raise HTTPException(status_code=400, detail="There was an error processing the file")
    # finally:
    #     os.remove(temp.name)

    return response

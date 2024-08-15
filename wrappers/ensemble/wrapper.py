from __future__ import annotations

from wrappers import Detector, Classifier
from .utils import *

import numpy as np

class Ensemble():
    """
    Класс для выполнения обнаружения лиц и классификации с использованием ансамбля моделей.

    Этот класс объединяет детектор лиц и классификатор для идентификации лиц на изображении
    и их классификации по возрасту и полу. Он использует указанные модели для 
    обнаружения и классификации, позволяя настраивать пороги для уверенности в обнаружении 
    и пересечения (IoU).

    Атрибуты:
        detector (Detector): Экземпляр класса Detector для обнаружения лиц.
        classifier (Classifier): Экземпляр класса Classifier для классификации возраста и пола.

    Параметры:
        url (str): URL-адрес для сервисов моделей.
        detector_model_name (str): Имя модели, используемой для обнаружения лиц.
        classifier_model_name (str): Имя модели, используемой для классификации возраста и пола.
        detector_conf_threshold (float, optional): Порог уверенности для детектора лиц. По умолчанию 0.2.
        detector_iou_threshold (float, optional): Порог IoU для детектора лиц. По умолчанию 0.5.
    """

    def __init__(self,
                 url: str, 
                 detector_model_name: str,
                 classifier_model_name: str,
                 detector_conf_threshold: float = 0.2,
                 detector_iou_threshold: float = 0.5,
                ) -> None:
        
        self.detector = Detector(url=url,
                                 model_name=detector_model_name,
                                 conf_threshold=detector_conf_threshold,
                                 iou_threshold=detector_iou_threshold)
        
        self.classifier = Classifier(url=url,
                                     model_name=classifier_model_name)
        
    
    def __call__(self, 
                 orig_image: np.ndarray) -> tuple[list, list, list]:
        """
        Обрабатывает входное изображение для обнаружения лиц и их классификации.

        Этот метод принимает изображение в качестве входных данных, обнаруживает лица на нем 
        и затем классифицирует каждое обнаруженное лицо, чтобы оценить возраст и пол. 
        Если лица не обнаружены, он возвращает пустые списки.

        Параметры:
            orig_image (np.ndarray): Исходное изображение, в котором необходимо обнаружить 
                                      и классифицировать лица. Должно быть в формате 
                                      массива NumPy.

        Возвращает:
            tuple[list, list, list]: Кортеж, содержащий три списка:
                - age_gender (list): Список кортежей, содержащих предсказания возраста и пола 
                                     для каждого обнаруженного лица.
                - det_bboxes (list): Список ограничивающих рамок для обнаруженных лиц.
                - det_conf (list): Список оценок уверенности, соответствующих каждому обнаруженному лицу.
        """

        det_bboxes, det_conf, det_classid, landmarks = self.detector(orig_image)
        
        if not np.any(det_bboxes):
            return [], [], []
        
        faces_batch = [crop_face(orig_image, bbox) for bbox in det_bboxes]
        age_gender = [self.classifier(face) for face in faces_batch]
        return age_gender, det_bboxes, det_conf
        



        
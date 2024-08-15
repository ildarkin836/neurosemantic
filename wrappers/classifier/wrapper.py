from __future__ import annotations
from typing import List

from ..base_wrapper import Wrapper

import numpy as np
import cv2
from easydict import EasyDict as edict
from .utils import *


class Classifier(Wrapper):
    """
    Класс для классификации возраста и пола на основе входных изображений.

    Этот класс наследует от Wrapper и предоставляет функциональность для 
    обработки изображений и получения предсказаний возраста и пола. 
    Он включает методы предварительной и последующей обработки данных.

    Атрибуты:
        meta (edict): Словарь метаданных, содержащий информацию о модели, 
                      включая минимальный и максимальный возраст, средний возраст, 
                      количество классов и другие параметры.
        mean (tuple): Средние значения для нормализации входных изображений.
        std (tuple): Стандартные отклонения для нормализации входных изображений.

    Параметры:
        url (str): URL-адрес для доступа к модели.
        model_name (str): Имя модели, используемой для классификации.
    """
    def __init__(self, 
                 url: str, 
                 model_name: str,
                 ) -> None:
        
        super().__init__(url=url, model_name=model_name)

        meta = {'min_age': 1, 'max_age': 95, 'avg_age': 48.0, 'num_classes': 3, 'in_chans': 3, 'with_persons_model': False, 'disable_faces': False,
                'use_persons': False, 'only_age': False, 'num_classes_gender': 2, 'use_person_crops': False, 'use_face_crops': True}
        self.meta = edict(meta)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def preprocess(self, 
                   batch: List[np.ndarray]) -> List[np.ndarray]:
        """
        Предварительная обработка входного батча изображений.

        Этот метод применяет преобразование к каждому изображению в батче, 
        чтобы подготовить их для классификации.

        Параметры:
            batch (List[np.ndarray]): Список изображений в формате массива NumPy 
                                      для предварительной обработки.

        Возвращает:
            List[np.ndarray]: Список преобразованных изображений.
        """
        transformed_batch = [letterbox(img) for img in batch]
        return transformed_batch
    
    def postprocess(self, 
                   tensor: np.ndarray) -> tuple[str, float]:
        """
        Постобработка выходного тензора для извлечения предсказаний.

        Этот метод принимает выходной тензор и извлекает предсказания возраста 
        и пола на основе значений в тензоре.

        Параметры:
            tensor (np.ndarray): Выходной тензор, полученный от модели после 
                                классификации.

        Возвращает:
            tuple[str, float]: Кортеж, содержащий:
                - gender (str): Предсказанный пол ("M" для мужского, "F" для женского).
                - age (float): Предсказанный возраст.
        """
        gemm_result, add_result = tensor[1], tensor[0]

        reduce_max = np.max(add_result, axis=1, keepdims=False)
        mul = np.multiply(reduce_max, 0.5)
        output = np.add(mul, gemm_result)[0]
        
        age = output[2] * (self.meta.max_age - self.meta.min_age) + self.meta.avg_age
        age = int(age)
        gender = "M" if output[0] > output[1] else "F"

        return gender, age
    
    
    

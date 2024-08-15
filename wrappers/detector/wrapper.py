from __future__ import annotations
from typing import Any, List, Optional

from ..base_wrapper import Wrapper

import numpy as np
import math
import cv2


class Detector(Wrapper):
    """
    Класс для обнаружения лиц на изображениях с использованием нейронной сети.

    Этот класс наследует от Wrapper и предоставляет функциональность для
    обнаружения лиц на входных изображениях. Он включает методы предварительной
    и последующей обработки данных, а также вспомогательные методы для работы с якорями и координатами.

    Атрибуты:
        input_height (int): Высота входного изображения для модели.
        input_width (int): Ширина входного изображения для модели.
        conf_threshold (float): Пороговое значение для уверенности в обнаружении лица.
        iou_threshold (float): Пороговое значение для пересечения по площади (IoU) при подавлении немаксимумов.
        reg_max (int): Максимальное значение регрессии для координат ограничивающих рамок.
        project (np.ndarray): Массив для проекции регрессионных значений.
        strides (tuple): Шаги сетки якорей.
        feats_hw (list): Размеры карт признаков для каждого шага сетки.
        anchors (dict): Словарь якорей для каждого шага сетки.

    Параметры:
        url (str): URL-адрес для доступа к модели.
        model_name (str): Имя модели, используемой для обнаружения лиц.
        conf_threshold (float, optional): Пороговое значение для уверенности в обнаружении лица. По умолчанию 0.2.
        iou_threshold (float, optional): Пороговое значение для IoU при подавлении немаксимумов. По умолчанию 0.5.
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        conf_threshold: float = 0.2,
        iou_threshold: float = 0.5,
    ) -> None:

        super().__init__(url=url, model_name=model_name)

        self.input_height, self.input_width = self.triton_model.inputs_shape[0][-2:]
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.reg_max = 16
        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [
            (
                math.ceil(self.input_height / self.strides[i]),
                math.ceil(self.input_width / self.strides[i]),
            )
            for i in range(len(self.strides))
        ]
        self.anchors = self.make_anchors(self.feats_hw)

    def make_anchors(self, feats_hw: List[Any], grid_cell_offset: float = 0.5) -> dict:
        """
        Генерирует якоря на основе карт признаков.

        Этот метод создает якоря для каждого шага сетки на основе размеров карт признаков.

        Параметры:
            feats_hw (list): Список размеров карт признаков для каждого шага сетки.
            grid_cell_offset (float, optional): Смещение ячейки сетки. По умолчанию 0.5.

        Возвращает:
            dict: Словарь якорей для каждого шага сетки.
        """
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset  # shift x
            y = np.arange(0, h) + grid_cell_offset  # shift y
            sx, sy = np.meshgrid(x, y)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Вычисляет softmax функцию по заданной оси.

        Этот метод применяет softmax функцию к входному массиву по указанной оси.

        Параметры:
            x (np.ndarray): Входной массив.
            axis (int, optional): Ось, по которой применяется softmax. По умолчанию 1.

        Возвращает:
            np.ndarray: Результат применения softmax функции.
        """
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def preprocess(
        self, image: np.ndarray, keep_ratio: bool = True
    ) -> List[np.ndarray, int, int, int, int]:
        """
        Предварительная обработка входного изображения.

        Этот метод применяет преобразования к изображению, чтобы подготовить его для обнаружения лиц.
        Он выполняет ресайз изображения с сохранением или без сохранения соотношения сторон,
        а также добавляет черные границы при необходимости.

        Параметры:
            image (np.ndarray): Входное изображение в формате массива NumPy.
            keep_ratio (bool, optional): Флаг, указывающий, нужно ли сохранять соотношение сторон при ресайзе. По умолчанию True.

        Возвращает:
            List[np.ndarray, int, int, int, int]: Список, содержащий:
                - preprocessed_image (np.ndarray): Предварительно обработанное изображение.
                - new_height (int): Новая высота изображения после ресайза.
                - new_width (int): Новая ширина изображения после ресайза.
                - top (int): Верхний отступ для добавления черных границ.
                - left (int): Левый отступ для добавления черных границ.
        """
        top, left, new_h, new_w = 0, 0, self.input_width, self.input_height
        if keep_ratio and image.shape[0] != image.shape[1]:
            hw_scale = image.shape[0] / image.shape[1]
            if hw_scale > 1:
                new_h, new_w = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - new_w) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    0,
                    0,
                    left,
                    self.input_width - new_w - left,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )  # add border
            else:
                new_h, new_w = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - new_h) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    top,
                    self.input_height - new_h - top,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
        else:
            img = cv2.resize(
                image,
                (self.input_width, self.input_height),
                interpolation=cv2.INTER_AREA,
            )
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img).astype(np.float32)
        img /= 255
        return img, new_h, new_w, top, left

    def postprocess(
        self, tensor: np.ndarray, scale_h: float, scale_w: float, padh: int, padw: int
    ) -> np.ndarray:
        """
        Постобработка выходного тензора для получения результатов обнаружения.

        Этот метод принимает выходной тензор модели и применяет к нему различные преобразования,
        чтобы получить ограничивающие рамки, оценки уверенности, классы и ключевые точки лиц.
        Он также выполняет подавление немаксимумов для удаления перекрывающихся рамок.

        Параметры:
            tensor (np.ndarray): Выходной тензор, полученный от модели после обнаружения.
            scale_h (float): Коэффициент масштабирования по высоте.
            scale_w (float): Коэффициент масштабирования по ширине.
            padh (int): Верхний отступ, добавленный при предварительной обработке.
            padw (int): Левый отступ, добавленный при предварительной обработке.

        Возвращает:
            np.ndarray: Массив, содержащий:
                - det_bboxes (np.ndarray): Ограничивающие рамки обнаруженных лиц.
                - det_conf (np.ndarray): Оценки уверенности для каждого обнаруженного лица.
                - det_classid (np.ndarray): Классы обнаруженных лиц.
                - landmarks (np.ndarray): Ключевые точки лиц.
        """
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(tensor):
            stride = int(self.input_height / pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))

            box = pred[..., : self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4 : -15])).reshape((-1, 1))
            kpts = pred[..., -15:].reshape(
                (-1, 15)
            )  ### x1,y1,score1, ..., x5,y5,score5

            # tmp = box.reshape(self.feats_hw[i][0], self.feats_hw[i][1], 4, self.reg_max)
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1, 4))

            bbox = (
                self.distance2bbox(
                    self.anchors[stride],
                    bbox_pred,
                    max_shape=(self.input_height, self.input_width),
                )
                * stride
            )
            kpts[:, 0::3] = (
                kpts[:, 0::3] * 2.0
                + (self.anchors[stride][:, 0].reshape((-1, 1)) - 0.5)
            ) * stride
            kpts[:, 1::3] = (
                kpts[:, 1::3] * 2.0
                + (self.anchors[stride][:, 1].reshape((-1, 1)) - 0.5)
            ) * stride
            kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]])  ###合理使用广播法则
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1, 15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1, 15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)

        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  ####xywh
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)  ####max_class_confidence

        mask = confidences > self.conf_threshold
        bboxes_wh = bboxes_wh[mask]  ###合理使用广播法则
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]

        indices = cv2.dnn.NMSBoxes(
            bboxes_wh.tolist(),
            confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )
        if len(indices) > 0:
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            landmarks = landmarks[indices]
            return mlvl_bboxes, confidences, classIds, landmarks
        else:
            return np.array([]), np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points: np.ndarray, distance: np.ndarray, max_shape=None):
        """
        Преобразует расстояния в ограничивающие рамки.

        Этот метод преобразует расстояния от якорей до ограничивающих рамок в координаты рамок.

        Параметры:
            points (np.ndarray): Якоря.
            distance (np.ndarray): Расстояния от якорей до ограничивающих рамок.
            max_shape (tuple, optional): Максимальные размеры изображения. По умолчанию None.

        Возвращает:
            np.ndarray: Ограничивающие рамки.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def __call__(self, tensor: np.ndarray):
        """
        Выполняет обнаружение лиц на входном изображении.

        Этот метод принимает изображение в качестве входных данных, предварительно обрабатывает его,
        передает в модель для обнаружения и применяет постобработку к выходному тензору.

        Параметры:
            tensor (np.ndarray): Входное изображение в формате массива NumPy.

        Возвращает:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Кортеж, содержащий:
                - det_bboxes (np.ndarray): Ограничивающие рамки обнаруженных лиц.
                - det_conf (np.ndarray): Оценки уверенности для каждого обнаруженного лица.
                - det_classid (np.ndarray): Классы обнаруженных лиц.
                - landmarks (np.ndarray): Ключевые точки лиц.
        """
        img, new_h, new_w, top, left = self.preprocess(tensor)
        scale_h, scale_w = tensor.shape[0] / new_h, tensor.shape[1] / new_w
        tensor = self.triton_model([img])
        det_bboxes, det_conf, det_classid, landmarks = self.postprocess(
            tensor, scale_h, scale_w, top, left
        )
        return det_bboxes.astype(int), det_conf.astype(float), det_classid, landmarks

    def draw_detections(self, image, boxes, scores, kpts):
        for box, score, kp in zip(boxes, scores, kpts):
            x, y, w, h = box.astype(int)
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
            cv2.putText(
                image,
                "face:" + str(round(score, 2)),
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                thickness=2,
            )
            for i in range(5):
                cv2.circle(
                    image,
                    (int(kp[i * 3]), int(kp[i * 3 + 1])),
                    4,
                    (0, 255, 0),
                    thickness=-1,
                )
        return image


if __name__ == "__main__":
    img = cv2.imread("1.jpg")

    detector = Detector("localhost:8000", "face_detection")
    det_bboxes, det_conf, det_classid, landmarks = detector(img)
    cv2.imwrite(
        "res.jpg", detector.draw_detections(img, det_bboxes, det_conf, landmarks)
    )

from typing import Any
from .triton_model import TritonInference
import numpy as np

class Wrapper():
    """
    Базовый класс-обертка для работы с моделями Triton Inference Server.

    Этот класс предоставляет общую функциональность для предварительной и 
    последующей обработки данных, а также для выполнения инференса с использованием 
    модели, развернутой на Triton Inference Server.

    Атрибуты:
        triton_model (TritonInference): Экземпляр класса TritonInference, 
                                         используемый для выполнения инференса.

    Параметры:
        url (str): URL-адрес для доступа к Triton Inference Server.
        model_name (str): Имя модели, используемой для инференса.
    """
    def __init__(self,
                 url: str,
                 model_name: str ) -> None:
        """
        Инициализирует экземпляр класса Wrapper.

        Параметры:
            url (str): URL-адрес для доступа к Triton Inference Server.
            model_name (str): Имя модели, используемой для инференса.
        """
        
        self.triton_model = TritonInference(url=url, model_name=model_name)


    def __call__(self, 
                 tensor: np.ndarray) -> Any:
        """
        Выполняет инференс на входном тензоре.

        Этот метод принимает входной тензор, выполняет его предварительную обработку, 
        передает в модель для инференса и затем применяет постобработку к выходному тензору.

        Параметры:
            tensor (np.ndarray): Входной тензор в формате массива NumPy.

        Возвращает:
            Any: Результат постобработки, полученный после инференса.
        """
        
        tensor = self.preprocess(tensor)
        tensor = self.triton_model(tensor)
        result = self.postprocess(tensor)
        return result

    def preprocess(self, 
                   tensor: np.ndarray) -> Any:
        """
        Предварительная обработка входного тензора.

        Этот метод предназначен для предварительной обработки входного тензора перед 
        передачей его в модель. Метод должен быть переопределен в производных классах.

        Параметры:
            tensor (np.ndarray): Входной тензор в формате массива NumPy.

        Возвращает:
            Any: Предварительно обработанный тензор.
        """
        
        pass

    def postprocess(self, 
                    tensor: np.ndarray) -> Any:
        """
        Постобработка выходного тензора.

        Этот метод предназначен для обработки выходного тензора после инференса. 
        Метод должен быть переопределен в производных классах.

        Параметры:
            tensor (np.ndarray): Выходной тензор, полученный от модели после инференса.

        Возвращает:
            Any: Результат постобработки выходного тензора.
        """
        pass


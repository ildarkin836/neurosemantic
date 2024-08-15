from typing import List
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
import uuid


class TritonInference:
    def __init__(self, 
                 url: str, 
                 model_name: str) -> None:
        self.model_name = model_name
        self.url = url

        self.triton_client = httpclient.InferenceServerClient(
            url=self.url, verbose=False
        )
        self.model_config = self.triton_client.get_model_metadata(
            model_name=self.model_name, model_version="1"
        )
        
        self.inputs_name = [input_tensor['name'] for input_tensor in self.model_config['inputs']]
        self.outputs_name = [output_tensor['name'] for output_tensor in self.model_config['outputs']]

        self.inputs_shape = [input_tensor['shape'] for input_tensor in self.model_config['inputs']]

        self.outputs_shape = [output_tensor['shape'] for output_tensor in self.model_config['outputs']]

        self.inputs = [httpclient.InferInput(name, shape, "FP32") for (name, shape) in zip(self.inputs_name, self.inputs_shape)]
        self.outputs = [httpclient.InferRequestedOutput(name) for name in self.outputs_name]

    def __call__(self, 
                 batch: List[np.ndarray]) -> np.ndarray:
        
        self.input = [input_tensor.set_data_from_numpy(data) for (input_tensor, data) in zip(self.inputs, batch)]
        results = self.triton_client.infer(
            self.model_name,
            self.inputs,
            outputs=self.outputs,
        )
        return [results.as_numpy(name) for name in self.outputs_name]
    

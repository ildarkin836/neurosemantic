from pydantic import BaseModel
from typing import List


class BaseResponse(BaseModel):
    age: int
    gender: str
    bbox: List[int]
    conf: float
    
    
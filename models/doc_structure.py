from docarray import DocList, BaseDoc
from docarray.typing import TorchTensor, AnyEmbedding
from typing import Optional

class PointCloud(BaseDoc):
    id: None
    points: TorchTensor
    embedding: Optional[AnyEmbedding]
    label: int
    object: str
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxslim {f} {f} && open {f}')  # pip install onnxslim
    ```
"""

from .block import (
    SPPF,
    ADown,
    Bottleneck,
    C2f,
)
from .conv import (
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DepthwiseConv,
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect, v10Detect
from .transformer import (
    LayerNorm2d,
)

__all__ = (
    "Conv",
    "Conv2",
    "ConvTranspose",
    "Concat",
    "LayerNorm2d",
    "SPPF",
    "C2f",
    "Bottleneck",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "RTDETRDecoder",
    "OBB",
    "WorldDetect",
    "v10Detect",
    "ADown",
    "DepthwiseConv",
)

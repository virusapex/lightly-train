#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, List, Optional, Union

import torch

from lightly_train.types import PathLike

try:
    import tensorrt as trt  # type: ignore[import-untyped,import-not-found]
except ImportError:
    trt = None

logger = logging.getLogger(__name__)


if trt is not None:

    class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        """Standard Int8 Entropy Calibrator for TensorRT.

        This calibrator reads a batch of data from an iterator, copies it to
        device memory (if not already there), and passes the device pointer to
        TensorRT. It also handles reading and writing the calibration cache.
        """

        def __init__(
            self,
            calibration_data: Iterator[torch.Tensor],
            cache_file: PathLike,
            batch_size: int = 1,
        ) -> None:
            """
            Args:
                calibration_data:
                    Iterator yielding batch tensors (N, C, H, W).
                    Data should be preprocessed (normalized, resized) exactly as
                    during inference.
                cache_file:
                    Path to read/write calibration cache.
                batch_size:
                    Batch size of the calibration data.
            """
            super().__init__()
            self.calibration_data = calibration_data
            self.cache_file = Path(cache_file)
            self.batch_size = batch_size
            self.device_input: Optional[torch.Tensor] = None

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names: List[str]) -> List[int] | None:
            try:
                data = next(self.calibration_data)
            except StopIteration:
                return None

            # Ensure data is on GPU and contiguous
            if not isinstance(data, torch.Tensor):
                 raise TypeError(f"Expected torch.Tensor, got {type(data)}")

            if data.device.type == "cpu":
                data = data.cuda()
            
            data = data.contiguous()

            # Keep reference to avoid GC
            self.device_input = data

            # Return pointer
            return [int(data.data_ptr())]

        def read_calibration_cache(self) -> bytes | None:
            if self.cache_file.exists():
                logger.info(f"Reading calibration cache from {self.cache_file}")
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            logger.info(f"Writing calibration cache to {self.cache_file}")
            # Ensure directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "wb") as f:
                f.write(cache)

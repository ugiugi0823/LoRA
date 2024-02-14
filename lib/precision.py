from contextlib import contextmanager
from typing import Any, Generator, Literal

import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import Module

from lightning.fabric.plugins.precision.utils import _convert_fp_tensor
from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin


class HalfPrecisionPlugin(PrecisionPlugin):
    """Plugin for training with half precision.

    Args:
        precision: Whether to use ``torch.float16`` (``'16-true'``) or ``torch.bfloat16`` (``'bf16-true'``).
    """

    precision: Literal["bf16-true", "16-true"] = "16-true"

    def __init__(self, precision: Literal["bf16-true", "16-true"] = "16-true") -> None:
        self.precision = precision
        self._desired_input_dtype = torch.bfloat16 if precision == "bf16-true" else torch.float16

    def convert_module(self, module: Module) -> Module:
        return module.to(dtype=self._desired_input_dtype)

    @contextmanager
    def init_context(self) -> Generator[None, None, None]:
        """A context manager to change the default tensor type when initializing module parameters or tensors.

        See: :meth:`torch.set_default_dtype`
        """
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self._desired_input_dtype)
        yield
        torch.set_default_dtype(default_dtype)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """A context manager to change the default tensor type when tensors get created during the module's
        forward.

        See: :meth:`torch.set_default_tensor_type`
        """
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self._desired_input_dtype)
        yield
        torch.set_default_dtype(default_dtype)

    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self._desired_input_dtype)
# Copyright (c) 2019,20-21-22-25 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Extracted unchanged from kaolin.utils.testing; tensor validation only.
import torch

def check_tensor(tensor, shape=None, dtype=None, device=None, throw=True):
    """Check if :class:`torch.Tensor` is valid given set of criteria.

    Args:
        tensor (torch.Tensor): the tensor to be tested.
        shape (list or tuple of int, optional): the expected shape,
            if a dimension is set at ``None`` then it's not verified.
        dtype (torch.dtype, optional): the expected dtype.
        device (torch.device, optional): the expected device.
        throw (bool): if true (default), will throw if checks fail

    Return:
        (bool) True if checks pass
    """
    if shape is not None:
        if len(shape) != tensor.ndim:
            if throw:
                raise ValueError(f"tensor have {tensor.ndim} ndim, should have {len(shape)}")
            return False
        for i, dim in enumerate(shape):
            if dim is not None and tensor.shape[i] != dim:
                if throw:
                    raise ValueError(f"tensor shape is {tensor.shape}, should be {shape}")
                return False
    if dtype is not None and dtype != tensor.dtype:
        if throw:
            raise TypeError(f"tensor dtype is {tensor.dtype}, should be {dtype}")
        return False
    if device is not None:
        expected_device = torch.device(device)
        actual_device = tensor.device
        mismatch = (expected_device.type != actual_device.type) or (
            expected_device.index is not None and expected_device.index != actual_device.index
        )
        if mismatch:
            if throw:
                raise TypeError(f"tensor device is {actual_device}, should be {expected_device}")
            return False
    return True


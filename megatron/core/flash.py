# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import os
import warnings
from datetime import timedelta
from functools import partial
from itertools import cycle
from typing import Callable, List, Optional

import torch

from .utils import GlobalMemoryBuffer

import fastalltoall.FlashAllToAll
import numpy as np

flash_scheduler = None
send_tensor = None
lbsend_tensor = None
lbrecv_tensor = None
cros1_tensor = None
cros2_tensor = None
rstr_tensor = None
megatron_workloads = []


def init_flash(args):
     # -----------------------------------------------------------------------
    # FLASH INITIALIATION
    # -----------------------------------------------------------------------
    device_count = torch.cuda.device_count()
    this_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    master_addr = os.environ['MASTER_ADDR']
    master_port = 31000
    if this_rank == 0:
        server_store = torch.distributed.TCPStore(master_addr, master_port, world_size, True, timedelta(seconds=30))
    else:
        client_store = torch.distributed.TCPStore(master_addr, master_port, world_size, False)

    if this_rank == 0:
        id_str = torch.cuda.nccl.unique_id()
        server_store.set("commID", id_str)
    else:
        id_str = client_store.get("commID")
    torch.distributed.barrier()
    torch_dtype_map = {
        torch.float32: 7,
        torch.float64: 8,
        torch.float16: 6,
        torch.bfloat16: 9,
        torch.uint8: 1,
        torch.uint32: 3,
        torch.uint64: 5,
        torch.int8: 0,
        torch.int32: 2,
        torch.int64: 4
    }
    global flash_scheduler
    flash_scheduler = fastalltoall.FlashAllToAll.flash_t(this_rank, world_size, world_size // device_count, device_count, args.hidden_size, torch_dtype_map[args.params_dtype], id_str)
    global send_tensor
    global lbsend_tensor
    global lbrecv_tensor
    global cros1_tensor
    global cros2_tensor
    global rstr_tensor
    send_tensor = torch.zeros(size=[204800, args.hidden_size],dtype=args.params_dtype,device=torch.cuda.current_device()).contiguous()
    lbsend_tensor = torch.zeros(size=[2048, args.hidden_size],dtype=args.params_dtype,device=torch.cuda.current_device()).contiguous()
    lbrecv_tensor = torch.zeros(size=[2048, args.hidden_size],dtype=args.params_dtype,device=torch.cuda.current_device()).contiguous()
    cros1_tensor = torch.zeros(size=[204800, args.hidden_size],dtype=args.params_dtype,device=torch.cuda.current_device()).contiguous()
    cros2_tensor = torch.zeros(size=[204800, args.hidden_size],dtype=args.params_dtype,device=torch.cuda.current_device()).contiguous()
    rstr_tensor = torch.zeros(size=[102400, args.hidden_size],dtype=args.params_dtype,device=torch.cuda.current_device()).contiguous()

    # -----------------------------------------------------------------------
    # END OF FLASH INITIALIATION
    # -----------------------------------------------------------------------

def get_flash():
    """Get the expert-model-parallel group the caller rank belongs to."""
    assert (
        flash_scheduler is not None
    ), 'flash scheduler is not initialized'
    return flash_scheduler

def get_buffers():
    return send_tensor, lbsend_tensor, lbrecv_tensor, cros1_tensor, cros2_tensor, rstr_tensor

def add_workloads():

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
buffer_tensors = None
megatron_workloads = []
stored_id = 0
hidden_size = 0
local_rank = 0
params_dtype = None
Timestamps = [[], [], [], []]
if_use_flash = None


moe_throughput_iteration = []
iteration_id = 0

def init_flash(args):
     # -----------------------------------------------------------------------
    # FLASH INITIALIATION
    # -----------------------------------------------------------------------
    device_count = torch.cuda.device_count()
    this_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print(f"this rank: {this_rank}, current rank: {os.environ['RANK']}, local rank: {args.local_rank}")
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
    global hidden_size
    hidden_size = args.hidden_size
    global params_dtype
    params_dtype = args.params_dtype
    global local_rank
    local_rank = this_rank % device_count
    global flash_scheduler
    flash_scheduler = fastalltoall.FlashAllToAll.flash_t(this_rank, world_size, world_size // device_count, device_count, args.hidden_size, torch_dtype_map[args.params_dtype], id_str)
    global buffer_tensors
    buffer_tensors = []
    for i in range(6):
        buffer_tensors.append(torch.zeros(size=[2048, args.hidden_size],dtype=args.params_dtype,device=f"cuda:{local_rank}", requires_grad=False).contiguous())

    global if_use_flash
    if_use_flash = True
    # -----------------------------------------------------------------------
    # END OF FLASH INITIALIATION
    # -----------------------------------------------------------------------

def get_flash():
    """Get the expert-model-parallel group the caller rank belongs to."""
    assert (
        flash_scheduler is not None
    ), 'flash scheduler is not initialized'
    return flash_scheduler


def get_buffers(buffer_szs):
    global buffer_tensors
    for i in range(6):
        if buffer_szs[i] >= buffer_tensors[i].size(dim=0):
            prev_tensor = buffer_tensors[i]
            buffer_tensors[i] = torch.zeros(size=[buffer_szs[i] * 2, hidden_size],dtype=params_dtype,device=f"cuda:{local_rank}", requires_grad=False).contiguous()
            del prev_tensor
    return buffer_tensors


def add_workloads(workload):
    global megatron_workloads
    global stored_id
    megatron_workloads.append(np.ravel(workload))
    print(stored_id)
    stored_id += 1
    WRITE_FREQUENCY = 100
    if stored_id % WRITE_FREQUENCY == 0:
        with open("4n_megatron_workload.txt", "a") as myfile:
            for i in range(stored_id - WRITE_FREQUENCY, stored_id):
                for idx in range(megatron_workloads[i].shape[0]):
                     myfile.write(f"{megatron_workloads[i][idx]}\n")


def record_timestamp(idx, ts, if_print):
    global Timestamps
    Timestamps[idx].append(ts)
    if if_print:
        print(f"timestamp: {ts} ms")
    if len(Timestamps[idx]) % 100 == 0:
        for i in range(4):
            print(f"idx {i} ts sum: {np.sum(Timestamps[i])} ms")


def record_throughput(tput):
    global moe_throughput_iteration
    global iteration_id
    global if_use_flash
    moe_throughput_iteration.append(tput)
    iteration_id += 1
    WRITE_FREQUENCY = 10
    print(f"Iteration {iteration_id}, tput: {tput}")
    if iteration_id % WRITE_FREQUENCY == 0:
        with open(f"4n_perf_e32_t4_" + ("flash" if if_use_flash else "rccl" ) + ".txt", "a") as myfile:
            for i in range(iteration_id - WRITE_FREQUENCY, iteration_id):
                myfile.write(f"{moe_throughput_iteration[i]:.1f}\n")


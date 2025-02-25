This repository contains two components:
* `fastalltoall` includes the source code of FLASH All-to-All scheduler and its test program, which can run independently
* `Megatron-LM` includes the training framework that can train the mixure-of-expert (MoE) models, which can uses the default or the FLASH All-to-All scheduler

# Run Scheduler Independently
Check the `README` in `fastalltoall` to run and test FLASH scheduler.

# Use Scheduler When Training MoE Models
First copy the `fastalltoall` to the `Megatron-LM` and compile the Python interfaces with the following command:
```
cp -r  fastalltoall Megatron-LM/
cd Megatron-LM/fastalltoall
make clean
make flash
```

To train the MoE model, check the `README` in the `Megatron-LM` to prepare training data.
To launch MoE training on multiple nodes, alter `MASTER_ADDR`, `NNODES`, `NODE_RANK` in `moe.sh` at each node and then run:
```
cd Megatron-LM
bash moe.sh  ~/megatron_data/checkpoint ./ ~/megatron_data/gpt2bpe_text_document
# the data path is: ~/megatron_data/gpt2bpe_text_document, you may change this if the data path is different
```
Make sure all the nodes have the same code.
When you change node number, also change the `--num-experts`, `--expert-model-parallel-size`, `--global-batch-size` in `moe.sh`, making sure (1) the total expert number equals the total GPU number, (2) global-batch-size = expert number * micro-batch-size.
You may also alter other parameters in `moe.sh` to change the MoE model structure.

You can turn on/off FLASH scheduler by setting `if_use_flash` to `True` or `False` in `megatron/core/flash.py`.
Setting to False means use the default RCCL all-to-all scheduling algorithm.

# Dependency and Docker

There is a docker container that has already installed all the dependencies and prepared the datasets. You can launch the docker with
```
bash /home/yiran.lei/docker_script/megatron.sh
```
Or download the docker image as follows and prepare datasets by yourself:
```
docker pull rocm/megatron-lm
```

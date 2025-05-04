# Copyright (c) 2025 SandAI. All Rights Reserved.
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

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ALGO=^NVLS

export PAD_HQ=1
export PAD_DURATION=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OFFLOAD_T5_CACHE=true
export OFFLOAD_VAE_CACHE=true
export TORCH_CUDA_ARCH_LIST="8.9;9.0"

GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
DISTRIBUTED_ARGS="
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:6009 \
    --nnodes=1 \
    --nproc_per_node=$GPUS_PER_NODE
"

MAGI_ROOT=$(git rev-parse --show-toplevel)
LOG_DIR=log_$(date "+%Y-%m-%d_%H:%M:%S").log

export PYTHONPATH="$MAGI_ROOT:$PYTHONPATH"
SDEDIT_STEPS=0
torchrun $DISTRIBUTED_ARGS inference/pipeline/entry.py \
    --config_file example/sdedit/24B_config_wp.json \
    --mode iv2v \
    --prompt "A scene with smoke and shoe moving forward" \
    --image_path /svl/u/zzli/projects/data/examples_1023/fluid/smoke_1.png \
    --output_path example/assets/output_wp_24fps_sdedit_${SDEDIT_STEPS}_steps_8.mp4 \
    --reference_path /viscam/projects/neural_wind_tunnel/Wonderland2/output/genesis/smoke_1/Gen-03-05_01-02-55/simulation/traj_00/render_video.mp4 \
    --start_step $SDEDIT_STEPS 
# no sdedit
# torchrun $DISTRIBUTED_ARGS inference/pipeline/entry.py \
#     --config_file example/sdedit/24B_config_wp.json \
#     --mode i2v \
#     --prompt "A scene with smoke and interaction with the shoe" \
#     --image_path /svl/u/zzli/projects/data/examples_1023/fluid/smoke_1.png \
#     --output_path example/assets/output_wp_steps_8.mp4 
#     --reference_path /viscam/projects/neural_wind_tunnel/Wonderland2/output/genesis/smoke_1/Gen-27-04_05-40-09/simulation/traj_00/render_video.mp4 \

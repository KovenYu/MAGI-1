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


import torch
import torch.distributed as dist

from inference.common import MagiConfig, event_path_timer, print_rank_0, set_random_seed
from inference.infra.distributed import dist_init
from inference.model.dit import get_dit

from .prompt_process import get_txt_embeddings
from .video_generate import generate_per_chunk
from .video_process import post_chunk_process, process_image, process_prefix_video, save_video_to_disk


class MagiPipeline:
    def __init__(self, config_path):
        self.config = MagiConfig.from_json(config_path)
        if self.config.runtime_config.seed is not None:
            set_random_seed(self.config.runtime_config.seed)
        dist_init(self.config)
        print_rank_0(self.config)

    def run_text_to_video(self, prompt: str, output_path: str):
        self._run(prompt, None, output_path)

    def run_image_to_video(self, prompt: str, image_path: str, output_path: str):
        prefix_video = process_image(image_path, self.config)
        self._run(prompt, prefix_video, output_path)

    def run_video_to_video(self, prompt: str, prefix_video_path: str, output_path: str):
        prefix_video = process_prefix_video(prefix_video_path, self.config)
        self._run(prompt, prefix_video, output_path)

    def _run(self, prompt: str, prefix_video: torch.Tensor, output_path: str):
        caption_embs, emb_masks = get_txt_embeddings(prompt, self.config)
        dit = get_dit(self.config)
        videos = torch.cat(
            [
                post_chunk_process(chunk, self.config)
                for chunk in generate_per_chunk(
                    model=dit, prompt=prompt, prefix_video=prefix_video, caption_embs=caption_embs, emb_masks=emb_masks
                )
            ],
            dim=0,
        )
        
        self.config.runtime_config.num_frames = 191
        caption_embs, emb_masks = get_txt_embeddings(prompt, self.config)
        dit = get_dit(self.config)
        for iii in range(3):        
            # Start timing video generation if rank 0
            if dist.get_rank() == 0:
                event_path_timer().synced_record("video_generation_start")
                print('Testing inference time cost.')
                
            videos = torch.cat(
                [
                    post_chunk_process(chunk, self.config)
                    for chunk in generate_per_chunk(
                        model=dit, prompt=prompt, prefix_video=prefix_video, caption_embs=caption_embs, emb_masks=emb_masks
                    )
                ],
                dim=0,
            )
            
            # End timing if rank 0
            if dist.get_rank() == 0:
                event_path_timer().synced_record("video_generation_end")
                print('End of testing inference time cost.')
        
        save_video_to_disk(videos, output_path, fps=self.config.runtime_config.fps)

        mem_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
        mem_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
        print_rank_0(
            f"Finish MagiPipeline, max memory allocated: {mem_allocated_gb:.2f} GB, max memory reserved: {mem_reserved_gb:.2f} GB"
        )

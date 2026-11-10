
import os
import os.path as osp
from typing import Optional

import cv2
import numpy as np
from mmengine.visualization import (LocalVisBackend, TensorboardVisBackend,
                                    WandbVisBackend)
from mmengine.visualization.vis_backend import force_init_env

from mmaction.registry import VISBACKENDS

try:
    import wandb
except ImportError:
    pass


@VISBACKENDS.register_module()
class LocalVisBackend(LocalVisBackend):
    

    @force_init_env
    def add_video(self,
                  name: str,
                  frames: np.ndarray,
                  step: int = 0,
                  fps: Optional[int] = 4,
                  out_type: Optional[int] = 'img',
                  **kwargs) -> None:
        
        assert frames.dtype == np.uint8

        if out_type == 'img':
            frames_dir = osp.join(self._save_dir, name, f'frames_{step}')
            os.makedirs(frames_dir, exist_ok=True)
            for idx, frame in enumerate(frames):
                drawn_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                save_file_name = f'{idx}.png'
                cv2.imwrite(osp.join(frames_dir, save_file_name), drawn_image)
        else:
            try:
                from moviepy.editor import ImageSequenceClip
            except ImportError:
                raise ImportError('Please install moviepy to enable '
                                  'output file.')

            frames = [x[..., ::-1] for x in frames]
            video_clips = ImageSequenceClip(frames, fps=fps)
            name = osp.splitext(name)[0]
            if out_type == 'gif':
                out_path = osp.join(self._save_dir, name + '.gif')
                video_clips.write_gif(out_path, logger=None)
            elif out_type == 'video':
                out_path = osp.join(self._save_dir, name + '.mp4')
                video_clips.write_videofile(
                    out_path, remove_temp=True, logger=None)


@VISBACKENDS.register_module()
class WandbVisBackend(WandbVisBackend):
    

    @force_init_env
    def add_video(self,
                  name: str,
                  frames: np.ndarray,
                  fps: int = 4,
                  **kwargs) -> None:
        
        frames = frames.transpose(0, 3, 1, 2)
        self._wandb.log({'video': wandb.Video(frames, fps=fps, format='gif')})


@VISBACKENDS.register_module()
class TensorboardVisBackend(TensorboardVisBackend):
    

    @force_init_env
    def add_video(self,
                  name: str,
                  frames: np.ndarray,
                  step: int = 0,
                  fps: int = 4,
                  **kwargs) -> None:
        
        frames = frames.transpose(0, 3, 1, 2)
        frames = frames.reshape(1, *frames.shape)
        self._tensorboard.add_video(name, frames, global_step=step, fps=fps)

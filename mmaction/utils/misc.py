
import ctypes
import os
import os.path as osp
import random
import string
from typing import Optional

import cv2
import mmcv
import numpy as np


def get_random_string(length: int = 15) -> str:
    
    return ''.join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length))


def get_thread_id() -> int:
    

    thread_id = ctypes.CDLL('libc.so.6').syscall(186)
    return thread_id


def get_shm_dir() -> str:
    
    return '/dev/shm'


def frame_extract(video_path: str,
                  short_side: Optional[int] = None,
                  out_dir: str = './tmp'):
    

    target_dir = osp.join(out_dir, osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)

    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    assert osp.exists(video_path), f'file not exit {video_path}'
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if short_side is not None:
            if new_h is None:
                h, w, _ = frame.shape
                new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
            frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames

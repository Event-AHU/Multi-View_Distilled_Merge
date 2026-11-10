
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmengine.dist import master_only
from mmengine.fileio.io import isdir, isfile, join_path, list_dir_or_file
from mmengine.visualization import Visualizer

from mmaction.registry import VISBACKENDS, VISUALIZERS
from mmaction.structures import ActionDataSample


def _get_adaptive_scale(img_shape: Tuple[int, int],
                        min_scale: float = 0.3,
                        max_scale: float = 3.0) -> float:
    
    short_edge_length = min(img_shape)
    scale = short_edge_length / 224.
    return min(max(scale, min_scale), max_scale)


@VISUALIZERS.register_module()
class ActionVisualizer(Visualizer):
    

    def __init__(
        self,
        name='visualizer',
        vis_backends: Optional[List[Dict]] = None,
        save_dir: Optional[str] = None,
        fig_save_cfg=dict(frameon=False),
        fig_show_cfg=dict(frameon=False)
    ) -> None:
        super().__init__(
            name=name,
            image=None,
            vis_backends=vis_backends,
            save_dir=save_dir,
            fig_save_cfg=fig_save_cfg,
            fig_show_cfg=fig_show_cfg)

    def _load_video(self,
                    video: Union[np.ndarray, Sequence[np.ndarray], str],
                    target_resolution: Optional[Tuple[int]] = None):
        
        if isinstance(video, np.ndarray) or isinstance(video, list):
            frames = video
        elif isinstance(video, str):

            if isfile(video):
                try:
                    import decord
                except ImportError:
                    raise ImportError(
                        'Please install decord to load video file.')
                video = decord.VideoReader(video)
                frames = [x.asnumpy()[..., ::-1] for x in video]

            elif isdir(video):
                frame_list = sorted(list_dir_or_file(video, list_dir=False))
                frames = [mmcv.imread(join_path(video, x)) for x in frame_list]
        else:
            raise TypeError(f'type of video {type(video)} not supported')

        if target_resolution is not None:
            w, h = target_resolution
            frame_h, frame_w, _ = frames[0].shape
            if w == -1:
                w = int(h / frame_h * frame_w)
            if h == -1:
                h = int(w / frame_w * frame_h)
            frames = [mmcv.imresize(f, (w, h)) for f in frames]

        return frames

    @master_only
    def add_datasample(self,
                       name: str,
                       video: Union[np.ndarray, Sequence[np.ndarray], str],
                       data_sample: Optional[ActionDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       draw_score: bool = True,
                       rescale_factor: Optional[float] = None,
                       show_frames: bool = False,
                       text_cfg: dict = dict(),
                       wait_time: float = 0.1,
                       out_path: Optional[str] = None,
                       out_type: str = 'img',
                       target_resolution: Optional[Tuple[int]] = None,
                       step: int = 0,
                       fps: int = 4) -> None:
        
        classes = None
        video = self._load_video(video, target_resolution)
        tol_video = len(video)

        if self.dataset_meta is not None:
            classes = self.dataset_meta.get('classes', None)

        if data_sample is None:
            data_sample = ActionDataSample()

        resulted_video = []
        for frame_idx, frame in enumerate(video):
            frame_name = 'frame %d of %s' % (frame_idx + 1, name)
            if rescale_factor is not None:
                frame = mmcv.imrescale(frame, rescale_factor)

            texts = ['Frame %d of total %d frames' % (frame_idx, tol_video)]
            self.set_image(frame)

            if draw_gt and 'gt_labels' in data_sample:
                gt_labels = data_sample.gt_labels
                idx = gt_labels.item.tolist()
                class_labels = [''] * len(idx)
                if classes is not None:
                    class_labels = [f' ({classes[i]})' for i in idx]
                labels = [
                    str(idx[i]) + class_labels[i] for i in range(len(idx))
                ]
                prefix = 'Ground truth: '
                texts.append(prefix + ('\n' + ' ' * len(prefix)).join(labels))

            if draw_pred and 'pred_labels' in data_sample:
                pred_labels = data_sample.pred_labels
                idx = pred_labels.item.tolist()
                score_labels = [''] * len(idx)
                class_labels = [''] * len(idx)
                if draw_score and 'score' in pred_labels:
                    score_labels = [
                        f', {pred_labels.score[i].item():.2f}' for i in idx
                    ]

                if classes is not None:
                    class_labels = [f' ({classes[i]})' for i in idx]

                labels = [
                    str(idx[i]) + score_labels[i] + class_labels[i]
                    for i in range(len(idx))
                ]
                prefix = 'Prediction: '
                texts.append(prefix + ('\n' + ' ' * len(prefix)).join(labels))

            img_scale = _get_adaptive_scale(frame.shape[:2])
            _text_cfg = {
                'positions':
                np.array([(img_scale * 5, ) * 2]).astype(np.int32),
                'font_sizes': int(img_scale * 7),
                'font_families': 'monospace',
                'colors': 'white',
                'bboxes': dict(facecolor='black', alpha=0.5, boxstyle='Round'),
            }
            _text_cfg.update(text_cfg)
            self.draw_texts('\n'.join(texts), **_text_cfg)
            drawn_img = self.get_image()
            resulted_video.append(drawn_img)

        if show_frames:
            frame_wait_time = 1. / fps
            for frame_idx, drawn_img in enumerate(resulted_video):
                frame_name = 'frame %d of %s' % (frame_idx + 1, name)
                if frame_idx < len(resulted_video) - 1:
                    wait_time = frame_wait_time
                else:
                    wait_time = wait_time
                self.show(
                    drawn_img[:, :, ::-1],
                    win_name=frame_name,
                    wait_time=wait_time)

        resulted_video = np.array(resulted_video)
        if out_path is not None:
            save_dir, save_name = osp.split(out_path)
            vis_backend_cfg = dict(type='LocalVisBackend', save_dir=save_dir)
            tmp_local_vis_backend = VISBACKENDS.build(vis_backend_cfg)
            tmp_local_vis_backend.add_video(
                save_name,
                resulted_video,
                step=step,
                fps=fps,
                out_type=out_type)
        else:
            self.add_video(
                name, resulted_video, step=step, fps=fps, out_type=out_type)
        return resulted_video

    @master_only
    def add_video(
        self,
        name: str,
        image: np.ndarray,
        step: int = 0,
        fps: int = 4,
        out_type: str = 'img',
    ) -> None:
        
        for vis_backend in self._vis_backends.values():
            vis_backend.add_video(
                name, image, step=step, fps=fps,
                out_type=out_type)

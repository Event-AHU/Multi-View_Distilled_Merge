
from pathlib import Path
from typing import List, Optional, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import track_iter_progress

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample


def init_recognizer(config: Union[str, Path, mmengine.Config],
                    checkpoint: Optional[str] = None,
                    device: Union[str, torch.device] = 'cuda:0') -> nn.Module:
    
    if isinstance(config, (str, Path)):
        config = mmengine.Config.fromfile(config)
    elif not isinstance(config, mmengine.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    init_default_scope(config.get('default_scope', 'mmaction'))

    if config.model.backbone.get('pretrained', None):
        config.model.backbone.pretrained = None
    model = MODELS.build(config.model)

    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def inference_recognizer(model: nn.Module,
                         video: Union[str, dict],
                         test_pipeline: Optional[Compose] = None
                         ) -> ActionDataSample:
    

    if test_pipeline is None:
        cfg = model.cfg
        test_pipeline_cfg = cfg.test_pipeline
        test_pipeline = Compose(test_pipeline_cfg)

    input_flag = None
    if isinstance(video, dict):
        input_flag = 'dict'
    elif isinstance(video, str):
        input_flag = 'video'
    else:
        raise RuntimeError(f'The type of argument `video` is not supported: '
                           f'{type(video)}')

    if input_flag == 'dict':
        data = video
    if input_flag == 'video':
        data = dict(filename=video, label=-1, start_index=0, modality='RGB')

    data = test_pipeline(data)
    data = pseudo_collate([data])


    with torch.no_grad():
        result = model.test_step(data)[0]

    return result


def detection_inference(det_config: Union[str, Path, mmengine.Config],
                        det_checkpoint: str,
                        frame_paths: List[str],
                        det_score_thr: float = 0.9,
                        det_cat_id: int = 0,
                        device: Union[str, torch.device] = 'cuda:0',
                        with_score: bool = False) -> tuple:
    
    try:
        from mmdet.apis import inference_detector, init_detector
        from mmdet.structures import DetDataSample
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import `inference_detector` and '
                          '`init_detector` from `mmdet.apis`. These apis are '
                          'required in this inference api! ')

    model = init_detector(
        config=det_config, checkpoint=det_checkpoint, device=device)

    results = []
    data_samples = []
    print('Performing Human Detection for each frame')
    for frame_path in track_iter_progress(frame_paths):
        det_data_sample: DetDataSample = inference_detector(model, frame_path)
        pred_instance = det_data_sample.pred_instances.cpu().numpy()
        bboxes = pred_instance.bboxes
        scores = pred_instance.scores


        valid_idx = np.logical_and(pred_instance.labels == det_cat_id,
                                   pred_instance.scores > det_score_thr)
        bboxes = bboxes[valid_idx]
        scores = scores[valid_idx]

        if with_score:
            bboxes = np.concatenate((bboxes, scores[:, None]), axis=-1)
        results.append(bboxes)
        data_samples.append(det_data_sample)

    return results, data_samples


def pose_inference(pose_config: Union[str, Path, mmengine.Config],
                   pose_checkpoint: str,
                   frame_paths: List[str],
                   det_results: List[np.ndarray],
                   device: Union[str, torch.device] = 'cuda:0') -> tuple:
    
    try:
        from mmpose.apis import inference_topdown, init_model
        from mmpose.structures import PoseDataSample, merge_data_samples
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import `inference_topdown` and '
                          '`init_model` from `mmpose.apis`. These apis '
                          'are required in this inference api! ')

    model = init_model(pose_config, pose_checkpoint, device)

    results = []
    data_samples = []
    print('Performing Human Pose Estimation for each frame')
    for f, d in track_iter_progress(list(zip(frame_paths, det_results))):
        pose_data_samples: List[PoseDataSample]\
            = inference_topdown(model, f, d[..., :4], bbox_format='xyxy')
        pose_data_sample = merge_data_samples(pose_data_samples)
        pose_data_sample.dataset_meta = model.dataset_meta
        poses = pose_data_sample.pred_instances.to_dict()
        results.append(poses)
        data_samples.append(pose_data_sample)

    return results, data_samples


from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
from mmengine.dataset import COLLATE_FUNCTIONS, pseudo_collate

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .video_dataset import VideoDataset


def get_type(transform: Union[dict, Callable]) -> str:
    
    if isinstance(transform, dict) and 'type' in transform:
        return transform['type']
    elif callable(transform):
        return transform.__repr__().split('(')[0]
    else:
        raise TypeError


@DATASETS.register_module()
class RepeatAugDataset(VideoDataset):
    

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]],
                 data_prefix: ConfigType = dict(video=''),
                 num_repeats: int = 4,
                 sample_once: bool = False,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 0,
                 modality: str = 'RGB',
                 **kwargs) -> None:

        use_decord = get_type(pipeline[0]) == 'DecordInit' and\
               get_type(pipeline[2]) == 'DecordDecode'

        assert use_decord, (
            'RepeatAugDataset requires decord as the video '
            'loading backend, will support more backends in the '
            'future')

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            test_mode=False,
            **kwargs)
        self.num_repeats = num_repeats
        self.sample_once = sample_once

    def prepare_data(self, idx) -> List[dict]:
        
        transforms = self.pipeline.transforms

        data_info = self.get_data_info(idx)
        data_info = transforms[0](data_info)

        frame_inds_list, frame_inds_length = [], [0]

        fake_data_info = dict(
            total_frames=data_info['total_frames'],
            start_index=data_info['start_index'])

        if not self.sample_once:
            for repeat in range(self.num_repeats):
                data_info_ = transforms[1](fake_data_info)
                frame_inds = data_info_['frame_inds']
                frame_inds_list.append(frame_inds.reshape(-1))
                frame_inds_length.append(frame_inds.size +
                                         frame_inds_length[-1])
        else:
            data_info_ = transforms[1](fake_data_info)
            frame_inds = data_info_['frame_inds']
            for repeat in range(self.num_repeats):
                frame_inds_list.append(frame_inds.reshape(-1))
                frame_inds_length.append(frame_inds.size +
                                         frame_inds_length[-1])

        for key in data_info_:
            data_info[key] = data_info_[key]

        data_info['frame_inds'] = np.concatenate(frame_inds_list)

        data_info = transforms[2](data_info)
        imgs = data_info.pop('imgs')

        data_info_list = []
        for repeat in range(self.num_repeats):
            data_info_ = deepcopy(data_info)
            start = frame_inds_length[repeat]
            end = frame_inds_length[repeat + 1]
            data_info_['imgs'] = imgs[start:end]
            for transform in transforms[3:]:
                data_info_ = transform(data_info_)
            data_info_list.append(data_info_)
        del imgs
        return data_info_list


@COLLATE_FUNCTIONS.register_module()
def repeat_pseudo_collate(data_batch: Sequence) -> Any:
    data_batch = [i for j in data_batch for i in j]
    return pseudo_collate(data_batch)

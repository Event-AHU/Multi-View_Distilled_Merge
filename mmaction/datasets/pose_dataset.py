import os.path as osp
from typing import Callable, Dict, List, Optional, Union

import mmengine

from mmaction.registry import DATASETS
from .base import BaseActionDataset


@DATASETS.register_module()
class PoseDataset(BaseActionDataset):
    

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[Dict, Callable]],
                 split: Optional[str] = None,
                 **kwargs) -> None:
        self.split = split
        super().__init__(
            ann_file, pipeline=pipeline, modality='Pose', **kwargs)

    def load_data_list(self) -> List[Dict]:
        
        assert self.ann_file.endswith('.pkl')
        mmengine.exists(self.ann_file)
        data_list = mmengine.load(self.ann_file)

        if self.split is not None:
            split, annos = data_list['split'], data_list['annotations']
            identifier = 'filename' if 'filename' in annos[0] else 'frame_dir'
            split = set(split[self.split])
            data_list = [x for x in annos if x[identifier] in split]


        if 'video' in self.data_prefix:
            for item in data_list:
                if 'filename' in item:
                    item['filename'] = osp.join(self.data_prefix['video'],
                                                item['filename'])
                if 'frame_dir' in item:
                    item['frame_dir'] = osp.join(self.data_prefix['video'],
                                                 item['frame_dir'])
        return data_list

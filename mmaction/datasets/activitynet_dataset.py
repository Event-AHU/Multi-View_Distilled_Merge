
from typing import Callable, List, Optional, Union

import mmengine
from mmengine.fileio import exists

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class ActivityNetDataset(BaseActionDataset):
    

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]],
                 data_prefix: Optional[ConfigType] = dict(video=''),
                 test_mode: bool = False,
                 **kwargs):

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        
        exists(self.ann_file)
        data_list = []
        anno_database = mmengine.load(self.ann_file)
        for video_name in anno_database:
            video_info = anno_database[video_name]
            feature_path = video_name + '.csv'
            feature_path = '%s/%s' % (self.data_prefix['video'], feature_path)
            video_info['feature_path'] = feature_path
            video_info['video_name'] = video_name
            data_list.append(video_info)
        return data_list



from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData, LabelData

from mmaction.structures import ActionDataSample


ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]

MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

InstanceList = List[InstanceData]
OptInstanceList = Optional[InstanceList]

LabelList = List[LabelData]
OptLabelList = Optional[LabelList]

SampleList = List[ActionDataSample]
OptSampleList = Optional[SampleList]

ForwardResults = Union[Dict[str, torch.Tensor], List[ActionDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


class SamplingResult:
    

    def __init__(self, *args, **kwargs):
        pass

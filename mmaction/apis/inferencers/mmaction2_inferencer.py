
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
from mmengine.infer import BaseInferencer
from mmengine.structures import InstanceData

from mmaction.utils import ConfigType
from .actionrecog_inferencer import ActionRecogInferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


class MMAction2Inferencer(BaseInferencer):
    

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = {
        'return_vis', 'show', 'wait_time', 'vid_out_dir', 'draw_pred', 'fps',
        'out_type', 'target_resolution'
    }
    postprocess_kwargs: set = {
        'print_result', 'pred_out_file', 'return_datasample'
    }

    def __init__(self,
                 rec: Optional[str] = None,
                 rec_weights: Optional[str] = None,
                 device: Optional[str] = None,
                 label_file: Optional[str] = None,
                 input_format: str = 'video') -> None:

        if rec is None:
            raise ValueError('rec algorithm should provided.')

        self.visualizer = None
        self.num_visualized_imgs = 0

        if rec is not None:
            self.actionrecog_inferencer = ActionRecogInferencer(
                rec, rec_weights, device, label_file, input_format)
            self.mode = 'rec'

    def _init_pipeline(self, cfg: ConfigType) -> None:
        pass

    def forward(self, inputs: InputType, batch_size: int,
                **forward_kwargs) -> PredType:
        
        result = {}
        if self.mode == 'rec':
            predictions = self.actionrecog_inferencer(
                inputs,
                return_datasamples=True,
                batch_size=batch_size,
                **forward_kwargs)['predictions']
            result['rec'] = [[p] for p in predictions]

        return result

    def visualize(self, inputs: InputsType, preds: PredType,
                  **kwargs) -> List[np.ndarray]:
        

        if 'rec' in self.mode:
            return self.actionrecog_inferencer.visualize(
                inputs, preds['rec'][0], **kwargs)

    def __call__(
        self,
        inputs: InputsType,
        batch_size: int = 1,
        **kwargs,
    ) -> dict:
        
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self._inputs_to_list(inputs)

        preds = self.forward(ori_inputs, batch_size, **forward_kwargs)

        visualization = self.visualize(
            ori_inputs, preds,
            **visualize_kwargs)
        results = self.postprocess(preds, visualization, **postprocess_kwargs)
        return results

    def _inputs_to_list(self, inputs: InputsType) -> list:
        
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def postprocess(self,
                    preds: PredType,
                    visualization: Optional[List[np.ndarray]] = None,
                    print_result: bool = False,
                    pred_out_file: str = ''
                    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        

        result_dict = {}
        pred_results = [{} for _ in range(len(next(iter(preds.values()))))]
        if 'rec' in self.mode:
            for i, rec_pred in enumerate(preds['rec']):
                result = dict(rec_labels=[], rec_scores=[])
                for rec_pred_instance in rec_pred:
                    rec_dict_res = self.actionrecog_inferencer.pred2dict(
                        rec_pred_instance)
                    result['rec_labels'].append(rec_dict_res['pred_labels'])
                    result['rec_scores'].append(rec_dict_res['pred_scores'])
                pred_results[i].update(result)

        result_dict['predictions'] = pred_results
        if print_result:
            print(result_dict)
        if pred_out_file != '':
            mmengine.dump(result_dict, pred_out_file)
        result_dict['visualization'] = visualization
        return result_dict

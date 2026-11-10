
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    

    def __init__(self,
                 model: nn.Module,
                 target_layer_name: str,
                 colormap: str = 'viridis') -> None:
        from ..models.recognizers import Recognizer2D, Recognizer3D
        if isinstance(model, Recognizer2D):
            self.is_recognizer2d = True
        elif isinstance(model, Recognizer3D):
            self.is_recognizer2d = False
        else:
            raise ValueError(
                'GradCAM utils only support Recognizer2D & Recognizer3D.')

        self.model = model
        self.model.eval()
        self.target_gradients = None
        self.target_activations = None

        import matplotlib.pyplot as plt
        self.colormap = plt.get_cmap(colormap)
        self._register_hooks(target_layer_name)

    def _register_hooks(self, layer_name: str) -> None:
        

        def get_gradients(module, grad_input, grad_output):
            self.target_gradients = grad_output[0].detach()

        def get_activations(module, input, output):
            self.target_activations = output.clone().detach()

        layer_ls = layer_name.split('/')
        prev_module = self.model
        for layer in layer_ls:
            prev_module = prev_module._modules[layer]

        target_layer = prev_module
        target_layer.register_forward_hook(get_activations)
        target_layer.register_backward_hook(get_gradients)

    def _calculate_localization_map(self,
                                    data: dict,
                                    use_labels: bool,
                                    delta=1e-20) -> tuple:
        
        inputs = data['inputs']


        self.model.cls_head.average_clips = 'score'

        results = self.model.test_step(data)
        preds = [result.pred_scores.item for result in results]
        preds = torch.stack(preds)

        if use_labels:
            labels = [result.gt_labels.item for result in results]
            labels = torch.stack(labels)
            score = torch.gather(preds, dim=1, index=labels)
        else:
            score = torch.max(preds, dim=-1)[0]
        self.model.zero_grad()
        score = torch.sum(score)
        score.backward()

        imgs = torch.stack(inputs)
        if self.is_recognizer2d:

            b, t, _, h, w = imgs.size()
        else:

            b1, b2, _, t, h, w = imgs.size()
            b = b1 * b2

        gradients = self.target_gradients
        activations = self.target_activations
        if self.is_recognizer2d:

            b_tg, c, _, _ = gradients.size()
            tg = b_tg // b
        else:

            _, c, tg, _, _ = gradients.size()

            gradients = gradients.permute(0, 2, 1, 3, 4)
            activations = activations.permute(0, 2, 1, 3, 4)


        weights = torch.mean(gradients.view(b, tg, c, -1), dim=3)
        weights = weights.view(b, tg, c, 1, 1)
        activations = activations.view([b, tg, c] +
                                       list(activations.size()[-2:]))
        localization_map = torch.sum(
            weights * activations, dim=2, keepdim=True)
        localization_map = F.relu(localization_map)
        localization_map = localization_map.permute(0, 2, 1, 3, 4)
        localization_map = F.interpolate(
            localization_map,
            size=(t, h, w),
            mode='trilinear',
            align_corners=False)


        localization_map_min, localization_map_max = (
            torch.min(localization_map.view(b, -1), dim=-1, keepdim=True)[0],
            torch.max(localization_map.view(b, -1), dim=-1, keepdim=True)[0])
        localization_map_min = torch.reshape(
            localization_map_min, shape=(b, 1, 1, 1, 1))
        localization_map_max = torch.reshape(
            localization_map_max, shape=(b, 1, 1, 1, 1))
        localization_map = (localization_map - localization_map_min) / (
            localization_map_max - localization_map_min + delta)
        localization_map = localization_map.data

        return localization_map.squeeze(dim=1), preds

    def _alpha_blending(self, localization_map: torch.Tensor,
                        input_imgs: torch.Tensor,
                        alpha: float) -> torch.Tensor:
        

        localization_map = localization_map.cpu()


        heatmap = self.colormap(localization_map.detach().numpy())
        heatmap = heatmap[..., :3]
        heatmap = torch.from_numpy(heatmap)
        input_imgs = torch.stack(input_imgs)

        if self.is_recognizer2d:

            curr_inp = input_imgs.permute(0, 1, 3, 4, 2)
        else:


            curr_inp = input_imgs.view([-1] + list(input_imgs.size()[2:]))
            curr_inp = curr_inp.permute(0, 2, 3, 4, 1)


        curr_inp = curr_inp.cpu().float()
        curr_inp /= 255.


        blended_imgs = alpha * heatmap + (1 - alpha) * curr_inp

        return blended_imgs

    def __call__(self,
                 data: dict,
                 use_labels: bool = False,
                 alpha: float = 0.5) -> tuple:
        



        localization_map, preds = self._calculate_localization_map(
            data, use_labels=use_labels)


        blended_imgs = self._alpha_blending(localization_map, data['inputs'],
                                            alpha)





        return blended_imgs, preds



import copy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger
from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.runner import load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmaction.registry import MODELS
from .lfb import LFB

try:
    from mmdet.registry import MODELS as MMDET_MODELS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


class NonLocalLayer(nn.Module):
    

    def __init__(self,
                 st_feat_channels,
                 lt_feat_channels,
                 latent_channels,
                 num_st_feat,
                 num_lt_feat,
                 use_scale=True,
                 pre_activate=True,
                 pre_activate_with_ln=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 dropout_ratio=0.2,
                 zero_init_out_conv=False):
        super().__init__()
        if conv_cfg is None:
            conv_cfg = dict(type='Conv3d')
        self.st_feat_channels = st_feat_channels
        self.lt_feat_channels = lt_feat_channels
        self.latent_channels = latent_channels
        self.num_st_feat = num_st_feat
        self.num_lt_feat = num_lt_feat
        self.use_scale = use_scale
        self.pre_activate = pre_activate
        self.pre_activate_with_ln = pre_activate_with_ln
        self.dropout_ratio = dropout_ratio
        self.zero_init_out_conv = zero_init_out_conv

        self.st_feat_conv = ConvModule(
            self.st_feat_channels,
            self.latent_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.lt_feat_conv = ConvModule(
            self.lt_feat_channels,
            self.latent_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.global_conv = ConvModule(
            self.lt_feat_channels,
            self.latent_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        if pre_activate:
            self.ln = nn.LayerNorm([latent_channels, num_st_feat, 1, 1])
        else:
            self.ln = nn.LayerNorm([st_feat_channels, num_st_feat, 1, 1])

        self.relu = nn.ReLU()

        self.out_conv = ConvModule(
            self.latent_channels,
            self.st_feat_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)

    def init_weights(self, pretrained=None):
        
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {pretrained}')
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
            if self.zero_init_out_conv:
                constant_init(self.out_conv, 0, bias=0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, st_feat, lt_feat):
        
        n, c = st_feat.size(0), self.latent_channels
        num_st_feat, num_lt_feat = self.num_st_feat, self.num_lt_feat

        theta = self.st_feat_conv(st_feat)
        theta = theta.view(n, c, num_st_feat)

        phi = self.lt_feat_conv(lt_feat)
        phi = phi.view(n, c, num_lt_feat)

        g = self.global_conv(lt_feat)
        g = g.view(n, c, num_lt_feat)



        theta_phi = torch.matmul(theta.permute(0, 2, 1), phi)
        if self.use_scale:
            theta_phi /= c**0.5

        p = theta_phi.softmax(dim=-1)



        out = torch.matmul(g, p.permute(0, 2, 1)).view(n, c, num_st_feat, 1, 1)



        if self.pre_activate:
            if self.pre_activate_with_ln:
                out = self.ln(out)
            out = self.relu(out)

        out = self.out_conv(out)

        if not self.pre_activate:
            out = self.ln(out)
        if self.dropout_ratio > 0:
            out = self.dropout(out)

        return out


class FBONonLocal(nn.Module):
    

    def __init__(self,
                 st_feat_channels,
                 lt_feat_channels,
                 latent_channels,
                 num_st_feat,
                 num_lt_feat,
                 num_non_local_layers=2,
                 st_feat_dropout_ratio=0.2,
                 lt_feat_dropout_ratio=0.2,
                 pre_activate=True,
                 zero_init_out_conv=False,
                 **kwargs):
        super().__init__()
        assert num_non_local_layers >= 1, (
            'At least one non_local_layer is needed.')
        self.st_feat_channels = st_feat_channels
        self.lt_feat_channels = lt_feat_channels
        self.latent_channels = latent_channels
        self.num_st_feat = num_st_feat
        self.num_lt_feat = num_lt_feat
        self.num_non_local_layers = num_non_local_layers
        self.st_feat_dropout_ratio = st_feat_dropout_ratio
        self.lt_feat_dropout_ratio = lt_feat_dropout_ratio
        self.pre_activate = pre_activate
        self.zero_init_out_conv = zero_init_out_conv

        self.st_feat_conv = nn.Conv3d(
            st_feat_channels, latent_channels, kernel_size=1)
        self.lt_feat_conv = nn.Conv3d(
            lt_feat_channels, latent_channels, kernel_size=1)

        if self.st_feat_dropout_ratio > 0:
            self.st_feat_dropout = nn.Dropout(self.st_feat_dropout_ratio)

        if self.lt_feat_dropout_ratio > 0:
            self.lt_feat_dropout = nn.Dropout(self.lt_feat_dropout_ratio)

        if not self.pre_activate:
            self.relu = nn.ReLU()

        self.non_local_layers = []
        for idx in range(self.num_non_local_layers):
            layer_name = f'non_local_layer_{idx + 1}'
            self.add_module(
                layer_name,
                NonLocalLayer(
                    latent_channels,
                    latent_channels,
                    latent_channels,
                    num_st_feat,
                    num_lt_feat,
                    pre_activate=self.pre_activate,
                    zero_init_out_conv=self.zero_init_out_conv))
            self.non_local_layers.append(layer_name)

    def init_weights(self, pretrained=None):
        
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            kaiming_init(self.st_feat_conv)
            kaiming_init(self.lt_feat_conv)
            for layer_name in self.non_local_layers:
                non_local_layer = getattr(self, layer_name)
                non_local_layer.init_weights(pretrained=pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, st_feat, lt_feat):
        

        st_feat = self.st_feat_conv(st_feat)
        if self.st_feat_dropout_ratio > 0:
            st_feat = self.st_feat_dropout(st_feat)


        lt_feat = self.lt_feat_conv(lt_feat)
        if self.lt_feat_dropout_ratio > 0:
            lt_feat = self.lt_feat_dropout(lt_feat)


        for layer_name in self.non_local_layers:
            identity = st_feat
            non_local_layer = getattr(self, layer_name)
            nl_out = non_local_layer(st_feat, lt_feat)
            nl_out = identity + nl_out
            if not self.pre_activate:
                nl_out = self.relu(nl_out)
            st_feat = nl_out

        return nl_out


class FBOAvg(nn.Module):
    

    def __init__(self, **kwargs):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, None, None))

    def init_weights(self, pretrained=None):

        pass

    def forward(self, st_feat, lt_feat):
        out = self.avg_pool(lt_feat)
        return out


class FBOMax(nn.Module):
    

    def __init__(self, **kwargs):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool3d((1, None, None))

    def init_weights(self, pretrained=None):
        
        pass

    def forward(self, st_feat, lt_feat):
        
        out = self.max_pool(lt_feat)
        return out


@MODELS.register_module()
class FBOHead(nn.Module):
    

    fbo_dict = {'non_local': FBONonLocal, 'avg': FBOAvg, 'max': FBOMax}

    def __init__(self,
                 lfb_cfg,
                 fbo_cfg,
                 temporal_pool_type='avg',
                 spatial_pool_type='max'):
        super().__init__()
        fbo_type = fbo_cfg.pop('type', 'non_local')
        assert fbo_type in FBOHead.fbo_dict
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']

        self.lfb_cfg = copy.deepcopy(lfb_cfg)
        self.fbo_cfg = copy.deepcopy(fbo_cfg)

        self.lfb = LFB(**self.lfb_cfg)
        self.fbo = self.fbo_dict[fbo_type](**self.fbo_cfg)


        if temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

    def init_weights(self, pretrained=None):
        
        self.fbo.init_weights(pretrained=pretrained)

    def sample_lfb(self, rois, img_metas):
        
        inds = rois[:, 0].type(torch.int64)
        lt_feat_list = []
        for ind in inds:
            lt_feat_list.append(self.lfb[img_metas[ind]['img_key']])
        lt_feat = torch.stack(lt_feat_list, dim=0)

        lt_feat = lt_feat.permute(0, 2, 1).contiguous()
        return lt_feat.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x, rois, img_metas, **kwargs):
        

        st_feat = self.temporal_pool(x)
        st_feat = self.spatial_pool(st_feat)
        identity = st_feat


        lt_feat = self.sample_lfb(rois, img_metas).to(st_feat.device)

        fbo_feat = self.fbo(st_feat, lt_feat)

        out = torch.cat([identity, fbo_feat], dim=1)
        return out


if mmdet_imported:
    MMDET_MODELS.register_module()(FBOHead)

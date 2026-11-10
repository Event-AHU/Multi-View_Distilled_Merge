
import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init
from mmengine.utils import digit_version

from mmaction.registry import MODELS


@MODELS.register_module()
class DividedTemporalAttentionWithNorm(BaseModule):
    

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.1),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

        if digit_version(torch.__version__) < digit_version('1.9.0'):
            kwargs.pop('batch_first', None)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.temporal_fc = nn.Linear(self.embed_dims, self.embed_dims)

        self.init_weights()

    def init_weights(self):
        
        constant_init(self.temporal_fc, val=0, bias=0)

    def forward(self, query, key=None, value=None, residual=None, **kwargs):
        
        assert residual is None, (
            'Always adding the shortcut in the forward function')

        init_cls_token = query[:, 0, :].unsqueeze(1)
        identity = query_t = query[:, 1:, :]


        b, pt, m = query_t.size()
        p, t = pt // self.num_frames, self.num_frames


        query_t = self.norm(query_t.reshape(b * p, t, m)).permute(1, 0, 2)
        res_temporal = self.attn(query_t, query_t, query_t)[0].permute(1, 0, 2)
        res_temporal = self.dropout_layer(
            self.proj_drop(res_temporal.contiguous()))
        res_temporal = self.temporal_fc(res_temporal)


        res_temporal = res_temporal.reshape(b, p * t, m)


        new_query_t = identity + res_temporal
        new_query = torch.cat((init_cls_token, new_query_t), 1)
        return new_query


@MODELS.register_module()
class DividedSpatialAttentionWithNorm(BaseModule):
    

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.1),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]
        if digit_version(torch.__version__) < digit_version('1.9.0'):
            kwargs.pop('batch_first', None)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

        self.init_weights()

    def init_weights(self):
        
        pass

    def forward(self, query, key=None, value=None, residual=None, **kwargs):
        
        assert residual is None, (
            'Always adding the shortcut in the forward function')

        identity = query
        init_cls_token = query[:, 0, :].unsqueeze(1)
        query_s = query[:, 1:, :]


        b, pt, m = query_s.size()
        p, t = pt // self.num_frames, self.num_frames


        cls_token = init_cls_token.repeat(1, t, 1).reshape(b * t,
                                                           m).unsqueeze(1)


        query_s = rearrange(query_s, 'b (p t) m -> (b t) p m', p=p, t=t)
        query_s = torch.cat((cls_token, query_s), 1)


        query_s = self.norm(query_s).permute(1, 0, 2)
        res_spatial = self.attn(query_s, query_s, query_s)[0].permute(1, 0, 2)
        res_spatial = self.dropout_layer(
            self.proj_drop(res_spatial.contiguous()))


        cls_token = res_spatial[:, 0, :].reshape(b, t, m)
        cls_token = torch.mean(cls_token, 1, True)


        res_spatial = rearrange(
            res_spatial[:, 1:, :], '(b t) p m -> b (p t) m', p=p, t=t)
        res_spatial = torch.cat((cls_token, res_spatial), 1)

        new_query = identity + res_spatial
        return new_query


@MODELS.register_module()
class FFNWithNorm(FFN):
    

    def __init__(self, *args, norm_cfg=dict(type='LN'), **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

    def forward(self, x, residual=None):
        
        assert residual is None, ('Cannot apply pre-norm with FFNWithNorm')
        return super().forward(self.norm(x), x)

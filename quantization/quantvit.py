""" Vision Transformer (ViT) in Pypaddle

A Pypaddle implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old Pypaddle weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pypaddle ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import paddle
import paddle.nn as nn
import warnings
from functools import partial
import numpy as np
from .lsq_layer import  QuantLinear, QuantAct, QuantConv2d
# paddle.disable_static()

class Mlp(nn.Layer):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.,
                 wbits=-1, 
                 abits=-1, 
                 offset=False, 
                 learned=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.quant1 = QuantAct(nbits=abits, offset=offset, learned=learned)
        self.fc1 = QuantLinear(in_features, hidden_features, nbits=wbits, learned=learned)
        self.act = act_layer(inplace=True) if isinstance(act_layer, nn.ReLU) else act_layer()
        self.quant2 = QuantAct(nbits=abits, signed=False, offset=True, learned=learned)
        self.fc2 = QuantLinear(hidden_features, out_features, nbits=wbits, learned=learned)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.quant1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.quant2(x)
        x = self.fc2(x)
        x = self.drop(x)
        # print(self.quant2.alpha[:4], self.quant2.beta)
        return x


class Attention(nn.Layer):
    def __init__(self, 
                 dim, 
                 num_heads=8, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.,
                 wbits=-1, 
                 abits=-1, 
                 offset=False, 
                 learned=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.quant_input = QuantAct(nbits=abits, offset=offset, learned=learned)

        self.qkv = QuantLinear(dim, dim * 3, bias=qkv_bias, nbits=wbits)
        # self.proj_q = QuantLinear(dim, dim, bias=qkv_bias, nbits=wbits, learned=learned)
        # self.proj_k = QuantLinear(dim, dim, bias=qkv_bias, nbits=wbits, learned=learned)
        # self.proj_v = QuantLinear(dim, dim, bias=qkv_bias, nbits=wbits, learned=learned)


        self.attn_drop = nn.Dropout(attn_drop)

        # self.quant_q = QuantAct(nbits=abits, offset=offset, learned=learned)
        # self.quant_k = QuantAct(nbits=abits, offset=offset, learned=learned)
        # self.quant_v = QuantAct(nbits=abits, offset=offset, learned=learned)
        # self.quant_attn = QuantAct(nbits=abits, signed=False, offset=False, learned=learned)

        # self.quant_output = QuantAct(nbits=abits, offset=offset, learned=learned)


        self.proj = QuantLinear(dim, dim, nbits=wbits, learned=learned)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        x = self.quant_input(x)
        qkv = self.qkv(x).reshape((B, N, 3, self.num_heads, C // self.num_heads)).transpose((2, 0, 3, 1, 4))
        # q.shape : B H N D
        # q = self.proj_q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # k = self.proj_k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # v = self.proj_v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v = self.quant_q(q), self.quant_k(k), self.quant_v(v)
        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # attn = self.quant_attn(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        # x = self.quant_output(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output

class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class Block(nn.Layer):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer='nn.LayerNorm',
                 wbits=-1, 
                 abits=-1, 
                 offset=False, 
                 learned=True,
                 epsilon=1e-5):
        super().__init__()
        self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        self.attn = Attention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop,
            proj_drop=drop, 
            wbits=wbits, 
            abits=abits, 
            offset=offset, 
            learned=learned)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, 
                       hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer,
                       drop=drop, 
                       wbits=wbits, 
                       abits=abits, 
                       offset=offset, 
                       learned=learned)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# 参数初始化配置
trunc_normal_ = nn.initializer.TruncatedNormal(std=.02)
zeros_ = nn.initializer.Constant(value=0.)
ones_ = nn.initializer.Constant(value=1.)

# 将输入 x 由 int 类型转为 tuple 类型
def to_2tuple(x):
    return tuple([x] * 2)

class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, learned=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.quant = QuantAct(nbits=8, signed=True, learned=learned)
        self.proj = QuantConv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, nbits=8, learned=learned)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.quant(x - 0.456) + 0.456
        x = self.quant(x)
        x = self.proj(x).flatten(2).transpose((0,2,1))
        return x


# class HybridEmbed(nn.Layer):
#     """ CNN Feature Map Embedding
#     Extract feature map from CNN, flatten, project to embedding dim.
#     """
#     def __init__(self, backbone, img_size=384, feature_size=None, in_chans=3, embed_dim=768):
#         super().__init__()
#         assert isinstance(backbone, nn.Layer)
#         img_size = to_2tuple(img_size)
#         self.img_size = img_size
#         self.backbone = backbone
#         if feature_size is None:
#             with paddle.no_grad():
#                 # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
#                 # map for all networks, the feature metadata has reliable channel and stride info, but using
#                 # stride to calc feature dim requires info about padding of each stage that isn't captured.
#                 training = backbone.training
#                 if training:
#                     backbone.eval()
#                 o = self.backbone(paddle.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
#                 feature_size = o.shape[-2:]
#                 feature_dim = o.shape[1]
#                 backbone.train(training)
#         else:
#             feature_size = to_2tuple(feature_size)
#             feature_dim = self.backbone.feature_info.channels()[-1]
#         self.num_patches = feature_size[0] * feature_size[1]
#         self.proj = nn.Linear(feature_dim, embed_dim)

#     def forward(self, x):
#         x = self.backbone(x)[-1]
#         x = x.flatten(2).transpose(1, 2)
#         x = self.proj(x)
#         return x


class QuantVisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 class_dim=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 wbits=-1, 
                 abits=-1, 
                 act_layer=nn.GELU, 
                 offset=False, 
                 learned=True):
        super().__init__()
        if wbits == -1:
            print("Use float weights.")
        else:
            print(f"Use {wbits} bit weights.")
        if abits == -1:
            print("Use float activations.")
        else:
            print(f"Use {abits} bit activations.")
            
        self.class_dim = class_dim
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # if hybrid_backbone is not None:
        #     self.patch_embed = HybridEmbed(
        #         hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        # else:
        #     self.patch_embed = PatchEmbed(
        #         img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, learned=learned)
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, learned=learned)
                
        num_patches = self.patch_embed.num_patches

        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)
        # 人为追加class token，并使用该向量进行分类预测
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, depth)

        if act_layer == nn.ReLU:
            print('using relu nonlinearity')
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer,
                wbits=wbits, 
                abits=abits, 
                act_layer=act_layer, 
                offset=offset, 
                learned=learned) for i in range(depth)
            ])
        self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.quant = QuantAct(nbits=8, offset=offset, learned=learned)
        self.head = QuantLinear(embed_dim, class_dim, nbits=8, learned=learned) if class_dim > 0 else Identity()

        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, class_dim, global_pool=''):
        self.class_dim = class_dim
        self.head = nn.Linear(self.embed_dim, class_dim) if class_dim > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand((B, -1, -1))  # stole cls_tokens impl from Phil Wang, thanks
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x) # 三个in ParamBase copy_to func

        x = self.norm(x)
        return x[:, 0]
        # return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.quant(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

# class DistilledVisionTransformer(QuantVisionTransformer):
#     def __init__(self,
#                  img_size=224,
#                  patch_size=16,
#                  in_chans=3, 
#                  class_dim=1000,
#                  embed_dim=768,
#                  depth=12,
#                  num_heads=12,
#                  mlp_ratio=4,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  drop_rate=0., 
#                  attn_drop_rate=0.,
#                  drop_path_rate=0., 
#                  norm_layer='nn.LayerNorm',
#                  epsilon=1e-5,
#                  wbits=-1, 
#                  abits=-1, 
#                  act_layer=nn.GELU, 
#                  offset=False, 
#                  learned=True): 
#         # ViT 结构
#         super().__init__()
#         if wbits == -1:
#             print("Use float weights.")
#         else:
#             print(f"Use {wbits} bit weights.")
#         if abits == -1:
#             print("Use float activations.")
#         else:
#             print(f"Use {abits} bit activations.")
#         # 由于增加了 distillation token，所以也需要调整位置编码的长度
#         self.pos_embed = self.create_parameter(
#             shape=(1, self.patch_embed.num_patches + 2, self.embed_dim),
#             default_initializer=zeros_)
#         self.add_parameter("pos_embed", self.pos_embed)
#         # distillation token
#         self.dist_token = self.create_parameter(
#             shape=(1, 1, self.embed_dim), default_initializer=zeros_)
#         self.add_parameter("cls_token", self.cls_token)
#         # Classifier head
#         self.head_dist = nn.Linear(
#             self.embed_dim,
#             self.class_dim) if self.class_dim > 0 else Identity()

#         trunc_normal_(self.dist_token)
#         trunc_normal_(self.pos_embed)
#         self.head_dist.apply(self._init_weights)
#     # 获取图像特征
#     def forward_features(self, x):
#         B = paddle.shape(x)[0]
#         # 将图片分块，并调整每个块向量的维度
#         x = self.patch_embed(x)
#         # 将class token、distillation token与前面的分块进行拼接
#         cls_tokens = self.cls_token.expand((B, -1, -1))
#         dist_token = self.dist_token.expand((B, -1, -1))
#         x = paddle.concat((cls_tokens, dist_token, x), axis=1)
#         # 将编码向量中加入位置编码
#         x = x + self.pos_embed
#         x = self.pos_drop(x)
#         # 堆叠 transformer 结构
#         for blk in self.blocks:
#             x = blk(x)
#         # LayerNorm
#         x = self.norm(x)
#         # 提取class token以及distillation token的输出
#         return x[:, 0], x[:, 1]

#     def forward(self, x):
#         # 获取图像特征
#         x, x_dist = self.forward_features(x)
#         # 图像分类
#         x = self.head(x)
#         x_dist = self.head_dist(x_dist)
#         # 取 class token以及distillation token 的平均值作为结果
#         return (x + x_dist) / 2
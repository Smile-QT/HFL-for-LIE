

from torchvision.utils import save_image
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from utils.util import opt_get
from models.modules.flow import Conv2dZeros
import numbers

from einops import rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

import numpy as np
import torch
import cv2

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.dwconv3x3_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.conv1x1_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias)

        self.dwconv3x3_2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.conv1x1_2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x_gelu = F.gelu(x1) * x2
        x_gelu = self.dwconv3x3_1(x_gelu)
        x_gelu = self.conv1x1_1(x_gelu)

        x_sigmoid = F.sigmoid(x2) * x1
        x_sigmoid = self.dwconv3x3_2(x_sigmoid)
        x_sigmoid = self.conv1x1_2(x_sigmoid)

        x = x_gelu + x_sigmoid
        x = self.project_out(x)
        return x





# 类MDTA注意力机制，模块输出通道注意力图
# 在使用时，只需要传入dim参数，不需要修改num_heads和bias的值
class lei_MDTA_Attention(nn.Module):
    def __init__(self, dim=16, num_heads=1, bias=True):
        super(lei_MDTA_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        return attn

# 2x2区域的通道注意力拼接函数,YXL_x1, YXL_x2, YXL_x3, YXL_x4不需是张量
def concat_2x2(YXL_x1, YXL_x2, YXL_x3, YXL_x4):
    y1 = torch.cat([YXL_x1, YXL_x2], 3)
    y2 = torch.cat([YXL_x3, YXL_x4], 3)
    z = torch.cat([y1, y2], 2)
    return z

# 3x3区域的通道注意力拼接函数,YXL_x1, YXL_x2, YXL_x3, YXL_x4, YXL_x5, YXL_x6, YXL_x7, YXL_x8, YXL_x9需是张量
def concat_3x3(YXL_x1, YXL_x2, YXL_x3, YXL_x4, YXL_x5, YXL_x6, YXL_x7, YXL_x8, YXL_x9):
    y1 = torch.cat([YXL_x1, YXL_x2, YXL_x3], 3)
    y2 = torch.cat([YXL_x4, YXL_x5, YXL_x6], 3)
    y3 = torch.cat([YXL_x7, YXL_x8, YXL_x9], 3)
    z = torch.cat([y1, y2, y3], 2)
    return z

# 1.图像尺寸改为1/2，求每一个部分的通道注意力图
class Attention_22(nn.Module):
    def __init__(self, dim=16):
        super(Attention_22, self).__init__()
        self.x22_1_lei_atten = Attention_11(dim)
        self.x22_2_lei_atten = Attention_11(dim)
        self.x22_3_lei_atten = Attention_11(dim)
        self.x22_4_lei_atten = Attention_11(dim)

    def forward(self, x):
        batch_szie, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        # 区域1的长宽开始和结束位置
        H22_1_start = 0
        H22_1_end = H//2
        W22_1_start = 0
        W22_1_end = W // 2

        # 区域2的长宽开始和结束位置
        H22_2_start = 0
        H22_2_end = H//2
        W22_2_start = W // 2
        W22_2_end = W


        # 区域3的长宽开始和结束位置
        H22_3_start = H//2
        H22_3_end = H
        W22_3_start = 0
        W22_3_end = W // 2

        # 区域4的长宽开始和结束位置
        H22_4_start = H//2
        H22_4_end = H
        W22_4_start = W // 2
        W22_4_end = W

        # 每一个区域的特征矩阵
        x22_1 = x[:, :, H22_1_start:H22_1_end, W22_1_start:W22_1_end]
        x22_2 = x[:, :, H22_2_start:H22_2_end, W22_2_start:W22_2_end]
        x22_3 = x[:, :, H22_3_start:H22_3_end, W22_3_start:W22_3_end]
        x22_4 = x[:, :, H22_4_start:H22_4_end, W22_4_start:W22_4_end]

        # 求解每一个区域的通道权重均衡后的结果
        x22_1_channel_output = self.x22_1_lei_atten(x22_1)
        x22_2_channel_output = self.x22_2_lei_atten(x22_2)
        x22_3_channel_output = self.x22_3_lei_atten(x22_3)
        x22_4_channel_output = self.x22_4_lei_atten(x22_4)

        X_2x2_output = concat_2x2(x22_1_channel_output, x22_2_channel_output, x22_3_channel_output, x22_4_channel_output)

        return X_2x2_output

# 2.图像尺寸改为1/3，求每一个部分的通道注意力图
class Attention_33(nn.Module):
    def __init__(self, dim=16):
        super(Attention_33, self).__init__()
        self.x33_1_lei_atten = Attention_11(dim)
        self.x33_2_lei_atten = Attention_11(dim)
        self.x33_3_lei_atten = Attention_11(dim)
        self.x33_4_lei_atten = Attention_11(dim)
        self.x33_5_lei_atten = Attention_11(dim)
        self.x33_6_lei_atten = Attention_11(dim)
        self.x33_7_lei_atten = Attention_11(dim)
        self.x33_8_lei_atten = Attention_11(dim)
        self.x33_9_lei_atten = Attention_11(dim)

    def forward(self, x):
        batch_szie, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        # 区域1的长宽开始和结束位置
        H33_1_start = 0
        H33_1_end = H//3
        W33_1_start = 0
        W33_1_end = W//3

        # 区域2的长宽开始和结束位置
        H33_2_start = 0
        H33_2_end = H//3
        W33_2_start = W//3
        W33_2_end = (W//3) *2

        # 区域3的长宽开始和结束位置
        H33_3_start = 0
        H33_3_end = H//3
        W33_3_start = (W//3) *2
        W33_3_end = W

        # 区域4的长宽开始和结束位置
        H33_4_start = H//3
        H33_4_end = (H//3) * 2
        W33_4_start = 0
        W33_4_end = W//3

        # 区域5的长宽开始和结束位置
        H33_5_start = H//3
        H33_5_end = (H//3) * 2
        W33_5_start = W//3
        W33_5_end = (W//3) * 2

        # 区域6的长宽开始和结束位置
        H33_6_start = H//3
        H33_6_end = (H//3) * 2
        W33_6_start = (W//3) * 2
        W33_6_end = W

        # 区域7的长宽开始和结束位置
        H33_7_start = (H//3) * 2
        H33_7_end = H
        W33_7_start = 0
        W33_7_end = W//3

        # 区域8的长宽开始和结束位置
        H33_8_start = (H//3) * 2
        H33_8_end = H
        W33_8_start = W//3
        W33_8_end = (W//3) * 2

        # 区域9的长宽开始和结束位置
        H33_9_start = (H//3) * 2
        H33_9_end = H
        W33_9_start = (W//3) * 2
        W33_9_end = W

        # 每一个区域的特征矩阵
        x33_1 = x[:, :, H33_1_start:H33_1_end, W33_1_start:W33_1_end]
        x33_2 = x[:, :, H33_2_start:H33_2_end, W33_2_start:W33_2_end]
        x33_3 = x[:, :, H33_3_start:H33_3_end, W33_3_start:W33_3_end]
        x33_4 = x[:, :, H33_4_start:H33_4_end, W33_4_start:W33_4_end]
        x33_5 = x[:, :, H33_5_start:H33_5_end, W33_5_start:W33_5_end]
        x33_6 = x[:, :, H33_6_start:H33_6_end, W33_6_start:W33_6_end]
        x33_7 = x[:, :, H33_7_start:H33_7_end, W33_7_start:W33_7_end]
        x33_8 = x[:, :, H33_8_start:H33_8_end, W33_8_start:W33_8_end]
        x33_9 = x[:, :, H33_9_start:H33_9_end, W33_9_start:W33_9_end]

        # 求解每一个区域的通道注意力图
        x33_1_channel_output = self.x33_1_lei_atten(x33_1)
        x33_2_channel_output = self.x33_2_lei_atten(x33_2)
        x33_3_channel_output = self.x33_3_lei_atten(x33_3)
        x33_4_channel_output = self.x33_4_lei_atten(x33_4)
        x33_5_channel_output = self.x33_5_lei_atten(x33_5)
        x33_6_channel_output = self.x33_6_lei_atten(x33_6)
        x33_7_channel_output = self.x33_7_lei_atten(x33_7)
        x33_8_channel_output = self.x33_8_lei_atten(x33_8)
        x33_9_channel_output = self.x33_9_lei_atten(x33_9)

        X_3x3_output = concat_3x3(x33_1_channel_output, x33_2_channel_output, x33_3_channel_output,
                               x33_4_channel_output, x33_5_channel_output, x33_6_channel_output,
                               x33_7_channel_output, x33_8_channel_output, x33_9_channel_output)

        return X_3x3_output

# 3.图像尺寸不变，求整个通道的通道注意力图
class Attention_11(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=True):
        super(Attention_11, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.YXL_LayerNorm = LayerNorm(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.global_residual_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.global_residual_weight.data.fill_(0.1)



    def forward(self, x):
        x_global_residual = x
        x = self.YXL_LayerNorm(x)
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.global_residual_weight * x_global_residual + out


        return out



# 4.将3种注意力結果輸出，得到最终的結果
class Attention_weight_add(nn.Module):
    def __init__(self, dim=16):
        super(Attention_weight_add, self).__init__()

        self.X_11_result = Attention_11(dim)
        self.X_22_result = Attention_22(dim)
        self.X_33_result = Attention_33(dim)


        self.X_22_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.X_22_weight.data.fill_(0.1)

        self.X_33_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.X_33_weight.data.fill_(0.1)

        self.gdfn_LayerNorm = LayerNorm(dim)
        self.gdfn_restoremr = FeedForward(dim)

    def forward(self, x):
        X_11_final_output = self.X_11_result(x)
        X_22_final_output = self.X_22_result(x)
        X_33_final_output = self.X_33_result(x)


        X_22_final_output = self.X_22_weight * X_22_final_output
        X_33_final_output = self.X_33_weight * X_33_final_output

        out = X_11_final_output + X_22_final_output + X_33_final_output

        gdfn_identity = out
        out = self.gdfn_LayerNorm(out)
        out = self.gdfn_restoremr(out)
        out = out + gdfn_identity

        return out












##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
#         super(FeedForward, self).__init__()
#
#         hidden_features = int(dim*ffn_expansion_factor)
#
#         self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
#
#         self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
#
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         return x






class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.quyu_11_22_33 = Attention_weight_add(64)

    def forward(self, x):
        x = self.quyu_11_22_33(x)
        return x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.conv_128_64 = nn.Conv2d(128, 64, 3, 1, 1)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)
        self.conv_192_64 = nn.Conv2d(192, 64, 3, 1, 1)

    def forward(self, x):
        out1 = self.RDB1(x)
        out2 = self.RDB2(out1)
        out2_1 = self.conv_128_64(torch.cat([out1, out2], 1))
        out3 = self.RDB3(out2_1)
        out3_1 = self.conv_192_64(torch.cat([out1, out2, out3], 1))
        return out3_1 * 0.2 + x


class ConEncoder1(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, opt=None):
        self.opt = opt
        self.gray_map_bool = False
        self.concat_color_map = False
        if opt['concat_histeq']:
            in_nc = in_nc + 3
        if opt['concat_color_map']:
            in_nc = in_nc + 3
            self.concat_color_map = True
        if opt['gray_map']:
            in_nc = in_nc + 1
            self.gray_map_bool = True
        in_nc = in_nc + 6
        super(ConEncoder1, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.scale = scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_second = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### downsampling
        self.downconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.downconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.awb_para = nn.Linear(nf, 3)
        self.fine_tune_color_map = nn.Sequential(nn.Conv2d(nf, 3, 1, 1),nn.Sigmoid())

        # self.conv_3_3 = nn.Conv2d(3, 3, 1)

    def forward(self, x, get_steps=False):
        if self.gray_map_bool:
            x = torch.cat([x, 1 - x.mean(dim=1, keepdim=True)], dim=1)
        if self.concat_color_map:
            x = torch.cat([x, x / (x.sum(dim=1, keepdim=True) + 1e-4)], dim=1)

        raw_low_input = x[:, 0:3].exp()
        # fea_for_awb = F.adaptive_avg_pool2d(fea_down8, 1).view(-1, 64)
        awb_weight = 1  # (1 + self.awb_para(fea_for_awb).unsqueeze(2).unsqueeze(3))
        low_after_awb = raw_low_input * awb_weight
        # import pdb
        # pdb.set_trace()
        color_map = low_after_awb / (low_after_awb.sum(dim=1, keepdims=True) + 1e-4)
        dx, dy = self.gradient(color_map)
        noise_map = torch.max(torch.stack([dx.abs(), dy.abs()], dim=0), dim=0)[0]
        # color_map = self.fine_tune_color_map(torch.cat([color_map, noise_map], dim=1))

        fea = self.conv_first(torch.cat([x, color_map, noise_map], dim=1))
        fea = self.lrelu(fea)
        fea = self.conv_second(fea)
        fea_head = F.max_pool2d(fea, 2)

        block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        block_results = {}
        fea = fea_head
        for idx, m in enumerate(self.RRDB_trunk.children()):
            fea = m(fea)
            for b in block_idxs:
                if b == idx:
                    block_results["block_{}".format(idx)] = fea
        trunk = self.trunk_conv(fea)
        # fea = F.max_pool2d(fea, 2)
        fea_down2 = fea_head + trunk

        fea_down4 = self.downconv1(F.interpolate(fea_down2, scale_factor=1 / 2, mode='bilinear', align_corners=False,
                                                 recompute_scale_factor=True))
        fea = self.lrelu(fea_down4)

        fea_down8 = self.downconv2(
            F.interpolate(fea, scale_factor=1 / 2, mode='bilinear', align_corners=False, recompute_scale_factor=True))
        # fea = self.lrelu(fea_down8)

        # fea_down16 = self.downconv3(
        #     F.interpolate(fea, scale_factor=1 / 2, mode='bilinear', align_corners=False, recompute_scale_factor=True))
        # fea = self.lrelu(fea_down16)

        results = {'fea_up0': fea_down8,
                   'fea_up1': fea_down4,
                   'fea_up2': fea_down2,
                   'fea_up4': fea_head,
                   'last_lr_fea': fea_down4,
                   'color_map': self.fine_tune_color_map(F.interpolate(fea_down2, scale_factor=2))
                   }


#####################################################################################################################################
        # color_map_1 = results['color_map']
        # color_map_1 = torch.squeeze(color_map_1, 0)
        # color_map_1 = color_map_1.cpu().detach().numpy()
        # color_map_1 = color_map_1.transpose(1, 2, 0)
        # # 将图像转换为 NumPy 数组
        # image_array = np.array(color_map_1)
        #
        # # image_array = torch.from_numpy(image_array)
        # # image_array = image_array.exp()
        # # image_array = image_array * 1
        # # image_array = image_array.detach().numpy()  # tensor转换为ndarray
        #
        # # 计算每个通道的平均值
        # image_array = image_array.transpose(2, 0, 1)
        # channel_means = np.mean(image_array, axis=0) + 1e-4
        # channel_means = torch.from_numpy(channel_means)
        # channel_means = torch.unsqueeze(channel_means, 0)
        #
        # # 将每个通道的值除以相应通道的平均值
        # image_array = torch.from_numpy(image_array)
        # normalized_image_array = image_array / channel_means
        #
        # normalized_image_array = (normalized_image_array * 75.0).detach().numpy().transpose(1, 2, 0)
        # # normalized_image_array = cv2.cvtColor(normalized_image_array, cv2.COLOR_RGB2BGR)
        # normalized_image_array = normalized_image_array.astype(np.uint8)
        #
        # cv2.imwrite('C:/YXL/Write_seconed_papers/chengping_figure_and_table/figure_and_table_v7/figure/wangluokuangjiatu/LLFlow-main/YXL/drawing_photo/data/result/shengcheng_C.png',normalized_image_array)
        #

        # color_map_1 = results['color_map']
        # # color_map_1 = self.conv_3_3(color_map_1)
        # color_map_1 = torch.squeeze(color_map_1, 0)
        # color_map_1 = color_map_1.cpu().detach().numpy()
        # # color_map_1 = color_map_2.transpose(1, 2, 0)
        # # 将图像转换为 NumPy 数组
        # image_array = np.array(color_map_1)
        #
        # # huise_photo_low_1
        # image_array = np.array(image_array[2, :, :])
        #
        # # 将灰度值映射到 0-255 的范围，以便创建灰度图
        # scaled_data = (image_array*100).astype(np.uint8)
        #
        # # 使用OpenCV创建灰度图像
        # scaled_data = cv2.cvtColor(scaled_data, cv2.COLOR_GRAY2RGB)  # 将灰度数据转换为BGR格式，以便OpenCV正确保存图像
        #
        # # 保存图像
        # cv2.imwrite(
        #     'C:\YXL\Write_seconed_papers\chengping_figure_and_table/figure_and_table_v7/figure\wangluokuangjiatu\LLFlow-main\YXL\drawing_photo\zahupu/huise_photo_low_2.png',
        #     scaled_data)

        ####################################################################################################################################

        # 'color_map': color_map}  # raw

        if get_steps:
            for k, v in block_results.items():
                results[k] = v
            return results
        else:
            return None

    def gradient(self, x):
        def sub_gradient(x):
            left_shift_x, right_shift_x, grad = torch.zeros_like(
                x), torch.zeros_like(x), torch.zeros_like(x)
            left_shift_x[:, :, 0:-1] = x[:, :, 1:]
            right_shift_x[:, :, 1:] = x[:, :, 0:-1]
            grad = 0.5 * (left_shift_x - right_shift_x)
            return grad

        return sub_gradient(x), sub_gradient(torch.transpose(x, 2, 3)).transpose(2, 3)


class NoEncoder(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, opt=None):
        self.opt = opt
        self.gray_map_bool = False
        self.concat_color_map = False
        if opt['concat_histeq']:
            in_nc = in_nc + 3
        if opt['concat_color_map']:
            in_nc = in_nc + 3
            self.concat_color_map = True
        if opt['gray_map']:
            in_nc = in_nc + 1
            self.gray_map_bool = True
        in_nc = in_nc + 6
        super(NoEncoder, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.scale = scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_second = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### downsampling
        self.downconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.downconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.awb_para = nn.Linear(nf, 3)
        self.fine_tune_color_map = nn.Sequential(nn.Conv2d(nf, 3, 1, 1),nn.Sigmoid())

    def forward(self, x, get_steps=False):
        if self.gray_map_bool:
            x = torch.cat([x, 1 - x.mean(dim=1, keepdim=True)], dim=1)
        if self.concat_color_map:
            x = torch.cat([x, x / (x.sum(dim=1, keepdim=True) + 1e-4)], dim=1)

        raw_low_input = x[:, 0:3].exp()
        # fea_for_awb = F.adaptive_avg_pool2d(fea_down8, 1).view(-1, 64)
        awb_weight = 1  # (1 + self.awb_para(fea_for_awb).unsqueeze(2).unsqueeze(3))
        low_after_awb = raw_low_input * awb_weight
        # import pdb
        # pdb.set_trace()
        color_map = low_after_awb / (low_after_awb.sum(dim=1, keepdims=True) + 1e-4)
        dx, dy = self.gradient(color_map)
        noise_map = torch.max(torch.stack([dx.abs(), dy.abs()], dim=0), dim=0)[0]
        # color_map = self.fine_tune_color_map(torch.cat([color_map, noise_map], dim=1))

        fea = self.conv_first(torch.cat([x, color_map, noise_map], dim=1))
        fea = self.lrelu(fea)
        fea = self.conv_second(fea)
        fea_head = F.max_pool2d(fea, 2)

        block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        block_results = {}
        fea = fea_head
        for idx, m in enumerate(self.RRDB_trunk.children()):
            fea = m(fea)
            for b in block_idxs:
                if b == idx:
                    block_results["block_{}".format(idx)] = fea
        trunk = self.trunk_conv(fea)
        # fea = F.max_pool2d(fea, 2)
        fea_down2 = fea_head + trunk

        fea_down4 = self.downconv1(F.interpolate(fea_down2, scale_factor=1 / 2, mode='bilinear', align_corners=False,
                                                 recompute_scale_factor=True))
        fea = self.lrelu(fea_down4)

        fea_down8 = self.downconv2(
            F.interpolate(fea, scale_factor=1 / 2, mode='bilinear', align_corners=False, recompute_scale_factor=True))
        # fea = self.lrelu(fea_down8)

        # fea_down16 = self.downconv3(
        #     F.interpolate(fea, scale_factor=1 / 2, mode='bilinear', align_corners=False, recompute_scale_factor=True))
        # fea = self.lrelu(fea_down16)

        results = {'fea_up0': fea_down8*0,
                   'fea_up1': fea_down4*0,
                   'fea_up2': fea_down2*0,
                   'fea_up4': fea_head*0,
                   'last_lr_fea': fea_down4*0,
                   'color_map': self.fine_tune_color_map(F.interpolate(fea_down2, scale_factor=2))*0
                   }

        # 'color_map': color_map}  # raw

        if get_steps:
            for k, v in block_results.items():
                results[k] = v
            return results
        else:
            return None

    def gradient(self, x):
        def sub_gradient(x):
            left_shift_x, right_shift_x, grad = torch.zeros_like(
                x), torch.zeros_like(x), torch.zeros_like(x)
            left_shift_x[:, :, 0:-1] = x[:, :, 1:]
            right_shift_x[:, :, 1:] = x[:, :, 0:-1]
            grad = 0.5 * (left_shift_x - right_shift_x)
            return grad

        return sub_gradient(x), sub_gradient(torch.transpose(x, 2, 3)).transpose(2, 3)

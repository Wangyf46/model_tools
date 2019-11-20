import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weight_init import constant_init, kaiming_init, normal_init
from ..runner import load_checkpoint


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False


class AffineChannel2d(nn.Module):
    """ A simple channel-wise affine transformation operation """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        return x * self.weight.view(1, self.num_features, 1, 1) + \
            self.bias.view(1, self.num_features, 1, 1)


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1, dilation=1):
    """3x3 convolution with padding
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding * dilation,
        bias=bias,
        groups=groups,
        dilation=dilation)


def conv1x1(in_channels, out_channels, groups,
            batch_norm=True, relu=False):
    """1x1 convolution with padding"""
    modules = OrderedDict()

    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

    modules['conv1x1'] = conv

    if batch_norm:
        # modules['batch_norm'] = AffineChannel2d(out_channels)
        modules['batch_norm'] = nn.BatchNorm2d(out_channels)
    if relu:
        modules['relu'] = nn.ReLU()
    if len(modules) > 1:
        return nn.Sequential(modules)
    else:
        return conv


def basic_bn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', conv3x3(3, 24, stride=2)),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, combine='add', dilation=1):

        super(ShuffleUnit, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4
        self.dilation = dilation

        # define the type of ShuffleUnit
        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            # ShuffleUnit Figure 2c
            self.depthwise_stride = 2
            self._combine_func = self._concat

            # ensure output of concat has the same channels as
            # original output channels.
            self.out_channels -= self.in_channels
        else:
            raise ValueError("Cannot combine tensors with \"{}\""
                             "Only \"add\" and \"concat\" are"
                             "supported".format(self.combine))
        if dilation != 1:
            self.depthwise_stride = 1

        self.g_conv_1x1_compress = conv1x1(
            self.in_channels,
            self.bottleneck_channels,
            groups=self.groups,
            batch_norm=True,
            relu=True
        )

        # 3x3 depthwise convolution followed by batch normalization
        self.depthwise_conv3x3 = conv3x3(
            self.bottleneck_channels, self.bottleneck_channels,
            stride=self.depthwise_stride, groups=self.bottleneck_channels,
            dilation=dilation)
        # self.bn_after_depthwise = AffineChannel2d(self.bottleneck_channels)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)

        # Use 1x1 grouped convolution to expand from
        # bottleneck_channels to out_channels
        self.g_conv_1x1_expand = conv1x1(
            self.bottleneck_channels,
            self.out_channels,
            groups=self.groups,
            batch_norm=True,
            relu=False
        )

    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        # save for combining later with output
        residual = x

        if self.combine == 'concat' and self.dilation == 1:
            residual = F.avg_pool2d(residual, kernel_size=3,
                                    stride=2, padding=1)

        out = self.g_conv_1x1_compress(x)

        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)

        out = self._combine_func(residual, out)
        return F.relu(out)


class XceptionLike(nn.Module):
    def __init__(self,
                 num_stages=3,
                 dilations=(1, 1, 2),
                 out_indices=(0, 1, 2),
                 frozen_stages=-1,
                 bn_eval=False,
                 bn_frozen=False,
                 ):
        super(XceptionLike, self).__init__()
        assert num_stages >= 1 and num_stages <= 4
        stage_blocks = [3, 7, 3]
        assert len(dilations) == num_stages
        assert max(out_indices) < num_stages
        self.groups = 1
        self.stage_out_channels = [-1, 24, 144, 288, 567]
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen

        self.conv1 = conv3x3(3, 24, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # build blocks
        self.res_layers = []

        for i in range(num_stages):
            num_blocks = stage_blocks[i]
            dilation = dilations[i]
            stage_num = i + 2
            stage_layer = self._make_stage(stage_num, num_blocks, dilation=dilation)
            layer_name = 'stage{}'.format(stage_num)
            self.add_module(layer_name, stage_layer)
            self.res_layers.append(layer_name)

    def _make_stage(self, stage, num_blocks, dilation=1):
        modules = OrderedDict()
        stage_name = "ShuffleUnit_Stage{}".format(stage)

        #  concatenation unit is always used.
        first_module = ShuffleUnit(
            self.stage_out_channels[stage - 1],
            self.stage_out_channels[stage],
            groups=self.groups,
            combine='concat',
            dilation=dilation
        )
        modules[stage_name + "_0"] = first_module

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(num_blocks):
            name = stage_name + "_{}".format(i + 1)
            module = ShuffleUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups=self.groups,
                combine='add',
                dilation=dilation
            )
            modules[name] = module
        return nn.Sequential(modules)

    def load_checkpoint(self, pretrained, strict=False, logger=None):
        pretrianed_state_dict = torch.load(pretrained)

        for name, mod in self.named_modules():
            if isinstance(mod, AffineChannel2d):
                bn_mean = pretrianed_state_dict[name + '.running_mean']
                bn_var = pretrianed_state_dict[name + '.running_var']
                scale = pretrianed_state_dict[name + '.weight']
                bias = pretrianed_state_dict[name + '.bias']
                std = torch.sqrt(bn_var + 1e-5)
                new_scale = scale / std
                new_bias = bias - bn_mean * scale / std
                pretrianed_state_dict[name + '.weight'] = new_scale
                pretrianed_state_dict[name + '.bias'] = new_bias

        self.load_state_dict(pretrianed_state_dict, strict=strict)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            # self.load_checkpoint(pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        super(XceptionLike, self).train(mode)

        # # freeze all bn(affine) layers
        # self.apply(lambda m: freeze_params(m) if isinstance(m,AffineChannel2d) else None)

        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False

        # freeze res1
        for param in self.conv1.parameters():
            param.requires_grad = False

        # freeze other params
        for i in range(2, self.frozen_stages + 1):
            freeze_params(getattr(self, 'stage{}'.format(i)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return tuple(outs)
        else:
            return tuple(outs)



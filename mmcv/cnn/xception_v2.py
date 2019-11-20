import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from .weight_init import constant_init, kaiming_init, normal_init
from ..runner import load_checkpoint


STAGE_REPEATS= [4, 8, 4]
STAGE_OUT_CHANNELS = {
    0.5: [-1, 24, 64, 128, 256],
    1.0: [-1, 24, 128, 256, 512],
    1.125: [-1, 24, 144, 288, 576]
}


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class BottleneckWithCompress(nn.Module):
    def __init__(
            self,
            inp,
            oup,
            stride,
            bench_model,
            dilation=1
    ):
        super(BottleneckWithCompress, self).__init__()
        self.benchmodel = bench_model
        assert stride in [1, 2]

        # compressed channels
        cp = oup // 4

        if bench_model == 1:
            self.branch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, cp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(cp),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(cp, cp, 3, stride, 1 * dilation, groups=cp, bias=False, dilation=dilation),
                nn.BatchNorm2d(cp),
                # pw
                nn.Conv2d(cp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.branch1 = nn.AvgPool2d(3, stride=stride, padding=1)

            branche_op = oup - inp
            self.branch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, cp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(cp),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(cp, cp, 3, stride, 1 * dilation, groups=cp, bias=False, dilation=dilation),
                nn.BatchNorm2d(cp),
                # pw
                nn.Conv2d(cp, branche_op, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branche_op),
            )
    @staticmethod
    def _add(x, out):
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            out = self._add(x, self.branch2(x))
        elif 2 == self.benchmodel:
            out = self._concat(self.branch1(x), self.branch2(x))
        return F.relu_(out)


class XceptionV2(nn.Module):
    def __init__(self,
                 num_stages=3,
                 out_indices=(0, 1, 2),
                 frozen_stages=-1,
                 bn_eval=False,
                 bn_frozen=False,
                 width_mult=1.0
                 ):

        super(XceptionV2, self).__init__()
        assert 1 <= num_stages <= len(STAGE_REPEATS)
        self.stage_repeats = STAGE_REPEATS
        assert max(out_indices) < num_stages

        if width_mult in STAGE_OUT_CHANNELS.keys():
            self.stage_out_channels = STAGE_OUT_CHANNELS[width_mult]
        else:
            raise ValueError("Unsupported width %f" % width_mult)

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            features = []
            for i in range(numrepeat):
                if i == 0:
                    features.append(BottleneckWithCompress(input_channel, output_channel, 2, 2))
                else:
                    features.append(BottleneckWithCompress(input_channel, output_channel, 1, 1))
                input_channel = output_channel
            stage_name = 'stage_%d' % (idxstage + 1)
            self.stages.append(stage_name)
            self.add_module(stage_name, nn.Sequential(*features))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            print("Body load checkpoint")
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, stage_name in enumerate(self.stages):
            feature_layer = self.__getattr__(stage_name)
            x = feature_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return tuple(outs)
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(XceptionV2, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if mode and self.frozen_stages >= 0:
            # frozen stem
            for m in self.conv1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                for params in m.parameters():
                    params.requires_grad = False
            # frozen stages
            for i in range(0, self.frozen_stages):
                mod = self.__getattr__(self.stages[i])
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False


class XceptionV2Imagenet(nn.Module):

    def __init__(self, num_classes=1000, width_mult=1.0):
        super(XceptionV2Imagenet, self).__init__()

        self.backbone = XceptionV2(num_stages=3, out_indices=(2,), width_mult=width_mult)
        self.backbone.init_weights()
        # building last several layers
        input_channel = self.backbone.stage_out_channels[-1]
        # building classifier
        self.classifier = nn.Linear(input_channel, num_classes)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)

    def forward(self, x):
        x = self.backbone(x)[0]
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

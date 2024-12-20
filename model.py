import torch
from typing import Optional, List, Tuple
import copy
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class layer(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

class CNN(nn.Module):
    def __init__(self, train_shape, category, compress_rate=1):
        super(CNN, self).__init__()
        self.c1 = layer(1, int(64 * compress_rate), 3, 1, 1)
        self.c2 = layer(int(64 * compress_rate), int(128 * compress_rate), 3, 1, 1)
        self.c3 = layer(int(128 * compress_rate), int(256 * compress_rate), 3, 1, 1)
        self.c4 = layer(int(256 * compress_rate), int(512 * compress_rate), 3, 1, 1)
        ada_pool = nn.AdaptiveAvgPool2d(output_size=1)
        fl = nn.Flatten()
        linear = nn.Sequential(
            nn.Linear(int(512 * compress_rate), category),
        )
        self.classifier = nn.Sequential(ada_pool, fl, linear)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.classifier(x)
        return x

class DepthSeparableConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(DepthSeparableConv2d, self).__init__()
        self.depthConv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size, groups=in_channel, stride=stride, padding=padding),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self.pointConv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (1, 1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthConv(x)
        x = self.pointConv(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, train_shape, category, compress_rate=1):
        super(MobileNet, self).__init__()
        self.c1 = layer(1, int(64 * compress_rate), 3, 1, 1)
        self.c2 = DepthSeparableConv2d(int(64 * compress_rate), int(128 * compress_rate), 3, 1, 1)
        self.c3 = DepthSeparableConv2d(int(128 * compress_rate), int(256 * compress_rate), 3, 1, 1)
        self.c4 = DepthSeparableConv2d(int(256 * compress_rate), int(512 * compress_rate), 3, 1, 1)
        ada_pool = nn.AdaptiveAvgPool2d(output_size=1)
        fl = nn.Flatten()
        linear = nn.Sequential(
            nn.Linear(int(512 * compress_rate), category),
        )
        self.classifier = nn.Sequential(ada_pool, fl, linear)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.classifier(x)
        return x

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class SEBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 rd_ratio: float = 0.0625) -> None:

        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels=in_channels,
                                out_channels=int(in_channels * rd_ratio),
                                kernel_size=1,
                                stride=1,
                                bias=True)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio),
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class RepMobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, inference_mode=False, num_conv_branches=3) -> None:
        super(RepMobileBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            return self.activation(self.reparam_conv(x))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(out)

    def reparameterize(self):
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size, padding):
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list

class RepMobile(nn.Module):
    def __init__(self, num_classes, compress_rate=1, inference_mode=False, num_conv_branches=3):
        super().__init__()
        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches
        self.stage1 = layer(1, int(64 * compress_rate), (3, 3), (1, 1), (1, 1))
        self.stage2 = self._make_stage(int(64 * compress_rate), int(128 * compress_rate))
        self.stage3 = self._make_stage(int(128 * compress_rate), int(256 * compress_rate))
        self.stage4 = self._make_stage(int(256 * compress_rate), int(512 * compress_rate))
        ada_pool = nn.AdaptiveAvgPool2d(output_size=1)
        fl = nn.Flatten()
        linear = nn.Linear(int(512 * compress_rate), num_classes)
        self.classifier = nn.Sequential(ada_pool, fl, linear)

    def _make_stage(self, in_channel, out_channel):
        DepthConv = RepMobileBlock(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        PointConv = RepMobileBlock(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, groups=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        return nn.Sequential(DepthConv, PointConv)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classifier(x)
        return x

def reparameterize_model(model: torch.nn.Module):
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model
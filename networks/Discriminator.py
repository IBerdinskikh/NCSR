from torch import nn as nn
from collections import OrderedDict


# class VGGStyleDiscriminator128(nn.Module):
#     """VGG style discriminator with input size 128 x 128.

#     It is used to train SRGAN and ESRGAN.

#     Args:
#         num_in_ch (int): Channel number of inputs. Default: 3.
#         num_feat (int): Channel number of base intermediate features.
#             Default: 64.
#     """

#     def __init__(self, num_in_ch, num_feat):
#         super(VGGStyleDiscriminator128, self).__init__()

#         self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
#         self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
#         self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

#         self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
#         self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
#         self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
#         self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

#         self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
#         self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
#         self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
#         self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

#         self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
#         self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
#         self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
#         self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

#         self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
#         self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
#         self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
#         self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

#         self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
#         self.linear2 = nn.Linear(100, 1)

#         # activation function
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#     def forward(self, x):
#         assert x.size(2) == 128 and x.size(3) == 128, (
#             f"Input spatial size must be 128x128, " f"but received {x.size()}."
#         )

#         feat = self.lrelu(self.conv0_0(x))
#         feat = self.lrelu(
#             self.bn0_1(self.conv0_1(feat))
#         )  # output spatial size: (64, 64)

#         feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
#         feat = self.lrelu(
#             self.bn1_1(self.conv1_1(feat))
#         )  # output spatial size: (32, 32)

#         feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
#         feat = self.lrelu(
#             self.bn2_1(self.conv2_1(feat))
#         )  # output spatial size: (16, 16)

#         feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
#         feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: (8, 8)

#         feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
#         feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: (4, 4)

#         feat = feat.view(feat.size(0), -1)
#         feat = self.lrelu(self.linear1(feat))
#         out = self.linear2(feat)
#         return out

####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            "activation layer [{:s}] is not found".format(act_type)
        )
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == "batch":
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == "instance":
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(
            "normalization layer [{:s}] is not found".format(norm_type)
        )
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == "reflect":
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == "replicate":
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError(
            "padding layer [{:s}] is not implemented".format(pad_type)
        )
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(
    in_nc,
    out_nc,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
    mode="CNA",
):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ["CNA", "NAC", "CNAC"], "Wong conv mode [{:s}]".format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != "zero" else None
    padding = padding if pad_type == "zero" else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    a = act(act_type) if act_type else None
    if "CNA" in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == "NAC":
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


####################
# Discriminator
####################

# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    """VGG style discriminator with input size 128 x 128.
    It is used to train SRGAN and ESRGAN.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
    """

    def __init__(self, num_in_ch, num_feat):
        super(Discriminator_VGG_128, self).__init__()

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == 128 and x.size(3) == 128, (
            f"Input spatial size must be 128x128, " f"but received {x.size()}."
        )

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(
            self.bn0_1(self.conv0_1(feat))
        )  # output spatial size: (64, 64)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(
            self.bn1_1(self.conv1_1(feat))
        )  # output spatial size: (32, 32)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(
            self.bn2_1(self.conv2_1(feat))
        )  # output spatial size: (16, 16)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: (8, 8)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: (4, 4)

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


class Discriminator_VGG_96(nn.Module):
    def __init__(
        self, in_nc, base_nf, norm_type="batch", act_type="leakyrelu", mode="CNA"
    ):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = conv_block(
            in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, mode=mode
        )
        conv1 = conv_block(
            base_nf,
            base_nf,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 48, 64
        conv2 = conv_block(
            base_nf,
            base_nf * 2,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv3 = conv_block(
            base_nf * 2,
            base_nf * 2,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 24, 128
        conv4 = conv_block(
            base_nf * 2,
            base_nf * 4,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv5 = conv_block(
            base_nf * 4,
            base_nf * 4,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 12, 256
        conv6 = conv_block(
            base_nf * 4,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv7 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 6, 512
        conv8 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv9 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 3, 512
        self.features = sequential(
            conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_192(nn.Module):
    def __init__(
        self, in_nc, base_nf, norm_type="batch", act_type="leakyrelu", mode="CNA"
    ):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = conv_block(
            in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, mode=mode
        )
        conv1 = conv_block(
            base_nf,
            base_nf,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 96, 64
        conv2 = conv_block(
            base_nf,
            base_nf * 2,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv3 = conv_block(
            base_nf * 2,
            base_nf * 2,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 48, 128
        conv4 = conv_block(
            base_nf * 2,
            base_nf * 4,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv5 = conv_block(
            base_nf * 4,
            base_nf * 4,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 24, 256
        conv6 = conv_block(
            base_nf * 4,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv7 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 12, 512
        conv8 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv9 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 6, 512
        conv10 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv11 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 3, 512
        self.features = sequential(
            conv0,
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            conv6,
            conv7,
            conv8,
            conv9,
            conv10,
            conv11,
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
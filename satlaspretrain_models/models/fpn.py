import torch
import collections
import torchvision


class FPN(torch.nn.Module):
    def __init__(self, backbone_channels):
        super(FPN, self).__init__()

        out_channels = 128
        in_channels_list = [ch[1] for ch in backbone_channels]
        self.fpn = torchvision.ops.FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)

        self.out_channels = [[ch[0], out_channels] for ch in backbone_channels]

    def forward(self, x):
        inp = collections.OrderedDict([('feat{}'.format(i), el) for i, el in enumerate(x)])
        output = self.fpn(inp)
        output = list(output.values())

        return output


class Upsample(torch.nn.Module):
    # Computes an output feature map at 1x the input resolution.
    # It just applies a series of transpose convolution layers on the
    # highest resolution features from the backbone (FPN should be applied first).

    def __init__(self, backbone_channels):
        super(Upsample, self).__init__()
        self.in_channels = backbone_channels

        out_channels = backbone_channels[0][1]
        self.out_channels = [(1, out_channels)] + backbone_channels

        layers = []
        depth, ch = backbone_channels[0]
        while depth > 1:
            next_ch = max(ch//2, out_channels)
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(ch, ch, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)
            ch = next_ch
            depth /= 2

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        output = self.layers(x[0])
        return [output] + x

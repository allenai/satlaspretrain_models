import torch
import collections
import torchvision

class FPN(torch.nn.Module):
    def __init__(self, backbone_channels, out_channels=128):
        super(FPN, self).__init__()

        in_channels_list = [ch[1] for ch in backbone_channels]
        self.fpn = torchvision.ops.FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)

        self.out_channels = [[ch[0], out_channels] for ch in backbone_channels]

    def forward(self, x):
        inp = collections.OrderedDict([('feat{}'.format(i), el) for i, el in enumerate(x)])
        output = self.fpn(inp)
        output = list(output.values())

        return output

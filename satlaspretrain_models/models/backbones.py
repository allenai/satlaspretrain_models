import torch.nn
import torchvision

class SwinBackbone(torch.nn.Module):
    def __init__(self, num_channels, arch='swinb'):
        super(SwinBackbone, self).__init__()

        if arch == 'swinb':
            self.backbone = torchvision.models.swin_v2_b()
            self.out_channels = [
                [4, 128],
                [8, 256],
                [16, 512],
                [32, 1024],
            ]
        elif arch == 'swint':
            self.backbone = torchvision.models.swin_v2_t()
            self.out_channels = [
                [4, 96],
                [8, 192],
                [16, 384],
                [32, 768],
            ]
        else:
            raise ValueError("Backbone architecture not supported.")

        self.backbone.features[0][0] = torch.nn.Conv2d(num_channels, self.backbone.features[0][0].out_channels, kernel_size=(4, 4), stride=(4, 4))

    def forward(self, x):
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]


class ResnetBackbone(torch.nn.Module):
    def __init__(self, num_channels, arch='resnet50'):
        super(ResnetBackbone, self).__init__()

        if arch == 'resnet50':
            self.resnet = torchvision.models.resnet.resnet50(weights=None)
            ch = [256, 512, 1024, 2048]
        elif arch == 'resnet152':
            self.resnet = torchvision.models.resnet.resnet152(weights=None)
            ch = [256, 512, 1024, 2048]
        else:
            raise ValueError("Backbone architecture not supported.")

        self.resnet.conv1 = torch.nn.Conv2d(num_channels, self.resnet.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.out_channels = [
            [4, ch[0]],
            [8, ch[1]],
            [16, ch[2]],
            [32, ch[3]],
        ]

    def train(self, mode=True):
        super(ResnetBackbone, self).train(mode)
        if self.freeze_bn:
            for module in self.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        layer1 = self.resnet.layer1(x)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)

        return [layer1, layer2, layer3, layer4]


class AggregationBackbone(torch.nn.Module):
    def __init__(self, num_channels, backbone):
        super(AggregationBackbone, self).__init__()

        # Number of channels to pass to underlying backbone.
        self.image_channels = num_channels

        # Prepare underlying backbone.
        self.backbone = backbone

        # Features from images within each group are aggregated separately.
        # Then the output is the concatenation across groups.
        # e.g. [[0], [1, 2]] to compare first image against the others
        self.groups = [[0, 1, 2, 3, 4, 5, 6, 7]]

        ngroups = len(self.groups)
        self.out_channels = [(depth, ngroups*count) for (depth, count) in self.backbone.out_channels]

        self.aggregation_op = 'max'

    def forward(self, x):
        # First get features of each image.
        all_features = []
        for i in range(0, x.shape[1], self.image_channels):
            features = self.backbone(x[:, i:i+self.image_channels, :, :])
            all_features.append(features)

        # Now compute aggregation over each group.
        # We handle each depth separately.
        l = []
        for feature_idx in range(len(all_features[0])):
            aggregated_features = []
            for group in self.groups:
                group_features = []
                for image_idx in group:
                    # We may input fewer than the maximum number of images.
                    # So here we skip image indices in the group that aren't available.
                    if image_idx >= len(all_features):
                        continue

                    group_features.append(all_features[image_idx][feature_idx])
                # Resulting group features are (depth, batch, C, height, width).
                group_features = torch.stack(group_features, dim=0)

                if self.aggregation_op == 'max':
                    group_features = torch.amax(group_features, dim=0)

                aggregated_features.append(group_features)

            # Finally we concatenate across groups.
            aggregated_features = torch.cat(aggregated_features, dim=1)

            l.append(aggregated_features)

        return l

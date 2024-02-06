import torch
import torch.nn
import requests
from io import BytesIO

from satlaspretrain_models.utils import Backbone, Head, SatlasPretrain_weights, adjust_state_dict_prefix
from satlaspretrain_models.models import *

class Weights:
    def __init__(self):
        """
        Class to manage downloading weights and formatting them to be loaded into SatlasPretrain models.
        """
        super(Weights, self).__init__()

    def get_pretrained_model(self, model_identifier, fpn=False, head=None, num_categories=None, device='cuda'):
        """
        Find and load pretrained SatlasPretrain weights, based on the model_identifier argument.
        Option to load pretrained FPN and/or a randomly initialized head.

        Args:
            model_identifier: 
            fpn (bool): Whether or not to load a pretrained FPN along with the Backbone.
            head (enum): If specified, a randomly initialized Head will be created along with the 
                        Backbone and [optionally] the FPN.
            num_categories (int): Number of categories to be included in output from prediction head.
        """
        # Validate that the model identifier is supported.
        if not model_identifier in SatlasPretrain_weights.keys():
            raise ValueError("Invalid model_identifier. See utils.SatlasPretrain_weights.")

        if head and (num_categories is None):
            raise ValueError("Must specify num_categories if head is desired.")

        model_info = SatlasPretrain_weights[model_identifier]

        # Use hardcoded huggingface url to download weights.
        weights_url = model_info['url']
        response = requests.get(weights_url)
        if response.status_code == 200:
            weights_file = BytesIO(response.content)
        else: 
            raise Exception(f"Failed to download weights from {url}")
        
        if device == 'cpu':
            weights = torch.load(weights_file, map_location=torch.device('cpu'))
        else:
            weights = torch.load(weights_file)

        # Initialize a pretrained model using the Model() class.
        model = Model(model_info['num_channels'], model_info['multi_image'], model_info['backbone'], fpn=fpn, head=head, 
                        num_categories=num_categories, weights=weights)
        return model


class Model(torch.nn.Module):
    def __init__(self, num_channels=3, multi_image=False, backbone=Backbone.SWINB, fpn=False, head=None, num_categories=None, weights=None):
        """
        Initializes a model, based on desired imagery source and model components. This class can be used directly to
        create a randomly initialized model (if weights=None) or can be called from the Weights class to initialize a 
        SatlasPretrain pretrained foundation model.

        Args:
            num_channels (int): Number of input channels that the backbone model should expect.
            multi_image (bool): Whether or not the model should expect single-image or multi-image input.
            backbone (Backbone): The architecture of the pretrained backbone. All image sources support SwinTransformer.
            fpn (bool): Whether or not to feed imagery through the pretrained Feature Pyramid Network after the backbone.
            head (Head): If specified, a randomly initialized head will be included in the model. 
            num_categories (int): If a Head is being returned as part of the model, must specify how many outputs are wanted.
            weights (torch weights): Weights to be loaded into the model. Defaults to None (random initialization) unless 
                                    initialized using the Weights class.
        """
        super(Model, self).__init__()

        # Validate user-provided arguments.
        if not isinstance(backbone, Backbone):
            raise ValueError("Invalid backbone.")
        if head and not isinstance(head, Head):
            raise ValueError("Invalid head.")
        if head and (num_categories is None):
            raise ValueError("Must specify num_categories if head is desired.")

        self.backbone = self._initialize_backbone(num_channels, backbone, multi_image, weights)
        self.fpn = self._initialize_fpn(self.backbone.out_channels, weights) if fpn else None
        if head:
            self.head = self._initialize_head(head, self.fpn.out_channels, num_categories) if fpn else self._initialize_head(head, self.backbone.out_channels, num_categories)
        else:
            self.head = None

    def _initialize_backbone(self, num_channels, backbone_arch, multi_image, weights):
        # Load backbone model according to specified architecture.
        if backbone_arch == Backbone.SWINB:
            backbone = SwinBackbone(num_channels, arch='swinb')
        elif backbone_arch == Backbone.SWINT:
            backbone = SwinBackbone(num_channels, arch='swint')
        elif backbone_arch == Backbone.RESNET50:
            backbone = ResnetBackbone(num_channels, arch='resnet50')
        elif backbone_arch == Backbone.RESNET152:
            backbone = ResnetBackbone(num_channels, arch='resnet152')
        else:
            raise ValueError("Unsupported backbone architecture.")
        
        # If using a model for multi-image, need the Aggretation to wrap underlying backbone model.
        prefix, prefix_allowed_count = None, None
        if backbone_arch in [Backbone.RESNET50, Backbone.RESNET152]:
            prefix_allowed_count = 0
        elif multi_image:
            backbone = AggregationBackbone(num_channels, backbone)
            prefix_allowed_count = 2
        else:
            prefix_allowed_count = 1

        # Load pretrained weights into the intialized backbone if weights were specified.
        if weights is not None:
            state_dict = adjust_state_dict_prefix(weights, 'backbone', 'backbone.', prefix_allowed_count)
            backbone.load_state_dict(state_dict)

        return backbone

    def _initialize_fpn(self, backbone_channels, weights):
        fpn = FPN(backbone_channels)

        # Load pretrained weights into the intialized FPN if weights were specified.
        if weights is not None:
            state_dict = adjust_state_dict_prefix(weights, 'fpn', 'intermediates.0.', 0)
            fpn.load_state_dict(state_dict)
        return fpn

    def _initialize_head(self, head, backbone_channels, num_categories):
        # Initialize the head (classification, detection, etc.) if specified
        if head == Head.CLASSIFY:
            return SimpleHead('classification', backbone_channels, num_categories)
        elif head == Head.MULTICLASSIFY:
            return SimpleHead('multi-label-classification', backbone_channels, num_categories)
        elif head == Head.SEGMENT:
            return SimpleHead('segment', backbone_channels, num_categories)
        elif head == Head.BINSEGMENT:
            return SimpleHead('bin_segment', backbone_channels, num_categories)
        elif head == Head.REGRESS:
            return SimpleHead('regress', backbone_channels, num_categories)
        elif head == Head.DETECT:
            return FRCNNHead('detect', backbone_channels, num_categories)
        elif head == Head.INSTANCE:
            return FRCNNHead('instance', backbone_channels, num_categories)
        return None

    def forward(self, imgs, targets=None):
        # Define forward pass
        x = self.backbone(imgs)
        if self.fpn:
            x = self.fpn(x)
        if self.head:
            x, loss = self.head(imgs, x, targets)
            return x, loss
        return x


if __name__ == "__main__":
    weights_manager = Weights()

    # Test loading in all available pretrained backbone models, without FPN or Head.
    # Test feeding in a random tensor as input.
    for model_id in SatlasPretrain_weights.keys():
        print("Attempting to load ...", model_id)
        model_info = SatlasPretrain_weights[model_id]
        model = weights_manager.get_pretrained_model(model_id)
        rand_img = torch.rand((8, model_info['num_channels'], 128, 128))
        output = model(rand_img)
        print("Successfully initialized the pretrained model with ID:", model_id)

    # Test loading in all available pretrained backbone models, with FPN, without Head.
    # Test feeding in a random tensor as input.
    for model_id in SatlasPretrain_weights.keys():
        print("Attempting to load ...", model_id, " with pretrained FPN.")
        model_info = SatlasPretrain_weights[model_id]
        model = weights_manager.get_pretrained_model(model_id, fpn=True)
        rand_img = torch.rand((8, model_info['num_channels'], 128, 128))
        output = model(rand_img)
        print("Successfully initialized the pretrained model with ID:", model_id, " with FPN.")

    # Test loading in all available pretrained backbones, with FPN and with every possible Head.
    # Test feeding in a random tensor as input. Randomly generated targets are fed into detection/instance heads.
    for model_id in SatlasPretrain_weights.keys():
        model_info = SatlasPretrain_weights[model_id]
        for head in Head:
            print("Attempting to load ...", model_id, " with pretrained FPN and randomly initialized ", head, " Head.")
            model = weights_manager.get_pretrained_model(model_id, fpn=True, head=head, num_categories=2)
            rand_img = torch.rand((1, model_info['num_channels'], 128, 128))

            rand_targets = None
            if head == Head.DETECT:
                rand_targets = [{   
                        'boxes': torch.tensor([[100, 100, 110, 110], [30, 30, 40, 40]], dtype=torch.float32),
                        'labels': torch.tensor([0,1], dtype=torch.int64)
                    }]
            elif head == Head.INSTANCE:
                rand_targets = [{
                        'boxes': torch.tensor([[100, 100, 110, 110], [30, 30, 40, 40]], dtype=torch.float32),
                        'labels': torch.tensor([0,1], dtype=torch.int64),
                        'masks': torch.zeros_like(rand_img)
                    }]
            output, loss = model(rand_img, rand_targets)
            print("Successfully initialized the pretrained model with ID:", model_id, " with FPN and randomly initialized ", head, " Head.") 

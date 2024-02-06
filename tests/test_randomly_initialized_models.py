import pytest
import torch

from satlaspretrain_models.model import Model
from satlaspretrain_models.utils import Backbone, Head

# Test loading randomly initialized backbone models without FPN or Head
@pytest.mark.parametrize("backbone", [backbone for backbone in Backbone])
def test_random_backbone(backbone):
    model = Model(num_channels=3, multi_image=False, backbone=backbone, fpn=False, head=None, num_categories=None, weights=None)
    rand_img = torch.rand((8, 3, 128, 128))
    output = model(rand_img)
    assert output is not None

# Test loading randomly initialized backbone models with FPN, without Head
@pytest.mark.parametrize("backbone", [backbone for backbone in Backbone])
def test_random_backbone_with_fpn(backbone):
    model = Model(num_channels=3, multi_image=False, backbone=backbone, fpn=True, head=None, num_categories=None, weights=None)
    rand_img = torch.rand((8, 3, 128, 128))
    output = model(rand_img)
    assert output is not None

# Test loading pretrained backbones with FPN and every possible Head
@pytest.mark.parametrize("backbone,head", [(backbone, head) for backbone in Backbone for head in Head])
def test_random_backbone_with_fpn_and_head(backbone, head):
    model = Model(num_channels=3, multi_image=False, backbone=backbone, fpn=True, head=head, num_categories=2, weights=None)
    rand_img = torch.rand((1, 3, 128, 128))

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

    output, loss = model(rand_img, rand_targets) if rand_targets else model(rand_img)
    assert output is not None and loss is not None

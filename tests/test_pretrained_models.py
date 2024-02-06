import pytest
import torch

from satlaspretrain_models.model import Weights
from satlaspretrain_models.utils import SatlasPretrain_weights, Head

# Fixture for weights manager
@pytest.fixture(scope="module")
def weights_manager():
    return Weights()

# Test loading pretrained backbone models without FPN or Head
@pytest.mark.parametrize("model_id", SatlasPretrain_weights.keys())
def test_pretrained_backbone(weights_manager, model_id):
    model_info = SatlasPretrain_weights[model_id]
    model = weights_manager.get_pretrained_model(model_id)
    rand_img = torch.rand((8, model_info['num_channels'], 128, 128))
    output = model(rand_img)
    assert output is not None

# Test loading pretrained backbone models with FPN, without Head
@pytest.mark.parametrize("model_id", SatlasPretrain_weights.keys())
def test_pretrained_backbone_with_fpn(weights_manager, model_id):
    model_info = SatlasPretrain_weights[model_id]
    model = weights_manager.get_pretrained_model(model_id, fpn=True)
    rand_img = torch.rand((8, model_info['num_channels'], 128, 128))
    output = model(rand_img)
    assert output is not None

# Test loading pretrained backbones with FPN and every possible Head
@pytest.mark.parametrize("model_id,head", [(model_id, head) for model_id in SatlasPretrain_weights.keys() for head in Head])
def test_pretrained_backbone_with_fpn_and_head(weights_manager, model_id, head):
    model_info = SatlasPretrain_weights[model_id]
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
    
    output, loss = model(rand_img, rand_targets) if rand_targets else model(rand_img)
    assert output is not None and loss is not None


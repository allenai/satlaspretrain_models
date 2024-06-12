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
    rand_img = torch.rand((8, model_info["num_channels"], 128, 128)).float()
    output = model(rand_img)
    assert output is not None


# Test loading pretrained backbone models with FPN, without Head
@pytest.mark.parametrize("model_id", SatlasPretrain_weights.keys())
def test_pretrained_backbone_with_fpn(weights_manager, model_id):
    model_info = SatlasPretrain_weights[model_id]
    model = weights_manager.get_pretrained_model(model_id, fpn=True)
    rand_img = torch.rand((8, model_info["num_channels"], 128, 128)).float()
    output = model(rand_img)
    assert output is not None


# Test loading pretrained backbones with FPN, every possible Head and every possible infra
@pytest.mark.parametrize(
    "model_id,head,infra",
    [
        (model_id, head, infra)
        for model_id in SatlasPretrain_weights.keys()
        for head in Head
        for infra in [0, 1]
    ],
)
def test_pretrained_backbone_with_fpn_and_head(weights_manager, model_id, head, infra):
    model_info = SatlasPretrain_weights[model_id]
    model = weights_manager.get_pretrained_model(
        model_id, fpn=True, head=head, num_categories=2, infra=infra
    )
    rand_img = torch.rand((1, model_info["num_channels"], 128, 128)).float()
    rand_ir = torch.rand((1, 1, 128, 128)).float()

    rand_targets = None
    if head == Head.DETECT:
        rand_targets = [
            {
                "boxes": torch.tensor(
                    [[100, 100, 110, 110], [30, 30, 40, 40]], dtype=torch.float32
                ),
                "labels": torch.tensor([0, 1], dtype=torch.int64),
            }
        ]
    elif head == Head.INSTANCE:
        rand_targets = [
            {
                "boxes": torch.tensor(
                    [[100, 100, 110, 110], [30, 30, 40, 40]], dtype=torch.float32
                ),
                "labels": torch.tensor([0, 1], dtype=torch.int64),
                "masks": torch.zeros_like(rand_img),
            }
        ]
    elif head == Head.BINSEGMENT:
        rand_targets = torch.zeros((1, 2, 32, 32))
    elif head == Head.REGRESS:
        rand_targets = torch.zeros((1, 2, 32, 32)).float()
    elif head == Head.CLASSIFY:
        rand_targets = torch.tensor([1])

    # TODO: add rand_targets for SEGMENT and MULTICLASSIFY

    if rand_targets is not None:
        out = model.backbone(rand_img)
        out = model.fpn(out)
        out = model.upsample(out)
        out[0] = torch.cat((out[0], rand_ir), dim=1) if infra == 1 else out[0]
        output, loss = model.head(rand_img, out, rand_targets)
        assert output is not None
        assert loss is not None
    else:
        out = model.backbone(rand_img)
        out = model.fpn(out)
        out = model.upsample(out)
        out[0] = torch.cat((out[0], rand_ir), dim=1) if infra == 1 else out[0]
        output = model.head(rand_img, out)
        assert output is not None

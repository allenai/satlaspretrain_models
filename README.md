SatlasPretrain Models: Foundation models for satellite and aerial imagery.
--------------------------------------------------------------------------

**SatlasPretrain** is a large-scale pre-training dataset for remote sensing image understanding. This work
was published at ICCV 2023. Details and download links for the dataset can be found
[here](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md).

This repository contains `satlaspretrain_models`, a lightweight library to easily load pretrained SatlasPretrain models for:
- Sentinel-2
- Sentinel-1
- Landsat 8/9
- 0.5-2 m/pixel aerial imagery

These models can be fine-tuned on downstream tasks that use these image sources, leading to faster training
and improved performance compared to training from other initializations.

Model Structure and Usage
-------------------------
The SatlasPretrain models consist of three main components: backbone, feature pyramid network (FPN), and prediction head.

![SatlasPretrain model architecture diagram, described in the next paragraph.](satlasnet.png)

For models trained on *multi-image* input, the backbone is applied on each individual image, and then max pooling is applied
in the temporal dimension, i.e., across the multiple aligned images. *Single-image* models input an individual image.

This package allows you to load the backbone or backbone+FPN using a model checkpoint ID from the tables below.

```python
MODEL_CHECKPOINT_ID = "Sentinel2_SwinB_SI_RGB"
model = weights_manager.get_pretrained_model(MODEL_CHECKPOINT_ID)
model = weights_manager.get_pretrained_model(MODEL_CHECKPOINT_ID, fpn=True)
```

The output of the model is the multi-scale feature map (either from the backbone or from the FPN).

For a complete fine-tuning example, [see our tutorial on fine-tuning the pre-trained model on EuroSAT](https://github.com/allenai/satlaspretrain_models/blob/main/demo.ipynb).
You can also use the pre-trained models in [TorchGeo](TODO2) (link TODO2), a library for training remote sensing models in PyTorch ([see our usage guide](TODO2)).

Installation
--------------
```
conda create --name satlaspretrain python==3.9
conda activate satlaspretrain
pip install satlaspretrain_models
```

Available Pretrained Models
---------------------------

The tables below list available model checkpoint IDs (like `Sentinel2_SwinB_SI_RGB`).
Checkpoints are released under [ODC-BY](https://github.com/allenai/satlas/blob/main/DataLicense).
This package will download model checkpoints automatically, but you can download them directly using the links below if desired.

#### Sentinel-2 Pretrained Models
|  | Single-image, RGB | Multi-image, RGB |
| ---------- | ------------ | ------------ |
| **Swin-v2-Base** | [Sentinel2_SwinB_SI_RGB](https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_si_rgb.pth?download=true) | [Sentinel2_SwinB_MI_RGB](https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_mi_rgb.pth?download=true) |
| **Swin-v2-Tiny** | [Sentinel2_SwinT_SI_RGB](https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_si_rgb.pth?download=true) | |
| **Resnet50** | [Sentinel2_Resnet50_SI_RGB](https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_si_rgb.pth?download=true) | [Sentinel2_Resnet50_MI_RGB](https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_mi_rgb.pth?download=true) |
| **Resnet152** | [Sentinel2_Resnet152_SI_RGB](https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_si_rgb.pth?download=true) | [Sentinel2_Resnet152_MI_RGB](https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_mi_rgb.pth?download=true) |

|  | Single-image, MS | Multi-image, MS |
| ---------- | ------------ | ------------ |
| **Swin-v2-Base** | [Sentinel2_SwinB_SI_MS](https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_si_ms.pth?download=true) | [Sentinel2_SwinB_MI_MS](https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_mi_ms.pth?download=true) |
| **Resnet50** | [Sentinel2_Resnet50_SI_MS](https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_si_ms.pth?download=true) | |
| **Resnet152** | [Sentinel2_Resnet152_SI_MS](https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_si_ms.pth?download=true) | |

#### Sentinel-1 Pretrained Models
|  | Single-image, VH+VV | Multi-image, VH+VV |
| ---------- | ------------ | ------------ |
| **Swin-v2-Base** | [Sentinel1_SwinB_SI](https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel1_swinb_si.pth?download=true) | [Sentinel1_SwinB_MI](https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel1_swinb_mi.pth?download=true) |

#### Landsat 8/9 Pretrained Models
|  | Single-image, all bands | Multi-image, all bands |
| ---------- | ------------ | ------------ |
| **Swin-v2-Base** | [Landsat_SwinB_SI](https://huggingface.co/allenai/satlas-pretrain/resolve/main/landsat_swinb_si.pth?download=true) | [Landsat_SwinB_MI](https://huggingface.co/allenai/satlas-pretrain/resolve/main/landsat_swinb_mi.pth?download=true) |

#### Aerial (0.5-2m/px high-res imagery) Pretrained Models
|  | Single-image, RGB | Multi-image, RGB |
| ---------- | ------------ | ------------ |
| **Swin-v2-Base** | [Aerial_SwinB_SI](https://huggingface.co/allenai/satlas-pretrain/resolve/main/aerial_swinb_si.pth?download=true) | [Aerial_SwinB_MI](https://huggingface.co/allenai/satlas-pretrain/resolve/main/aerial_swinb_mi.pth?download=true) |


*Single-image* models learn strong representations for individual satellite or aerial images, while *multi-image* models use multiple image captures of the same location for added robustness when making predictions about static objects. In *multi-image* models, feature maps from the backbone are passed through temporal max pooling, so the backbone itself is still applied on individual images, but is trained to provide strong representations after the temporal max pooling step. See [ModelArchitecture.md](https://github.com/allenai/satlas/blob/main/ModelArchitecture.md) for more details.

Sentinel-2 *RGB* models input the B4, B3, and B2 bands only, while the multi-spectral (*MS*) models input 9 bands. The aerial (0.5-2m/px high-res imagery) models input *RGB* NAIP and other high-res images, and we have found them to be effective on aerial imagery from a variety of sources. Landsat models input B1-B11 (*all bands*). Sentinel-1 models input *VV and VH* bands.
See [Normalization.md](https://github.com/allenai/satlas/blob/main/Normalization.md) for details on how pixel values should be normalized for input to the pre-trained models.

Usage Examples
--------------
First initialize a `Weights` instance:

```python
import satlaspretrain_models
import torch
weights_manager = satlaspretrain_models.Weights()
```

Then choose a **model_identifier** from the tables above to specify the pretrained model you want to load.
Below are examples showing how to load in a few of the available models.

#### Pretrained single-image Sentinel-2 RGB model, backbone only:
```python
model = weights_manager.get_pretrained_model(model_identifier="Sentinel2_SwinB_SI_RGB")

# Expected input is a portion of a Sentinel-2 L1C TCI image.
# The 0-255 pixel values should be divided by 255 so they are 0-1.
# tensor = tci_image[None, :, :, :] / 255
tensor = torch.zeros((1, 3, 512, 512), dtype=torch.float32)

# Since we only loaded the backbone, it outputs feature maps from the Swin-v2-Base backbone.
output = model(tensor)
print([feature_map.shape for feature_map in output])
# [torch.Size([1, 128, 128, 128]), torch.Size([1, 256, 64, 64]), torch.Size([1, 512, 32, 32]), torch.Size([1, 1024, 16, 16])]
```

#### Pretrained single-image Sentinel-1 model, backbone+FPN
```python
model = weights_manager.get_pretrained_model("Sentinel1_SwinB_SI", fpn=True)

# Expected input is a portion of a Sentinel-1 vh+vv image (in that order).
# The 16-bit pixel values should be divided by 255 and clipped to 0-1 (any pixel values greater than 255 become 1).
# tensor = torch.clip(torch.stack([vh_image, vv_image], dim=0)[None, :, :, :] / 255, 0, 1)
tensor = torch.zeros((1, 2, 512, 512), dtype=torch.float32)

# The model outputs feature maps from the FPN.
output = model(tensor)
print([feature_map.shape for feature_map in output])
# [torch.Size([1, 128, 128, 128]), torch.Size([1, 128, 64, 64]), torch.Size([1, 128, 32, 32]), torch.Size([1, 128, 16, 16])]
```

#### Prediction heads

Although the checkpoints include prediction head parameters, these heads are task-specific, so loading the head parameters is not supported in this repository.
Computing outputs from the pre-trained prediction heads is supported [in the dataset codebase](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md#visualizing-outputs-on-new-images).

For convenience when fine-tuning on certain types of tasks, though, `satlaspretrain_models` supports attaching certain heads (initialized randomly) to the pre-trained model:

```python
# Backbone and FPN parameters initialized from checkpoint, head parameters initialized randomly.
model = weights_manager.get_pretrained_model(MODEL_CHECKPOINT_ID, fpn=True, head=satlaspretrain_models.Head.CLASSIFY, head_outputs=2)
```

The following head architectures are available:
- *Segmentation*: U-Net Decoder w/ Cross Entropy loss
- *Detection*: Faster R-CNN Decoder
- *Instance Segmentation*: Mask R-CNN Decoder
- *Regression*: U-Net Decoder w/ L1 loss
- *Classification*: Pooling + Linear layers
- *Multi-label Classification*: Pooling + Linear layers

#### Pretrained multi-image aerial model, backbone + FPN + classification head:
```python
# num_categories is the number of categories to predict.
# All heads are randomly initialized and provided only for convenience for fine-tuning.
model = weights_manager.get_pretrained_model("Aerial_SwinB_MI", fpn=True, head=satlaspretrain_models.Head.CLASSIFY, num_categories=2)

# Expected input is 8-bit (0-255) aerial images at 0.5 - 2 m/pixel.
# The 0-255 pixel values should be divided by 255 so they are 0-1.
# This multi-image model is trained to input 4 images but should perform well with different numbers of images.
# tensor = torch.stack([rgb_image1, rgb_image2], dim=0)[None, :, :, :, :] / 255
tensor = torch.zeros((1, 4, 3, 512, 512), dtype=torch.float32)

# The head needs to be fine-tuned on a downstream classification task.
# It outputs classification probabilities.
model.eval()
output = model(tensor.reshape(1, 4*3, 512, 512))
print(output)
# tensor([[0.0266, 0.9734]])
```

#### Pretrained multi-image Landsat model, backbone + FPN + detection head
```python
# num_categories is the number of bounding box detection categories.
# All heads are randomly initialized and provided only for convenience for fine-tuning.
model = weights_manager.get_pretrained_model("Landsat_SwinB_MI", fpn=True, head=satlaspretrain_models.Head.DETECT, num_categories=5)

# Expected input is Landsat B1-B11 stacked in order.
# This multi-image model is trained to input 8 images but should perform well with different numbers of images.
# The 16-bit pixel values are normalized as follows:
# landsat_images = torch.stack([landsat_image1, landsat_image2], dim=0)
# tensor = torch.clip(landsat_images[None, :, :, :, :]-4000)/16320, 0, 1)
tensor = torch.zeros((1, 8, 11, 512, 512), dtype=torch.float32)

# The head needs to be fine-tuned on a downstream object detection task.
# It outputs bounding box detections.
model.eval()
output = model(tensor.reshape(1, 8*11, 512, 512))
print(output)
#[{'boxes': tensor([[ 67.0772, 239.2646, 95.6874, 16.3644], ...]),
# 'labels': tensor([3, ...]),
# 'scores': tensor([0.5443, ...])}]
```

Demos
-----
We provide a [demo](https://github.com/allenai/satlaspretrain_models/blob/main/demo.py) showing how to finetune a 
SatlasPretrain Sentinel-2 model on the EuroSAT classification task.

Tests
-----
There are tests to test loading pretrained models and one to test randomly initialized models.

To run the tests, run the following command from the root directory:
`pytest tests/`

Contact
-------
If you have any questions, please email `satlas@allenai.org` or open an issue [here](https://github.com/allenai/satlaspretrain_models/issues/new).

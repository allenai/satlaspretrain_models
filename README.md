SatlasPretrain Models: Foundation models for satellite and aerial imagery. 
--------------------------------------------------------------------------

**SatlasPretrain** is a large-scale pre-training dataset for remote sensing image understanding. This work 
was published at ICCV 2023. Details and download links for the dataset can be found 
[here](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md).

This repo contains code to easily load pretrained SatlasPretrain models for:
- Sentinel-2
- Sentinel-1
- Landsat 8/9
- 0.5-2m/px aerial imagery

These models can be fine-tuned to downstream tasks that use these image sources, leading to faster training 
and improved performance compared to training from scratch.

Model Structure
--------------
The SatlasPretrain models consist of three main components: backbone, feature pyramid network (FPN), and prediction head.

For models trained on *multi-image* input, the backbone is applied on each individual image, and then max pooling is applied
in the temporal dimension, i.e., across the multiple aligned images. *Single-image* models input an individual image.

This package allows you to load any of the following:
- pretrained backbone
- pretrained backbone + pretrained FPN
- pretrained backbone + pretrained FPN + randomly initialized head
- pretrained backbone + randomly initialized head
- randomly initialized backbone and/or FPN and/or head

The following randomly initialized heads are available:
- *Segmentation*: U-Net Decoder w/ Cross Entropy loss
- *Detection*: Faster R-CNN Decoder
- *Instance Segmentation*: Mask R-CNN Decoder
- *Regression*: U-Net Decoder w/ L1 loss
- *Classification*: Pooling + Linear layers
- *Multi-label Classification*: Pooling + Linear layers

Installation
--------------
```
conda create --name satlaspretrain python==3.9
conda activate satlaspretrain
pip install satlaspretrain_models
```

Available Pretrained Models model_ids
----------------------------
#### Sentinel-2 Pretrained Models
| Configuration | Swin-v2-Base | Swin-v2-Tiny | Resnet50 | Resnet152 |
| ---------- | ------------ | ------------ | -------- | --------- |
| *Single-image, RGB* | Sentinel2_SwinB_SI_RGB | Sentinel2_SwinT_SI_RGB | Sentinel2_Resnet50_SI_RGB | Sentinel2_Resnet152_SI_RGB |
| *Single-image, MS* | Sentinel2_SwinB_SI_MS | unavailable | Sentinel2_Resnet50_SI_MS | Sentinel2_Resnet152_SI_MS |
| *Multi-image, RGB* | Sentinel2_SwinB_MI_RGB | unavailable | Sentinel2_Resnet50_MI_RGB | Sentinel2_Resnet152_MI_RGB |
| *Multi-image, MS* | Sentinel2_SwinB_MI_MS | unavailable | unavailable | unavailable |

#### Sentinel-1 Pretrained Models
| Configuration | Swin-v2-Base | Swin-v2-Tiny | Resnet50 | Resnet152 |
| ---------- | ------------ | ------------ | -------- | --------- |
| *Single-image, vh+vv* | Sentinel1_SwinB_SI | unavailable | unavailable | unavailable |
| *Multi-image, vh+vv* | Sentinel1_SwinB_MI | unavailable | unavailable | unavailable |

#### Landsat 8/9 Pretrained Models
| Configuration | Swin-v2-Base | Swin-v2-Tiny | Resnet50 | Resnet152 |
| ---------- | ------------ | ------------ | -------- | --------- |
| *Single-image, all bands* | Landsat_SwinB_SI | unavailable | unavailable | unavailable |
| *Multi-image, all bands* | Landsat_SwinB_MI | unavailable | unavailable | unavailable |

#### Aerial (0.5-2m/px high-res imagery) Pretrained Models
| Configuration | Swin-v2-Base | Swin-v2-Tiny | Resnet50 | Resnet152 |
| ---------- | ------------ | ------------ | -------- | --------- |
| *Single-image, RGB* | Aerial_SwinB_SI | unavailable | unavailable | unavailable |
| *Multi-image, RGB* | Aerial_SwinB_MI | unavailable | unavailable | unavailable |


*Single-image* models learn strong representations for individual satellite or aerial images, while *multi-image* models use multiple image captures of the same location for added robustness when making predictions about static objects. In *multi-image* models, feature maps from the backbone are passed through temporal max pooling, so the backbone itself is still applied on individual images, but is trained to provide strong representations after the temporal max pooling step. See [ModelArchitecture.md](ModelArchitecture.md) for more details.

Sentinel-2 *RGB* models input the B2, B3, and B4 bands only, while the multi-spectral (*MS*) models input 9 bands (see [Normalization.md](Normalization.md#sentinel-2-images) for details). The aerial (0.5-2m/px high-res imagery) models input *RGB* NAIP and other high-res images, and we have found them to be effective on aerial imagery from a variety of sources and datasets. Landsat models input B1-B11 (*all bands*). Sentinel-1 models input *VV and VH* bands. 

Pretrained Model Examples
---------------
Choose a **model_id** from one of the Available Pretrained Models model_ids tables to specify the pretrained model you want to load. Below are
examples showing how to load in a few of the available models.

To load a pretrained single-image Sentinel-2 backbone model:
```
import satlaspretrain_models

# This example loads a Swin-v2-Base single-image model that was pretrained on Sentinel-2 RGB images.
model_id = 'Sentinel2_SwinB_SI_RGB'

weights_manager = satlaspretrain_models.Weights()
model = weights_manager.get_pretrained_model(model_id)
```

To load a pretrained single-image Sentinel-1 backbone + FPN:
```
import satlaspretrain_models

# This example loads a Swin-v2-Base single-image model that was pretrained on Sentinel-1 images.
model_id = 'Sentinel1_SwinB_SI'

weights_manager = satlaspretrain_models.Weights()
model = weights_manager.get_pretrained_model(model_id, fpn=True)
```

To load a pretrained multi-image Aerial backbone + FPN and a randomly initialized classification head:
```
import satlaspretrain_models

# This examples loads a Swin-v2-Base multi-image model that was pretrained on Aerial RGB images.
model_id = 'Aerial_SwinB_MI'

weights_manager = satlaspretrain_models.Weights()
model = weights_manager.get_pretrained_model(model_id, fpn=True, head=satlaspretrain_models.Head.CLASSIFY, head_outputs)
```

To load a pretrained multi-image Landsat 8/9 backbone + FPN and a randomly initialized detection head:
```
import satlaspretrain_models

# This examples loads a Swin-v2-Base multi-image model that was pretrained on Landsat 8/9 images.
model_id = 'Landsat_SwinB_MI'

weights_manager = satlaspretrain_models.Weights()
model = weights_manager.get_pretrained_model(model_id, fpn=True, head=satlaspretrain_models.Head.DETECT, head_outputs=2)
```

To load a randomly initialized single-image RGB Sentinel-2 backbone + FPN + segmentation head:
```
import satlaspretrain_models

model = Model(num_channels=3, multi_image=False, backbone=satlaspretrain_models.Backbone.SWIN, 
		fpn=True, head=satlaspretrain_models.Head.SEGMENT, head_outputs=2, weights=None) 
```

Tests
-----
There are tests to test loading pretrained models and one to test randomly initialized models.

To run the tests, run the following command from the root directory:
`pytest tests/`

Contact
-------
If you have any questions, please email `piperw@allenai.org` or open an issue [here](https://github.com/allenai/satlaspretrain_models/issues/new).

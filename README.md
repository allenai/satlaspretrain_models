SatlasPretrain Models: Foundation models for satellite and aerial imagery. 
--------------------------------------------------------------------------

**SatlasPretrain** is a large-scale pre-training dataset for remote sensing image understanding. This work 
was published at ICCV 2023. Details and download links for the dataset can be found 
[here](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md).

This repo contains code to easily load pretrained SatlasPretrain models. There are models for **Sentinel-2, 
Sentinel-1, and Landsat satellite imagery, as well as for 0.5-2 m/pixel aerial imagery.**
These models can be fine-tuned to downstream tasks that use those image sources, and improve performance compared to 
training from scratch.

Installation
--------------
`pip install satlaspretrain_models`

Model Structure
--------------
The SatlasPretrain models consist of three main components: backbone, feature pyramid network (FPN), and prediction head.

For models trained on **multi-image** input, the backbone is applied on each individual image, and then max pooling is applied
in the temporal dimension, i.e., across the multiple aligned images. **Sinlge-image** models input an individual image.

This package allows you to load any of the following:
- pretrained backbone
- pretrained backbone + pretrained FPN
- pretrained backbone + pretrained FPN + randomly initialized head
- pretrained backbone + randomly initialized head

The following randomly initialized heads are available:
- *Segmentation*: U-Net Decoder w/ Cross Entropy loss
- *Detection*: Faster R-CNN Decoder
- *Instance Segmentation*: Mask R-CNN Decoder
- *Regression*: U-Net Decoder w/ L1 loss
- *Classification*: Pooling + Linear layers
- *Multi-label Classification*: Pooling + Linear layers

Available Pretrained Models
----------------------------
Sentinel-2 Pretrained Models
| Configuration | Swin-v2-Base | Swin-v2-Tiny | Resnet50 | Resnet152 |
| ---------- | ------------ | ------------ | -------- | --------- |
| Single-image, RGB | Sentinel2_SwinB_SI_RGB | Sentinel2_SwinT_SI_RGB | Sentinel2_Resnet50_SI_RGB | Sentinel2_Resnet152_SI_RGB |
| Single-image, MS | Sentinel2_SwinB_SI_MS | unavailable | Sentinel2_Resnet50_SI_MS | Sentinel2_Resnet152_SI_MS |
| Multi-image, RGB | Sentinel2_SwinB_MI_RGB | unavailable | Sentinel2_Resnet50_MI_RGB | Sentinel2_Resnet152_MI_RGB |
| Multi-image, MS | Sentinel2_SwinB_MI_MS | unavailable | unavailable | unavailable |

Sentinel-1 Pretrained Models
| Configuration | Swin-v2-Base | Swin-v2-Tiny | Resnet50 | Resnet152 |
| ---------- | ------------ | ------------ | -------- | --------- |
| Single-image, vh+vv | Sentinel1_SwinB_SI | unavailable | unavailable | unavailable |
| Multi-image, vh+vv | Sentinel1_SwinB_MI | unavailable | unavailable | unavailable |

Landsat 8/9 Pretrained Models
| Configuration | Swin-v2-Base | Swin-v2-Tiny | Resnet50 | Resnet152 |
| ---------- | ------------ | ------------ | -------- | --------- |
| Single-image, all bands | Landsat_SwinB_SI | unavailable | unavailable | unavailable |
| Multi-image, all bands | Landsat_SwinB_MI | unavailable | unavailable | unavailable |

Aerial (0.5-2m/px high-res imagery) Pretrained Models
| Configuration | Swin-v2-Base | Swin-v2-Tiny | Resnet50 | Resnet152 |
| ---------- | ------------ | ------------ | -------- | --------- |
| Single-image, RGB | Aerial_SwinB_SI | unavailable | unavailable | unavailable |
| Multi-image, RGB | Aerial_SwinB_MI | unavailable | unavailable | unavailable |


Single-image models learn strong representations for individual satellite or aerial images, while multi-image models use multiple image captures of the same location for added robustness when making predictions about static objects. In multi-image models, feature maps from the backbone are passed through temporal max pooling, so the backbone itself is still applied on individual images, but is trained to provide strong representations after the temporal max pooling step. See [ModelArchitecture.md](ModelArchitecture.md) for more details.

Sentinel-2 RGB models input B2, B3, and B4 only, while the multi-spectral (MS) models input 9 bands (see [Normalization.md](Normalization.md#sentinel-2-images) for details). The aerial (0.5-2m/px high-res imagery) models input RGB NAIP and other high-res images, and we have found them to be effective on aerial imagery from a variety of sources and datasets. Landsat models input B1-B11 (all bands). Sentinel-1 models input the VV and VH bands. 

Usage Examples
---------------
To load a pretrained single-image Sentinel-2 backbone model:
```
import satlaspretrain_models

# Choose a model_id from the one of the tables above, dependent on the desired image type and model architecture.
# This example loads a Swin-v2-Base single-image model that was pretrained on Sentinel-2 RGB images.
model_id = 'Sentinel2_SwinB_SI_RGB'

weights_manager = satlaspretrain_models.Weights()
model = weights_manager.get_pretrained_model(model_id)
```

To load a pretrained single-image Sentinel-1 backbone + FPN:
```
import satlaspretrain_models

# Choose a model_id from the one of the tables above, dependent on the desired image type and model architecture.
# This example loads a Swin-v2-Base single-image model that was pretrained on Sentinel-1 images.
model_id = 'Sentinel1_SwinB_SI'

weights_manager = satlaspretrain_models.Weights()
model = weights_manager.get_pretrained_model(model_id, fpn=True)
```

To load a pretrained multi-image Aerial backbone + FPN and a randomly initialized classification head:
```
import satlaspretrain_models

# Choose a model_id from the one of the tables above, dependent on the desired image type and model architecture.
# This examples loads a Swin-v2-Base multi-image model that was pretrained on Aerial RGB images.
model_id = 'Aerial_SwinB_MI'

weights_manager = satlaspretrain_models.Weights()
model = weights_manager.get_pretrained_model(model_id, fpn=True, head=satlaspretrain_models.Head.CLASSIFY, head_outputs)
```

To load a pretrained multi-image Landsat 8/9 backbone + FPN and a randomly initialized detection head:
```
import satlaspretrain_models

# Choose a model_id from the one of the tables above, dependent on the desired image type and model architecture.
# This examples loads a Swin-v2-Base multi-image model that was pretrained on Landsat 8/9 images.
model_id = 'Landsat_SwinB_MI'

weights_manager = satlaspretrain_models.Weights()
model = weights_manager.get_pretrained_model(model_id, fpn=True, head=satlaspretrain_models.Head.DETECT, head_outputs=2)
```

Contact
-------
If you have any questions, please email `piperw@allenai.org` or open an issue [here](https://github.com/allenai/satlaspretrain_models/issues/new).

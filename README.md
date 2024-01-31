##SatlasPretrain Models: Foundation models for satellite and aerial imagery.

SatlasPretrain is a large-scale pre-training dataset for remote sensing image understanding. Details and download 
links for the dataset can be found [here](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md)

This repo contains code to easily load pretrained SatlasPretrain models. There are models traned on each of the Sentinel-2, 
Sentinel-1, and LandSat satellites, as well as models trained on 0.5-2 m/pixel aerial imagery.
These models can be fine-tuned to downstream tasks that use those image sources, and improve performance compared to 
training from scratch.

# Installation
`pip install satlaspretrain_models`

# Model Structure
The SatlasPretrain models consist of three main components: backbone, feature pyramid network (FPN), and prediction head.

This package allows you to load any of the following:
- pretrained backbone
- pretrained backbone + pretrained FPN
- pretrained backbone + pretrained FPN + randomly initialized head
- pretrained backbone + randomly initialized head

# Available Pretrained Models
To load a pretrained backbone model, you will just need the following code:
```
import satlaspretrain_models

# Choose a model_id from the table below, dependent on the desired image type and model architecture.
model_id = 'Sentinel2_SwinB_SI_RGB'  # model initialization using sample model_id from the table below

weights_manager = satlaspretrain_models.Weights() 
model = weights_manager.get_pretrained_model(model_id)
```

| Image Type | Swin-v2-Base | Swin-v2-Tiny | Resnet50 | Resnet152 |
| ---------- | ------------ | ------------ | -------- | --------- |
| Sentinel-2, single-image, RGB | Sentinel2_SwinB_SI_RGB | Sentinel2_SwinT_SI_RGB | Sentinel2_Resnet50_SI_RGB | Sentinel2_Resnet152_SI_RGB |
| Sentinel-2, single-image, RGB | Sentinel2_SwinB_SI_MS | unavailable | Sentinel2_Resnet50_SI_MS | Sentinel2_Resnet152_SI_MS |
| Sentinel-2, multi-image, RGB | Sentinel2_SwinB_MI_RGB | unavailable | Sentinel2_Resnet50_MI_RGB | Sentinel2_Resnet152_MI_RGB |
| Sentinel-2, multi-image, RGB | Sentinel2_SwinB_MI_MS | unavailable | unavailable | unavailable |
| Sentinel-1, single-image, vh+vv | Sentinel1_SwinB_SI | unavailable | unavailable | unavailable |
| Sentinel-1, multi-image, vh+vv | Sentinel1_SwinB_MI | unavailable | unavailable | unavailable |
| Landsat 8/9, single-image, all bands | Landsat_SwinB_SI | unavailable | unavailable | unavailable |
| Landsat 8/9, multi-image, all bands | Landsat_SwinB_MI | unavailable | unavailable | unavailable |

Single-image models learn strong representations for individual satellite or aerial images, while multi-image models use multiple image captures of the same location for added robustness when making predictions about static objects. In multi-image models, feature maps from the backbone are passed through temporal max pooling, so the backbone itself is still applied on individual images, but is trained to provide strong representations after the temporal max pooling step. See [ModelArchitecture.md](ModelArchitecture.md) for more details.

Sentinel-2 RGB models input B2, B3, and B4 only, while the multi-band models input 9 bands (see [Normalization.md](Normalization.md#sentinel-2-images) for details). NAIP models input RGB aerial images, and we have found them to be effective on aerial imagery from a variety of sources and datasets. Landsat models input B1-B11 (all bands).

# Usage Examples


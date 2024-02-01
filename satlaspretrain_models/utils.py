from enum import Enum, auto

class Backbone(Enum):
    SWINB = auto()
    SWINT = auto()
    RESNET50 = auto()
    RESNET152 = auto()

class Head(Enum):
    CLASSIFY = auto()
    MULTICLASSIFY = auto()
    DETECT = auto()
    INSTANCE = auto()
    SEGMENT = auto()
    BINSEGMENT = auto()
    REGRESS = auto()

# Dictionary of arguments needed to load in each SatlasPretrain pretrained model.
SatlasPretrain_weights = {
    'Sentinel2_SwinB_SI_RGB': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_si_rgb.pth?download=true',
        'backbone': Backbone.SWINB,
        'num_channels': 3,
        'multi_image': False
    },
    'Sentinel2_SwinB_MI_RGB': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_mi_rgb.pth?download=true',
        'backbone': Backbone.SWINB,
        'num_channels': 3,
        'multi_image': True
    },
    'Sentinel2_SwinB_SI_MS': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_si_ms.pth?download=true',
        'backbone': Backbone.SWINB,
        'num_channels': 9,
        'multi_image': False
    },
    'Sentinel2_SwinB_MI_MS': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_mi_ms.pth?download=true',
        'backbone': Backbone.SWINB,
        'num_channels': 9,
        'multi_image': True
    },
    'Sentinel1_SwinB_SI': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel1_swinb_si.pth?download=true',
        'backbone': Backbone.SWINB,
        'num_channels': 2,
        'multi_image': False
    },
    'Sentinel1_SwinB_MI': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel1_swinb_mi.pth?download=true',
        'backbone': Backbone.SWINB,
        'num_channels': 2,
        'multi_image': True
    },
    'Landsat_SwinB_SI': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/landsat_swinb_si.pth?download=true',
        'backbone': Backbone.SWINB,
        'num_channels': 11,
        'multi_image': False
    },
    'Landsat_SwinB_MI': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/landsat_swinb_mi.pth?download=true',
        'backbone': Backbone.SWINB,
        'num_channels': 11,
        'multi_image': True
    },
    'Aerial_SwinB_SI': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/aerial_swinb_si.pth?download=true',
        'backbone': Backbone.SWINB,
        'num_channels': 3,
        'multi_image': False
    },
    'Aerial_SwinB_MI': {
        'url':'https://huggingface.co/allenai/satlas-pretrain/resolve/main/aerial_swinb_mi.pth?download=true',
        'backbone': Backbone.SWINB,
        'num_channels': 3,
        'multi_image': True
    },
    'Sentinel2_SwinT_SI_RGB': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_si_rgb.pth?download=true',
        'backbone': Backbone.SWINT,
        'num_channels': 3,
        'multi_image': False
    },
    'Sentinel2_SwinT_SI_MS': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_si_ms.pth?download=true',
        'backbone': Backbone.SWINT,
        'num_channels': 9,
        'multi_image': False
    },
    'Sentinel2_Resnet50_SI_RGB': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_si_rgb.pth?download=true',
        'backbone': Backbone.RESNET50,
        'num_channels': 3,
        'multi_image': False
    },
    'Sentinel2_Resnet50_SI_MS': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_si_ms.pth?download=true',
        'backbone': Backbone.RESNET50,
        'num_channels': 9,
        'multi_image': False
    },
    'Sentinel2_Resnet50_MI_RGB': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_mi_rgb.pth?download=true',
        'backbone': Backbone.RESNET50,
        'num_channels': 3,
        'multi_image': True
    },
    'Sentinel2_Resnet152_SI_RGB': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_si_rgb.pth?download=true',
        'backbone': Backbone.RESNET152,
        'num_channels': 3,
        'multi_image': False
    },
    'Sentinel2_Resnet152_SI_MS': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_si_ms.pth?download=true',
        'backbone': Backbone.RESNET152,
        'num_channels': 9,
        'multi_image': False
    },
    'Sentinel2_Resnet152_MI_RGB': {
        'url': 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_mi_rgb.pth?download=true',
        'backbone': Backbone.RESNET152,
        'num_channels': 3,
        'multi_image': True
    },
}


def adjust_state_dict_prefix(state_dict, needed, prefix=None, prefix_allowed_count=None):
    """
    Adjusts the keys in the state dictionary by replacing 'backbone.backbone' prefix with 'backbone'.

    Args:
        state_dict (dict): Original state dictionary with 'backbone.backbone' prefixes.

    Returns:
        dict: Modified state dictionary with corrected prefixes.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Assure we're only keeping keys that we need for the current model component. 
        if not needed in key:
            continue

        # Update the key prefixes to match what the model expects.
        if prefix is not None:
            while key.count(prefix) > prefix_allowed_count:
                key = key.replace(prefix, '', 1)

        new_state_dict[key] = value
    return new_state_dict

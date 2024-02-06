import collections
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


class NoopTransform(torch.nn.Module):
    def __init__(self):
        super(NoopTransform, self).__init__()

        self.transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            min_size=800,
            max_size=800,
            image_mean=[],
            image_std=[],
        )

    def forward(self, images, targets):
        images = self.transform.batch_images(images, size_divisible=32)
        image_sizes = [(image.shape[1], image.shape[2]) for image in images]
        image_list = torchvision.models.detection.image_list.ImageList(images, image_sizes)
        return image_list, targets

    def postprocess(self, detections, image_sizes, orig_sizes):
        return detections


class FRCNNHead(torch.nn.Module):
    def __init__(self, task, backbone_channels, num_categories=2):
        super(FRCNNHead, self).__init__()

        self.task_type = task
        self.use_layers = list(range(len(backbone_channels)))
        num_channels = backbone_channels[self.use_layers[0]][1]
        featmap_names = ['feat{}'.format(i) for i in range(len(self.use_layers))]
        num_classes = num_categories

        self.noop_transform = NoopTransform()

        # RPN
        anchor_sizes = [[32], [64], [128], [256]]
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = torchvision.models.detection.anchor_utils.AnchorGenerator(anchor_sizes, aspect_ratios)
        rpn_head = torchvision.models.detection.rpn.RPNHead(num_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_pre_nms_top_n = dict(training=2000, testing=2000)
        rpn_post_nms_top_n = dict(training=2000, testing=2000)
        rpn_nms_thresh = 0.7
        self.rpn = torchvision.models.detection.rpn.RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
        )

        # ROI
        box_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=featmap_names, output_size=7, sampling_ratio=2)
        box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(backbone_channels[0][1] * box_roi_pool.output_size[0] ** 2, 1024)
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, num_classes)
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 512
        box_positive_fraction = 0.25
        bbox_reg_weights = None
        box_score_thresh = 0.05
        box_nms_thresh = 0.5
        box_detections_per_img = 100
        self.roi_heads = torchvision.models.detection.roi_heads.RoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        if self.task_type == 'instance':
            # Use Mask R-CNN stuff.
            self.roi_heads.mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=featmap_names, output_size=14, sampling_ratio=2)

            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            self.roi_heads.mask_head = torchvision.models.detection.mask_rcnn.MaskRCNNHeads(backbone_channels[0][1], mask_layers, mask_dilation)

            mask_predictor_in_channels = 256
            mask_dim_reduced = 256
            self.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

    def forward(self, image_list, raw_features, targets=None):
        device = image_list[0].device
        images, targets = self.noop_transform(image_list, targets)

        features = collections.OrderedDict()
        for i, idx in enumerate(self.use_layers):
            features['feat{}'.format(i)] = raw_features[idx]

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        losses = {'base': torch.tensor(0, device=device, dtype=torch.float32)}
        losses.update(proposal_losses)
        losses.update(detector_losses)

        loss = sum(x for x in losses.values())
        return detections, loss


class SimpleHead(torch.nn.Module):
    def __init__(self, task, backbone_channels, num_categories=2):
        super(SimpleHead, self).__init__()

        self.task_type = task 

        use_channels = backbone_channels[0][1]
        num_layers = 2
        self.num_outputs = num_categories
        if self.num_outputs is None:
            if task_type == 'regress':
                self.num_outputs = 1
            else:
                self.num_outputs = 2

        layers = []
        for _ in range(num_layers-1):
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(use_channels, use_channels, 3, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)

        if self.task_type == 'segment':
            layers.append(torch.nn.Conv2d(use_channels, self.num_outputs, 3, padding=1))
            self.loss_func = lambda logits, targets: torch.nn.functional.cross_entropy(logits, targets, reduction='none')

        elif self.task_type == 'bin_segment':
            layers.append(torch.nn.Conv2d(use_channels, self.num_outputs, 3, padding=1))
            def loss_func(logits, targets):
                targets = targets.argmax(dim=1)
                return torch.nn.functional.cross_entropy(logits, targets, reduction='none')[:, None, :, :]
            self.loss_func = loss_func

        elif self.task_type == 'regress':
            layers.append(torch.nn.Conv2d(use_channels, self.num_outputs, 3, padding=1))
            self.loss_func = lambda outputs, targets: torch.square(outputs - targets)

        elif self.task_type == 'classification':
            self.extra = torch.nn.Linear(use_channels, self.num_outputs)
            self.loss_func = lambda logits, targets: torch.nn.functional.cross_entropy(logits, targets, reduction='none')

        elif self.task_type == 'multi-label-classification':
            self.extra = torch.nn.Linear(use_channels, self.num_outputs)
            self.loss_func = lambda logits, targets: torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, image_list, raw_features, targets=None):
        raw_outputs = self.layers(raw_features[0])
        loss = None

        if self.task_type == 'segment':
            outputs = torch.nn.functional.softmax(raw_outputs, dim=1)

            if targets is not None:
                task_targets = torch.stack([target for target in targets], dim=0)
                loss = self.loss_func(raw_outputs, task_targets.long())
                loss = loss.mean()

        elif self.task_type == 'bin_segment':
            outputs = torch.nn.functional.softmax(raw_outputs, dim=1)

            if targets is not None:
                task_targets = torch.stack([target for target in targets], dim=0)
                loss = self.loss_func(raw_outputs, task_targets.float())
                loss = loss.mean()

        elif self.task_type == 'regress':
            raw_outputs = raw_outputs[:, 0, :, :]
            outputs = 255*raw_outputs

            if targets is not None:
                task_targets = torch.stack([target for target in targets], dim=0)
                loss = self.loss_func(raw_outputs, task_targets.float()/255)
                loss = loss.mean()

        elif self.task_type == 'classification':
            features = torch.amax(raw_outputs, dim=(2,3))
            logits = self.extra(features)
            outputs = torch.nn.functional.softmax(logits, dim=1)

            if targets is not None:
                task_targets = targets.to(torch.long)
                loss = self.loss_func(logits, task_targets)
                loss = loss.mean()

        elif self.task_type == 'multi-label-classification':
            features = torch.amax(raw_outputs, dim=(2,3))
            logits = self.extra(features)
            outputs = torch.sigmoid(logits)

            if targets is not None:
                task_targets = torch.cat([target for target in targets], dim=0).to(torch.float32)
                loss = self.loss_func(logits, task_targets)
                loss = loss.mean()

        return outputs, loss


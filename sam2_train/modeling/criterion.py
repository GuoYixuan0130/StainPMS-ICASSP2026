import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_toolbelt.losses import BinaryFocalLoss, DiceLoss

from sam2_train.modeling.matcher import build_matcher
from sam2_train.modeling.utils import get_world_size, is_dist_avail_and_initialized


class MaskIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        pred_prob = torch.sigmoid(pred_mask[:, 0])
        intersection = torch.sum(pred_prob * ground_truth_mask, dim=(1, 2))
        union = (
            torch.sum(pred_prob, dim=(1, 2))
            + torch.sum(ground_truth_mask, dim=(1, 2))
            - intersection
        )
        iou = (intersection + 1e-7) / (union + 1e-7)
        return torch.mean((iou - pred_iou) ** 2)


class Criterion(nn.Module):
    def __init__(
        self,
        num_classes,
        matcher,
        class_weight,
        loss_weight,
        reg_loss_type="l2",
        pms_loss_coef=0.0,
        pms_focal_weight=1.0,
        pms_dice_weight=1.0,
        pms_iou_weight=1.0,
        pms_object_weight=1.0,
        pms_residual_mask_weight=1.0,
        pms_preserve_loss_coef=1.0,
    ):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.reg_loss_type = reg_loss_type

        self.focal_loss = BinaryFocalLoss()
        self.dice_loss = DiceLoss("binary")
        self.iou_loss = MaskIoULoss()

        self.pms_loss_coef = float(pms_loss_coef)
        self.pms_focal_weight = float(pms_focal_weight)
        self.pms_dice_weight = float(pms_dice_weight)
        self.pms_iou_weight = float(pms_iou_weight)
        self.pms_object_weight = float(pms_object_weight)
        self.pms_residual_mask_weight = float(pms_residual_mask_weight)
        self.pms_preserve_loss_coef = float(pms_preserve_loss_coef)

    def loss_reg(self, outputs, targets, indices, num_points):
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs["pred_coords"][idx]
        target_points = torch.cat(
            [gt_points[j] for gt_points, (_, j) in zip(targets["gt_points"], indices)],
            dim=0,
        )
        if self.reg_loss_type == "l2":
            loss_point = F.mse_loss(src_points, target_points, reduction="none")
        else:
            loss_point = F.l1_loss(src_points, target_points, reduction="none")
        return loss_point.sum() / (num_points + 1e-7)

    def loss_cls(self, outputs, targets, indices, num_points):
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs["pred_logits"]
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.long,
            device=src_logits.device,
        )
        target_classes_o = torch.cat(
            [cls[j] for cls, (_, j) in zip(targets["gt_labels"], indices)]
        )
        target_classes[idx] = target_classes_o
        return F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.class_weight)

    def loss_mask(self, outputs, targets, indices, num_points):
        pred_masks = outputs["pred_masks"]
        gt_masks = targets["gt_masks"]
        return self.focal_loss(pred_masks.squeeze(1), gt_masks)

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def forward(self, outputs1, targets1, pred, pred_iou, true2, epoch):
        num_points = sum(targets1["gt_nums"])
        num_points = torch.as_tensor(
            [num_points], dtype=torch.float, device=outputs1["pred_logits"].device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()
        indices = self.matcher(outputs1, targets1)

        loss_dict = {
            "loss_reg": self.loss_reg(outputs1, targets1, indices, num_points) * 20,
            "loss_cls": self.loss_cls(outputs1, targets1, indices, num_points) * 20,
            "loss_mask": self.loss_mask(outputs1, targets1, indices, num_points) * 20,
            "loss_focal": self.dice_loss(pred.unsqueeze(1), true2),
            "loss_dice": self.focal_loss(pred.unsqueeze(1), true2.unsqueeze(1)),
            "loss_iou": self.iou_loss(pred.unsqueeze(1), true2.float(), pred_iou),
        }

        for key in loss_dict:
            loss_dict[key] *= self.loss_weight[key](epoch)
        return loss_dict


def build_criterion(cfg, device):
    class_weight = torch.ones(cfg.data.num_classes + 1, dtype=torch.float).to(device)
    class_weight[-1] = cfg.criterion.eos_coef
    loss_weight = {
        "loss_focal": lambda epoch: cfg.criterion.loss_focal,
        "loss_dice": lambda epoch: cfg.criterion.loss_dice,
        "loss_iou": lambda epoch: cfg.criterion.loss_iou,
        "loss_cls": lambda epoch: cfg.criterion.cls_loss_coef,
        "loss_reg": lambda epoch: cfg.criterion.reg_loss_coef,
        "loss_mask": lambda epoch: cfg.criterion.mask_loss_coef,
    }

    matcher = build_matcher(cfg)
    criterion = Criterion(
        cfg.data.num_classes,
        matcher,
        class_weight=class_weight,
        loss_weight=loss_weight,
        pms_loss_coef=float(getattr(cfg.criterion, "pms_loss_coef", 0.0)),
        pms_focal_weight=float(getattr(cfg.criterion, "pms_focal_weight", 1.0)),
        pms_dice_weight=float(getattr(cfg.criterion, "pms_dice_weight", 1.0)),
        pms_iou_weight=float(getattr(cfg.criterion, "pms_iou_weight", 1.0)),
        pms_object_weight=float(getattr(cfg.criterion, "pms_object_weight", 1.0)),
        pms_residual_mask_weight=float(getattr(cfg.criterion, "pms_residual_mask_weight", 1.0)),
        pms_preserve_loss_coef=float(getattr(cfg.criterion, "pms_preserve_loss_coef", 1.0)),
    )
    return criterion, matcher

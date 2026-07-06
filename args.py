prompter = dict(
    backbone=dict(
        model_name="convnext_xlarge_in22k",
        pretrained=True,
        num_classes=0,
        global_pool="",
    ),
    neck=dict(
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4,
        add_extra_convs="on_input",
    ),
    dropout=0.1,
    space=16,
    hidden_dim=256,
)

input_shape = 256
data = dict(
    num_classes=1,
    train=dict(
        transform=[
            dict(type="RandomCrop", height=256, width=256, p=1),
            dict(type="RandomGridShuffle", grid=(4, 4), p=0.5),
            dict(type="ColorJitter", brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05, p=0.2),
            dict(type="RandomRotate90", p=0.5),
            dict(type="HorizontalFlip", p=0.5),
            dict(type="VerticalFlip", p=0.5),
            dict(type="Downscale", scale_max=0.5, scale_min=0.5, p=0.15),
            dict(type="Blur", blur_limit=9, p=0.2),
            dict(type="GaussNoise", var_limit=50, p=0.25),
            dict(type="ColorJitter", brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05, p=0.2),
            dict(type="Superpixels", p=0.1, p_replace=0.1, n_segments=200, max_size=int(input_shape / 2)),
            dict(type="ZoomBlur", p=0.1, max_factor=1.05),
            dict(type="HorizontalFlip", p=0.5),
            dict(type="VerticalFlip", p=0.5),
            dict(type="ShiftScaleRotate", shift_limit=0.3, scale_limit=0.1, rotate_limit=0, border_mode=0, value=0, p=0.5),
            dict(
                type="PadIfNeeded",
                min_height=None,
                min_width=None,
                pad_height_divisor=prompter["space"],
                pad_width_divisor=prompter["space"],
                position="top_left",
                p=1,
            ),
            dict(type="Normalize"),
        ]
    ),
    val=dict(transform=[dict(type="Normalize")]),
    test=dict(transform=[dict(type="Normalize")]),
    post=dict(iou_threshold=0.5),
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=1e-4,
)

criterion = dict(
    matcher=dict(type="HungarianMatcher", dis_type="l2", set_cost_point=0.1, set_cost_class=1),
    eos_coef=0.25,
    reg_loss_coef=5e-3,
    cls_loss_coef=1.0,
    mask_loss_coef=1.0,
    loss_focal=20,
    loss_dice=1,
    loss_iou=1,

    # Stain candidate mining parameters used by PMS.
    stain_top_k=20,
    stain_min_distance=12,
    stain_open_disk=2,
    stain_sigma=1.0,
    stain_baseline_dilate_radius=5,
    stain_merge_aware=False,
    stain_merge_min_distance=6,
    stain_merge_num_peaks=3,
    hed_alpha=1.0,
    hed_beta=0.0,
    hed_gamma=0.0,

    # Prompt-Mask Supervision.
    pms_loss_coef=0.0,
    pms_gt_match_radius=8,
    pms_focal_weight=1.0,
    pms_dice_weight=1.0,
    pms_iou_weight=1.0,
    pms_residual_mask_weight=1.0,
    pms_preserve_loss_coef=1.0,
    pms_baseline_prompts=False,
    pms_preserve_max_prompts=0,
    pms_object_weight=1.0,

    # ICASSP experimental branch: residual PMS prompts can also supervise the
    # automatic point head. Disabled by default to keep StainPMS unchanged.
    pms_point_loss_coef=0.0,
    pms_point_reg_weight=1.0,
    pms_point_cls_weight=1.0,

    # ICASSP experimental branch: optional soft coverage confidence cache.
    # Used only when --coverage_probabilistic is enabled.
    coverage_prob_threshold=0.6,
    coverage_prob_min_residual=0.05,
)

test = dict(nms_thr=12, match_dis=12, filtering=True)

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="StainPMS / CA-SAM2 training and evaluation"
    )

    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--clip-grad", dest="clip_grad", default=0.1, type=float)
    parser.add_argument("--overlap", default=92, type=int)
    parser.add_argument("--crop_size", default=256, type=int)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--out_size", default=256, type=int)
    parser.add_argument("--b", default=4, type=int, help="SAM2 crop batch size")
    parser.add_argument("--num_workers", default=2, type=int)

    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--eval_on_train",
        action="store_true",
        help="Evaluate the train split with the same sliding-window path used for test evaluation.",
    )
    parser.add_argument("--tta", action="store_true")
    parser.add_argument(
        "--load",
        default="unclockwise",
        choices=["sequence", "unsequence", "clockwise", "unclockwise"],
        help="Sliding-window crop traversal order.",
    )
    parser.add_argument("--texture", action="store_true")
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--texture_memory_bank_size", default=64, type=int)
    parser.add_argument("--context_memory_bank_size", default=100, type=int)
    parser.add_argument("--context_atten_k", default=1, type=int)

    parser.add_argument("--net", default="sam2", choices=["sam2"])
    parser.add_argument("--exp_name", default="StainPMS", type=str)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--val_start_epoch", default=30, type=int)
    parser.add_argument("--val_freq", default=1, type=int)

    parser.add_argument("--lr", default=-1.0, type=float)
    parser.add_argument("--lr_min", default=-1.0, type=float)
    parser.add_argument("--lr_cosine_t_max", default=-1, type=int)
    parser.add_argument("--weight_decay", default=-1.0, type=float)
    parser.add_argument("--lr_milestones", default=[80, 140, 200], nargs="+", type=int)

    parser.add_argument("--gpu", default=True, type=bool)
    parser.add_argument("--gpu_device", default=0, type=int)
    parser.add_argument("--distributed", default="none", type=str)
    parser.add_argument("--dataset", default="monuseg", choices=["monuseg"])
    parser.add_argument("--data_path", default="./data/monuseg", type=str)
    parser.add_argument("--sam_ckpt", default="./checkpoints/sam2_hiera_large.pt", type=str)
    parser.add_argument("--sam_config", default="sam2_hiera_l", type=str)

    parser.add_argument("--test_nms_thr", default=-1, type=int)
    parser.add_argument("--test_filtering", default="", choices=["", "true", "false"])

    parser.add_argument("--use_pms", action="store_true")
    parser.add_argument("--pms_loss_coef", default=-1.0, type=float)
    parser.add_argument("--pms_object_weight", default=-1.0, type=float)
    parser.add_argument("--pms_residual_mask_weight", default=-1.0, type=float)
    parser.add_argument("--pms_preserve_loss_coef", default=-1.0, type=float)
    parser.add_argument("--pms_gt_match_radius", default=-1, type=int)
    parser.add_argument("--pms_baseline_prompts", action="store_true")
    parser.add_argument("--pms_preserve_covered", action="store_true")
    parser.add_argument("--pms_preserve_max_prompts", default=-1, type=int)

    parser.add_argument("--pms_self_bootstrap", action="store_true")
    parser.add_argument("--dump_baseline_masks_dir", default="", type=str)
    parser.add_argument("--baseline_masks_dir", default="", type=str)
    parser.add_argument("--iterative_baseline_refresh_every", default=0, type=int)
    parser.add_argument("--coverage_accumulate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pms_start_epoch", default=0, type=int)

    parser.add_argument("--stain_baseline_dilate_radius", default=-1, type=int)
    parser.add_argument("--stain_min_distance", default=-1, type=int)
    parser.add_argument("--stain_top_k", default=-1, type=int)
    parser.add_argument("--stain_sigma", default=-1.0, type=float)
    parser.add_argument("--stain_merge_aware", action="store_true")
    parser.add_argument("--stain_merge_min_distance", default=-1, type=int)
    parser.add_argument("--stain_merge_num_peaks", default=-1, type=int)

    opt = parser.parse_args()

    return opt

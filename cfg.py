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

    parser.add_argument("--prompt_credit_enabled", action="store_true")
    parser.add_argument("--prompt_credit_grad_scale", default=0.0, type=float)
    parser.add_argument("--prompt_credit_quality_loss_coef", default=0.0, type=float)
    parser.add_argument(
        "--prompt_score_mode",
        default="objectness",
        choices=["objectness", "objectness_x_quality", "quality"],
    )

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
    parser.add_argument(
        "--dump_eval_artifacts_dir",
        default="",
        type=str,
        help=(
            "Directory for rich evaluation artifacts used by StainPQR: "
            "per-image prediction/GT maps plus mask-level assembly metadata."
        ),
    )
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

    parser.add_argument(
        "--stage1_coverage_oracle",
        action="store_true",
        help="Run StainPQR Stage 1B coverage-action decoder oracle and exit.",
    )
    parser.add_argument("--oracle_artifacts_dir", default="", type=str)
    parser.add_argument("--oracle_out_dir", default="", type=str)
    parser.add_argument("--oracle_split", default="test", choices=["test", "train"])
    parser.add_argument("--oracle_max_images", default=0, type=int)
    parser.add_argument("--oracle_coverage_top_k", default=20, type=int)
    parser.add_argument("--oracle_coverage_min_distance", default=12, type=int)
    parser.add_argument("--oracle_coverage_dilate_radius", default=5, type=int)
    parser.add_argument("--oracle_gt_match_radius", default=8, type=int)
    parser.add_argument("--oracle_min_added_area", default=8, type=int)

    parser.add_argument(
        "--stage1_stainroute_oracle",
        action="store_true",
        help="Run Stage 1 ADD/SPLIT oracle only on a frozen train or calibration manifest split.",
    )
    parser.add_argument("--stainroute_split_manifest", default="", type=str)
    parser.add_argument("--stainroute_split", default="router_train", choices=["router_train", "calibration"])
    parser.add_argument("--stainroute_action_config", default="configs/stainroute/stage1_oracle_v1.yaml", type=str)
    parser.add_argument("--stainroute_out_dir", default="", type=str)
    parser.add_argument("--stainroute_baseline_manifest", default="logs/stainroute/stage1/baseline_v1_manifest.json", type=str)
    parser.add_argument("--stainroute_max_images", default=0, type=int)
    parser.add_argument("--stainroute_exact_max_candidates", default=18, type=int)
    parser.add_argument("--stainroute_beam_width", default=64, type=int)
    parser.add_argument("--stainroute_bootstrap_samples", default=2000, type=int)

    parser.add_argument(
        "--stage1_monuseg_futility",
        choices=["runtime_profile", "candidate_audit", "add_pilot"],
        default="",
        help="Run the project-lead-approved MoNuSeg futility gate; router_train only.",
    )
    parser.add_argument(
        "--stainroute_futility_config",
        default="configs/stainroute/monuseg_futility_v1.yaml",
        type=str,
    )
    parser.add_argument("--stainroute_futility_pilot_manifest", default="", type=str)
    parser.add_argument("--stainroute_futility_pilot_batch", choices=["1", "2"], default="1")

    parser.add_argument(
        "--stage2_selective_refine",
        action="store_true",
        help="Run StainPQR Stage 2C selective coverage refinement and exit.",
    )
    parser.add_argument("--selective_artifacts_dir", default="", type=str)
    parser.add_argument("--selective_out_dir", default="", type=str)
    parser.add_argument("--selective_split", default="test", choices=["test", "train"])
    parser.add_argument("--selective_actions_csv", nargs="+", default=[])
    parser.add_argument("--selective_predictions_csv", default="", type=str)
    parser.add_argument("--selective_score", default="selector_prob_iou_area", type=str)
    parser.add_argument("--selective_budget", default=2, type=int)
    parser.add_argument("--selective_min_score", default=-1e30, type=float)

    opt = parser.parse_args()

    return opt

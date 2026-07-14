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

    # Explicit split/crop manifests are deliberately supported at the loader
    # boundary.  In particular, a test directory may contain inaccessible
    # patients, so the dataset must not enumerate it when a manifest is given.
    parser.add_argument("--train_manifest", default="", type=str)
    parser.add_argument("--test_manifest", default="", type=str)
    parser.add_argument("--train_crop_manifest", default="", type=str)
    parser.add_argument("--eval_crop_manifest", default="", type=str)
    # Formal ResiMix runs never infer development roots from ``data_path``:
    # TNBC 7--8 and the MoNuSeg-Lite holdout both live in explicitly admitted
    # roots, so the official test directory is never opened by accident.
    parser.add_argument("--train_image_root", default="", type=str)
    parser.add_argument("--train_label_root", default="", type=str)
    parser.add_argument("--test_image_root", default="", type=str)
    parser.add_argument("--test_label_root", default="", type=str)
    parser.add_argument(
        "--train_only_eval",
        action="store_true",
        help="Construct only a test-transform view of the admitted train split (static coverage only).",
    )
    parser.add_argument("--coverage_manifest", default="", type=str)
    parser.add_argument(
        "--data_identity",
        default="",
        choices=["", "tnbc", "monuseg_lite"],
        help="Activates fail-closed ResiMix split checks for the named formal dataset.",
    )
    parser.add_argument("--allowed_patient_ids", default="", type=str)
    parser.add_argument("--train_allowed_patient_ids", default="", type=str)
    parser.add_argument("--test_allowed_patient_ids", default="", type=str)
    parser.add_argument("--forbidden_patient_ids", default="9,10,11", type=str)

    parser.add_argument("--test_nms_thr", default=-1, type=int)
    parser.add_argument("--test_filtering", default="", choices=["", "true", "false"])

    # Reproducible experiment plumbing.  These switches are inert for the
    # historical CA-SAM2/StainPMS commands unless explicitly supplied.
    parser.add_argument("--artifact_dir", default="", type=str)
    parser.add_argument(
        "--evaluation_epochs",
        default="",
        type=str,
        help="Comma-separated post-initialization evaluation checkpoints, e.g. 0,2,4,6,8,10.",
    )
    parser.add_argument("--save_eval_checkpoints", action="store_true")
    parser.add_argument("--per_image_metrics_path", default="", type=str)

    # ResiMix-PMS is a training-time augmentation only.  Its implementation is
    # activated exclusively by an explicit JSON config; this keeps the default
    # StainPMS pixel path unchanged.
    parser.add_argument("--resimix_config", default="", type=str)
    parser.add_argument("--resimix_enabled", action="store_true")

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

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
    parser.add_argument(
        "--dataset",
        default="monuseg",
        choices=["monuseg", "tnbc"],
        help=(
            "Dataset loader identity. TNBC is manifest-only for Phase 0.5 "
            "smoke runs; it never discovers a directory or selects p9--11."
        ),
    )
    parser.add_argument("--data_path", default="./data/monuseg", type=str)
    parser.add_argument(
        "--train_manifest",
        default="",
        type=str,
        help="Ordered manifest for optimization data; empty preserves the legacy directory loader.",
    )
    parser.add_argument(
        "--eval_manifest",
        default="",
        type=str,
        help="Ordered manifest for development/final evaluation data.",
    )
    parser.add_argument(
        "--verify_manifest_hashes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Verify every image/label SHA256 before constructing a manifest-backed loader.",
    )
    parser.add_argument("--sam_ckpt", default="./checkpoints/sam2_hiera_large.pt", type=str)
    parser.add_argument("--sam_config", default="sam2_hiera_l", type=str)

    parser.add_argument("--test_nms_thr", default=-1, type=int)
    parser.add_argument("--test_filtering", default="", choices=["", "true", "false"])
    parser.add_argument(
        "--evaluator_mode",
        default="legacy_skip",
        choices=["legacy_skip", "strict"],
        help="Empty-case inclusion policy; metric definitions and thresholds are unchanged.",
    )
    parser.add_argument(
        "--metrics_output_dir",
        default="",
        type=str,
        help="Write per-image CSV/JSON and macro summaries for complete-image evaluation.",
    )
    parser.add_argument(
        "--train_only_smoke_steps",
        default=0,
        type=int,
        help=(
            "Run exactly N successful optimizer updates from a manifest-ordered "
            "training loader for a smoke test, then exit."
        ),
    )
    parser.add_argument(
        "--smoke_output",
        default="",
        type=str,
        help="Required JSON output path when --train_only_smoke_steps is non-zero.",
    )
    parser.add_argument(
        "--phase2a_timing_profile",
        default="",
        choices=["", "base", "pms_active"],
        help=(
            "Run the Phase 2A train-only timing protocol and exit. 'base' "
            "measures the pre-PMS objective; 'pms_active' first creates a "
            "train-only coverage cache and measures the active PMS objective."
        ),
    )
    parser.add_argument("--phase2a_timing_output", default="", type=str)
    parser.add_argument("--phase2a_warmup_updates", default=10, type=int)
    parser.add_argument("--phase2a_timed_updates", default=100, type=int)
    parser.add_argument(
        "--phase2a_generic_checkpoint_sha256",
        default="7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b",
        type=str,
        help="Required byte identity of the generic SAM2 initialization in Phase 2A.",
    )
    parser.add_argument(
        "--phase2a_baseline",
        action="store_true",
        help="Run the frozen Phase 2A protocol-clean StainPMS baseline.",
    )
    parser.add_argument("--phase2a_recipe", default="", type=str)
    parser.add_argument("--phase2a_output_dir", default="", type=str)
    parser.add_argument(
        "--phase2a_eval_policy",
        default="",
        choices=["", "tnbc_patient_macro", "none"],
    )
    parser.add_argument("--phase2a_resume_checkpoint", default="", type=str)
    parser.add_argument("--phase2a_budget_gate_report", default="", type=str)

    parser.add_argument(
        "--warmstart_stage",
        default="",
        choices=[
            "",
            "prepare_coverage",
            "smoke",
            "timing",
            "formal_tnbc_5epoch",
            "formal_tnbc_pqbest_ablation_5epoch",
            "formal_tnbc_pqbest_repro_5epoch",
            "formal_tnbc_pqbest_third_seed_5epoch",
            "formal_tnbc_c2_ar_5epoch",
            "formal_tnbc_c2_component_5epoch",
        ],
        help=(
            "Exploratory train-only C0/C1 stage. formal_tnbc_5epoch is the "
            "owner-approved fixed five-epoch TNBC screening run."
        ),
    )
    parser.add_argument(
        "--warmstart_candidate_arm",
        default="",
        choices=["", "legacy", "c0", "c1", "c2_ar", "c2_e", "c2_u", "coverage_only", "quality_only"],
        help=(
            "legacy is an equivalence reference only; c0/c1 share the explicit "
            "four-native-candidate decoder call."
        ),
    )
    parser.add_argument("--warmstart_output", default="", type=str)
    parser.add_argument(
        "--warmstart_dev_manifest",
        default="",
        type=str,
        help="Frozen TNBC p7/p8 manifest used only by PQ-best development evaluation.",
    )
    parser.add_argument(
        "--warmstart_resume_checkpoint",
        default="",
        type=str,
        help=(
            "Formal TNBC C0/C1 recovery checkpoint. Only resumes an interrupted "
            "five-epoch screen after validating its frozen train-only contract."
        ),
    )
    parser.add_argument("--warmstart_checkpoint_sha256", default="", type=str)
    parser.add_argument("--warmstart_coverage_manifest", default="", type=str)
    parser.add_argument(
        "--warmstart_screen_config",
        default="",
        type=str,
        help="Hash-recorded frozen config required by each formal TNBC warm-start run.",
    )
    parser.add_argument(
        "--warmstart_required_free_gib",
        default=0.0,
        type=float,
        help="Frozen C2-only minimum free storage before retaining all five full states.",
    )
    parser.add_argument("--warmstart_smoke_updates", default=0, type=int)
    parser.add_argument("--candidate_coverage_tau", default=0.1, type=float)
    parser.add_argument("--candidate_coverage_coefficient", default=1.0, type=float)
    parser.add_argument("--candidate_quality_coefficient", default=1.0, type=float)
    parser.add_argument("--c2_ar_exclusivity_coefficient", default=0.0, type=float)
    parser.add_argument("--c2_ar_utility_coefficient", default=0.0, type=float)
    parser.add_argument("--c2_ar_neighbor_radius", default=2, type=int)
    parser.add_argument("--c2_ar_match_iou", default=0.5, type=float)
    parser.add_argument(
        "--c2_ar_merge_risk_overlap_fraction", default=0.1, type=float
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

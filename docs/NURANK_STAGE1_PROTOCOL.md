# NuRank Stage 1 Protocol

## Scope and status

NuSet Stage 0 established multimask headroom on the immutable six-image audit: Baseline-Single PQ `0.74999`, Existing-All-Pred PQ `0.75365` (`+0.00366`), and Oracle-All PQ `0.75845` (`+0.00846`). Existing All-Pred recovers `43.32%` of that headroom. Automatic matched prompts have oracle mean `ΔIoU=+0.01817`; a non-token-0 candidate is oracle best for `82.46%` of those prompts. The frozen original IoU head has top-1 groupwise accuracy `19.09%` despite global Spearman `0.62184`.

NuRank Stage 1 therefore trains a small, token-shared ranker only. It does not create masks, train SAM2, train mask tokens, change prompts, change NMS/filtering/assembly, or invoke any terminated StainRoute, PromptCredit-v1, or PromptQ workflow.

## Fixed data and integrity boundaries

- Frozen checkpoint SHA256: `44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781`.
- Seed `3407`; NMS `12`; TTA off; texture/context enabled; unclockwise crop traversal and overlap `32`; inclusive instance matching IoU `>= 0.5`.
- Train cache: TNBC patients 1–6 only. Development cache: patients 7–8 only. Patients 9–11 and MoNuSeg are closed.
- Patients 7–8 are explicitly development data, not an independent leakage-free validation claim, because baseline initialization may have seen them.
- The automated cache resolver names only permitted files under `train_12`; it never enumerates a test path.

All baseline modules are `eval()` and `requires_grad=False`: StainPMS point backbone/heads, image/prompt/mask decoder, multimask tokens/hypernetworks, original IoU head, and memory. The only trainable module is NuRank. Cache construction writes before/after checksums of point model and SAM2; any change is a hard failure.

## One-pass automatic-prompt cache

For each standard automatic prompt group, one standard frozen forward calls `predict_masks()` once and exposes its four existing tokens. No extra image encode, prompt encode, decoder call, prompt perturbation, action enumeration, or GT candidate generation is permitted.

Each group stores four float32 mask logits (to retain exact token-0 replay), four float16 `mask_tokens_out` embeddings, original predicted IoU, coordinates, objectness, class/prompt/crop IDs, and all non-GT morphology values: soft/hard area fraction, stability, mean probability, mean absolute logit, boundary-band entropy, and point-in-hard-mask. The cache records four hard/soft IoU targets afterwards. A point inside a GT gets that instance; an unmatched point receives maximum mask IoU over pre-existing crop GT instances. GT never removes, moves, or adds a prompt. Cache manifests record source hashes, group order, four-token shape checks, quantization errors, frozen checksums, call counts, and the 10% time forecast. A forecast above the fixed six GPU-hour Stage-1 cap stops the run.

## NuRank model and fixed objective

The same ranker is applied to all four tokens in a prompt group. It receives the 256-d mask token after `LayerNorm`, then eight continuous values: original predicted IoU plus the seven stored morphology values. The continuous inputs use mean/std estimated from train cache only. Architecture is `Linear(264,128) → GELU → Linear(128,1) → Sigmoid`, with no dropout and no token-ID embedding; parameter count must stay below 0.1M.

For hard-IoU target `r_ik` and ranker score `q_ik`, the fixed loss is:

`L = SmoothL1(q_ik, r_ik) + mean[max(0, |d_kl| - sign(d_kl)(q_ik-q_il))]`.

The pair term averages valid unordered pairs only and excludes `|d_kl| < 1e-3`. There are no tunable weights, temperatures, auxiliary losses, threshold changes, or early stopping. Offline training uses AdamW (`lr=1e-3`, weight decay `1e-4`), batch 256 groups (only power-of-two fallback for OOM), seed 3407, exactly 30 epochs, and fixed epoch-30 checkpoint. Development curves are diagnostics only.

## Development replay and analysis

The immutable development cache is replayed through four fixed selections: token 0, existing original-IoU argmax, NuRank argmax, and oracle hard-IoU argmax. All use identical automatic prompts, point NMS, filtering, edge penalty, bbox NMS, overlap processing, assembly, and evaluator. Unmatched prompts remain and use their predefined maximum-IoU target only for oracle analysis.

It writes per-prompt ranking records, confusion counts `(oracle, existing, NuRank)`, changed-winner utility statistics, per-image Dice/AJI/AJI+/DQ/SQ/PQ/TP/FP/FN, token histograms, 2,000-resample image-level paired bootstrap, call counts, ranker timing, and output checksums. Replay invokes no frozen model; reported frozen-model call counts therefore come from cache extraction and are identical across paths.

## Preregistered decision

STRONG GO requires all: NuRank `ΔPQ` vs Single `>= +0.005`; NuRank vs Existing-All-Pred `>= +0.0015`; oracle recovery `>=60%`; top-1 improvement `>=15` points; mean regret reduction `>=35%`; AJI non-decrease; at least 5/7 images PQ non-decreasing; largest positive contribution `<=60%`; no unmatched-FP increase; and runtime overhead `<=5%`.

CONDITIONAL covers `ΔPQ` in `[+0.003,+0.005)`, smaller but positive improvement over Existing-All-Pred, substantial ranking improvement with limited assembly gain, or concentration in few images. NO-GO includes oracle headroom `<+0.003`, NuRank `ΔPQ <+0.003`, failure to beat Existing-All-Pred, no regret/PQ improvement, AJI decrease, clear FP growth, any need to retune NMS/threshold/loss, or time-budget breach. No result authorizes TNBC test or MoNuSeg automatically.

## Authorized AutoDL sequence

Choose a previously unused run ID and run the commands below exactly once; the first command creates the immutable stage directory. Redirect the test output and command output to files inside that directory so they become part of the artifact.

```bash
RUN=logs/nurank/stage1_tnbc_dev/<run_id>
python -m unittest discover -s tests/nurank -v
python tools/build_nurank_cache.py --data-root ./data/tnbc --checkpoint ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth --stage-dir "$RUN" --role train
python tools/build_nurank_cache.py --data-root ./data/tnbc --checkpoint ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth --stage-dir "$RUN" --role development
python tools/train_nurank.py --stage-dir "$RUN"
python tools/evaluate_nurank.py --stage-dir "$RUN" --data-root ./data/tnbc --checkpoint ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth
```

The evaluation tool reruns the NuRank unit suite and saves it as `tests.txt` before finalizing `SHA256SUMS`. If the 10% cache forecast exceeds six GPU hours, preserve the partial directory and stop. Do not rerun with fewer images, altered split, a second seed, changed loss, or changed threshold.

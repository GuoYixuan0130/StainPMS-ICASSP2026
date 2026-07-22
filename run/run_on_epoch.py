import hashlib
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from run.utils import vis_inst_image
from sam2_train.modeling.stats_utils import *
from sam2_train.modeling.utils import *
from stainpms.candidate_coverage import aggregate_candidate_prompt_groups
from stainpms.evaluator import evaluate_instance_pair, write_evaluation_outputs


def _safe_image_name(name):
    if isinstance(name, (list, tuple)):
        return "_".join(str(item) for item in name)
    return str(name)


def _as_jsonable(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    return value


def _ori_hw(ori_shape):
    if torch.is_tensor(ori_shape):
        arr = ori_shape.detach().cpu().numpy()
    else:
        arr = np.asarray(ori_shape)
    arr = arr.reshape(-1)
    return int(arr[0]), int(arr[1])


def _refine_sam_masks(cfgs, low_res_multimasks):
    return F.interpolate(
        low_res_multimasks,
        size=(cfgs.out_size, cfgs.out_size),
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def _decode_training_candidates(
    cfgs,
    net,
    *,
    image_embed,
    sparse_prompt_embeddings,
    dense_prompt_embeddings,
    cell_nums,
    high_res_feats,
    runtime_stats,
):
    """Use one common four-token decoder call for approved C0 and C1 arms.

    Legacy StainPMS computes all four native masks inside ``predict_masks`` and
    then returns token 0 when ``multimask_output=False`` during training.  The
    warm-start path makes that mapping explicit while retaining all candidates
    for the optional C1 auxiliary objective.
    """
    arm = str(getattr(cfgs, "warmstart_candidate_arm", "") or "").lower()
    common_four_candidate_path = arm in {"c0", "c1"}
    if common_four_candidate_path:
        all_masks, all_quality, _, object_logits = net.sam_mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=net.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=False,
            cell_nums=cell_nums,
            high_res_features=high_res_feats,
        )
        if all_masks.ndim != 4 or all_masks.shape[1] != 4:
            raise RuntimeError(
                f"warm-start candidate path expected four masks, got {tuple(all_masks.shape)}"
            )
        if all_quality.shape != all_masks.shape[:2]:
            raise RuntimeError(
                "warm-start quality shape does not match native mask candidates: "
                f"{tuple(all_quality.shape)} vs {tuple(all_masks.shape[:2])}"
            )
        if runtime_stats is not None:
            runtime_stats["native_candidate_decoder_calls"] = int(
                runtime_stats.get("native_candidate_decoder_calls", 0)
            ) + 1
            runtime_stats["native_candidate_prompt_count"] = int(
                runtime_stats.get("native_candidate_prompt_count", 0)
            ) + int(all_masks.shape[0])
            runtime_stats["native_mask_token_count"] = 4
            runtime_stats["original_supervised_mask_token"] = 0
        return (
            all_masks[:, 0:1],
            all_quality[:, 0:1],
            object_logits,
            all_masks,
            all_quality,
        )

    selected_masks, selected_quality, _, object_logits = net.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=net.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_prompt_embeddings,
        dense_prompt_embeddings=dense_prompt_embeddings,
        multimask_output=False,
        repeat_image=False,
        cell_nums=cell_nums,
        high_res_features=high_res_feats,
    )
    return selected_masks, selected_quality, object_logits, None, None


def _l2_gradient_norm(parameters):
    squared = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        squared += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(squared) if found else 0.0


def _small_gradient_snapshot(named_parameters, *, maximum_elements=4096):
    candidates = []
    for name, parameter in named_parameters:
        if parameter.grad is None or parameter.numel() > maximum_elements:
            continue
        candidates.append((parameter.numel(), name, parameter))
    if not candidates:
        return None
    _, name, parameter = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    values = parameter.grad.detach().float().reshape(-1).cpu()
    return {
        "name": name,
        "shape": list(parameter.shape),
        "values": values.tolist(),
        "l2_norm": float(torch.linalg.vector_norm(values)),
        "max_abs": float(values.abs().max()) if values.numel() else 0.0,
    }


def _record_gradient_audit(runtime_stats, point_net, net):
    if runtime_stats is None:
        return
    decoder = net.sam_mask_decoder
    groups = {
        "point_head": _l2_gradient_norm(point_net.parameters()),
        "mask_decoder": _l2_gradient_norm(decoder.parameters()),
        "quality_head": _l2_gradient_norm(decoder.iou_prediction_head.parameters()),
    }
    audit = runtime_stats.setdefault(
        "gradient_audit",
        {"step_count": 0, "group_l2_sum": {}, "group_l2_max": {}, "key_gradients": {}},
    )
    audit["step_count"] += 1
    for name, value in groups.items():
        audit["group_l2_sum"][name] = float(audit["group_l2_sum"].get(name, 0.0)) + value
        audit["group_l2_max"][name] = max(
            float(audit["group_l2_max"].get(name, 0.0)), value
        )
    if not audit["key_gradients"]:
        snapshots = {
            "point_head": _small_gradient_snapshot(point_net.named_parameters()),
            "mask_token_embedding": _small_gradient_snapshot(
                [("mask_tokens.weight", decoder.mask_tokens.weight)]
            ),
            "quality_head": _small_gradient_snapshot(
                (
                    (f"iou_prediction_head.{name}", parameter)
                    for name, parameter in decoder.iou_prediction_head.named_parameters()
                )
            ),
        }
        audit["key_gradients"] = {
            name: value for name, value in snapshots.items() if value is not None
        }


def _record_candidate_loss_audit(
    runtime_stats,
    *,
    stainpms_loss,
    coverage_loss,
    quality_loss,
    weighted_coverage,
    weighted_quality,
    group_audit,
):
    if runtime_stats is None:
        return
    total = stainpms_loss + weighted_coverage + weighted_quality
    extra = weighted_coverage + weighted_quality
    audit = runtime_stats.setdefault(
        "candidate_loss_audit",
        {
            "step_count": 0,
            "stainpms_loss_sum": 0.0,
            "coverage_loss_sum": 0.0,
            "quality_loss_sum": 0.0,
            "weighted_extra_sum": 0.0,
            "total_loss_sum": 0.0,
            "extra_to_total_ratio_sum": 0.0,
            "groups": {},
        },
    )
    audit["step_count"] += 1
    values = {
        "stainpms_loss_sum": stainpms_loss,
        "coverage_loss_sum": coverage_loss,
        "quality_loss_sum": quality_loss,
        "weighted_extra_sum": extra,
        "total_loss_sum": total,
    }
    for key, value in values.items():
        audit[key] += float(value.detach().float().cpu())
    ratio = extra.detach().float() / total.detach().float().abs().clamp_min(1e-12)
    audit["extra_to_total_ratio_sum"] += float(ratio.cpu())
    for name, record in group_audit.items():
        target = audit["groups"].setdefault(
            name,
            {
                "valid_prompt_count": 0,
                "coverage_prompt_weighted_sum": 0.0,
                "quality_prompt_weighted_sum": 0.0,
                "best_softmin_weight_prompt_weighted_sum": 0.0,
                "effective_candidate_count_prompt_weighted_sum": 0.0,
                "alpha": float(record["alpha"]),
            },
        )
        count = int(record["valid_prompt_count"])
        target["valid_prompt_count"] += count
        target["coverage_prompt_weighted_sum"] += float(record["coverage_mean"]) * count
        target["quality_prompt_weighted_sum"] += float(record["quality_mean"]) * count
        if count and record["best_softmin_gradient_weight_mean"] is not None:
            target["best_softmin_weight_prompt_weighted_sum"] += (
                float(record["best_softmin_gradient_weight_mean"]) * count
            )
        if count and record["effective_candidate_count_mean"] is not None:
            target["effective_candidate_count_prompt_weighted_sum"] += (
                float(record["effective_candidate_count_mean"]) * count
            )


def _dump_eval_artifacts(
    dump_dir,
    image_name,
    gt_inst_map,
    pred_inst_map,
    candidate_records,
    selected_records,
):
    os.makedirs(dump_dir, exist_ok=True)
    stem = os.path.join(dump_dir, image_name)
    np.save(stem + "_gt.npy", np.asarray(gt_inst_map).astype(np.int32))
    np.save(stem + "_pred.npy", np.asarray(pred_inst_map).astype(np.int32))
    meta = {
        "image_name": image_name,
        "shape": list(np.asarray(pred_inst_map).shape),
        "num_candidates": len(candidate_records),
        "num_selected": len(selected_records),
        "candidates": candidate_records,
        "selected": selected_records,
    }
    with open(stem + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(_as_jsonable(meta), f, indent=2)


def _assemble_instance_map(
    all_boxes,
    all_scores,
    all_masks,
    all_inds,
    inst_shape,
    iou_threshold,
    all_records=None,
    return_records=False,
):
    if len(all_masks) == 0:
        empty = np.zeros(inst_shape, dtype=int)
        if return_records:
            return empty, []
        return empty

    all_boxes = torch.as_tensor(all_boxes)
    all_scores = torch.as_tensor(all_scores)
    all_inds = np.asarray(all_inds)
    unique_inds, counts = np.unique(all_inds, return_counts=True)

    keep_prior = np.ones(len(all_inds), dtype=bool)
    for i in np.where(counts > 1)[0]:
        inds = np.where(all_inds == unique_inds[i])[0]
        inds = np.delete(inds, np.argmax(all_scores[inds]))
        keep_prior[inds] = False
    keep_prior_t = torch.from_numpy(keep_prior)

    kept_orig_indices = np.where(keep_prior)[0]
    all_boxes = all_boxes[keep_prior_t]
    all_scores = all_scores[keep_prior_t]
    all_masks = [all_masks[ind] for ind in kept_orig_indices]

    if len(all_boxes.shape) == 1:
        cross_categories = torch.zeros_like(all_boxes)
    else:
        cross_categories = torch.zeros_like(all_boxes[:, 0])
    keep_by_nms = batched_nms(
        all_boxes.float(),
        all_scores,
        cross_categories,
        iou_threshold=iou_threshold,
    ).numpy()

    inst_map = np.zeros(inst_shape, dtype=int)
    selected_records = []
    for iid, ind in enumerate(keep_by_nms[::-1]):
        if inst_map[all_masks[ind]].all() == 0:
            inst_map[all_masks[ind]] = iid + 1
            if return_records and all_records is not None:
                source_idx = int(kept_orig_indices[int(ind)])
                record = dict(all_records[source_idx])
                record["source_candidate_index"] = source_idx
                record["final_id"] = int(iid + 1)
                record["final_area"] = int(np.asarray(all_masks[ind]).sum())
                selected_records.append(record)
    if return_records:
        return inst_map, selected_records
    return inst_map


def _accumulate_coverage(prev_map, new_map, overlap_thr=0.5):
    prev = np.asarray(prev_map).astype(np.int32)
    new = np.asarray(new_map).astype(np.int32)
    if prev.shape != new.shape:
        return new

    out = prev.copy()
    prev_bin = prev > 0
    next_id = int(out.max()) + 1 if out.size else 1
    for nid in np.unique(new):
        if nid == 0:
            continue
        mask = new == nid
        area = int(mask.sum())
        if area == 0:
            continue
        if int((mask & prev_bin).sum()) / area >= overlap_thr:
            continue
        add_region = mask & (~prev_bin)
        if int(add_region.sum()) == 0:
            continue
        out[add_region] = next_id
        next_id += 1
    return out


def _append_metric_scores(
    inst_map,
    pred_map,
    score_lists,
    *,
    evaluator_mode="legacy_skip",
    sample_id=None,
):
    record = evaluate_instance_pair(
        inst_map,
        pred_map,
        mode=evaluator_mode,
        match_iou=0.5,
        sample_id=sample_id,
    )
    score_lists.setdefault("_records", []).append(record)
    if not record["included_in_macro"]:
        return
    for name, value in record["metrics"].items():
        score_lists[name].append(value)


def _mean_metric_tuple(score_lists):
    def mean_or_nan(values):
        return float(np.nanmean(values)) if values else float("nan")

    return (
        mean_or_nan(score_lists["dice1"]),
        mean_or_nan(score_lists["dice2"]),
        mean_or_nan(score_lists["aji"]),
        mean_or_nan(score_lists["aji_p"]),
        mean_or_nan(score_lists["dq"]),
        mean_or_nan(score_lists["sq"]),
        mean_or_nan(score_lists["pq"]),
    )


def find_nearest_points(pred_coords, points_choose):
    nearest_points = []
    for i in range(pred_coords.shape[0]):
        pred_points = pred_coords[i].float()
        chosen_points = points_choose[i].view(-1, 2).float()
        distances = torch.cdist(pred_points.unsqueeze(0), chosen_points.unsqueeze(0)).squeeze(0)
        nearest_indices = torch.argmin(distances, dim=0)
        nearest_points.append(pred_points[nearest_indices].unsqueeze(1))
    return nearest_points


def train_on_epoch(
    cfgs,
    point_net,
    point_encoder,
    net,
    train_loader,
    criterion,
    optimizer,
    epoch,
    texture_memory_bank_list,
    device,
    runtime_stats=None,
    max_optimizer_steps=None,
):
    if max_optimizer_steps is not None and int(max_optimizer_steps) <= 0:
        raise ValueError("max_optimizer_steps must be positive when specified")
    if runtime_stats is not None:
        runtime_stats.setdefault("images_seen", 0)
        runtime_stats.setdefault("crop_batches_seen", 0)
        runtime_stats.setdefault("crops_seen", 0)
        runtime_stats.setdefault("optimizer_steps", 0)
        runtime_stats.setdefault("shape_skips", 0)
        runtime_stats.setdefault("nonfinite_loss_skips", 0)
        runtime_stats.setdefault("nonfinite_gradient_skips", 0)
        runtime_stats.setdefault("native_candidate_decoder_calls", 0)
        runtime_stats.setdefault("native_candidate_prompt_count", 0)
        runtime_stats.setdefault("no_prompt_batch_count", 0)
        if runtime_stats.get("record_no_prompt_batches", False):
            runtime_stats.setdefault("no_prompt_batches", [])
    point_net.train()
    net.train()
    criterion.train()
    optimizer.zero_grad()

    log_info = {}
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    feat_sizes = [(64, 64), (32, 32), (16, 16)]

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch}", unit="img") as pbar:
        stop_after_step_limit = False
        for data_iter_step, batch in enumerate(
            metric_logger.log_every(train_loader, cfgs.print_freq, header)
        ):
            if runtime_stats is not None:
                runtime_stats["images_seen"] += 1
            (
                images_lists,
                inst_masks_lists,
                points_choose_lists,
                labels_choose_lists,
                points_lists,
                labels_lists,
                cell_nums_lists,
                masks_lists,
                ori_shape_lists,
                xs_lists,
                ys_lists,
                b_coords_lists,
                b_weights_lists,
                b_gt_masks_lists,
                b_neg_coords_lists,
                b_preserve_counts_lists,
            ) = batch
            context_memory_bank_list = []

            k_crops = images_lists.size(0)
            cumulative_sums = np.cumsum(cell_nums_lists)
            for start_idx in range(0, k_crops, cfgs.b):
                if (
                    max_optimizer_steps is not None
                    and runtime_stats is not None
                    and runtime_stats["optimizer_steps"] >= int(max_optimizer_steps)
                ):
                    stop_after_step_limit = True
                    break
                if runtime_stats is not None:
                    runtime_stats["crop_batches_seen"] += 1
                end_idx = min(start_idx + cfgs.b, k_crops)
                if runtime_stats is not None:
                    runtime_stats["crops_seen"] += int(end_idx - start_idx)
                start_cell = 0 if start_idx == 0 else cumulative_sums[start_idx - 1]
                end_cell = cumulative_sums[end_idx - 1]

                images = images_lists[start_idx:end_idx]
                inst_masks = inst_masks_lists[start_cell:end_cell]
                points_choose = points_choose_lists[start_idx:end_idx]
                labels_choose = labels_choose_lists[start_idx:end_idx]
                points_list = points_lists[start_idx:end_idx]
                labels_list = labels_lists[start_idx:end_idx]
                cell_nums = cell_nums_lists[start_idx:end_idx]
                masks = masks_lists[start_idx:end_idx]
                ori_shape = ori_shape_lists[start_idx:end_idx]
                xs = xs_lists[start_idx:end_idx]
                ys = ys_lists[start_idx:end_idx]
                b_coords_batch = b_coords_lists[start_idx:end_idx]
                b_gt_masks_batch = b_gt_masks_lists[start_idx:end_idx]
                b_neg_coords_batch = b_neg_coords_lists[start_idx:end_idx]
                b_preserve_counts_batch = b_preserve_counts_lists[start_idx:end_idx]

                imgs = images.to(device)
                masks = masks.to(device).float()
                targets = {
                    "gt_masks": masks,
                    "gt_nums": [len(points) for points in points_list],
                    "gt_points": [points.view(-1, 2).to(device).float() for points in points_list],
                    "gt_labels": [labels.to(device).long() for labels in labels_list],
                    "b_coords": b_coords_batch,
                    "b_gt_masks": b_gt_masks_batch,
                    "b_preserve_counts": b_preserve_counts_batch,
                }

                feats, _ = point_encoder(imgs)
                backbone_out, _ = net.forward_image(imgs, feats)
                _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
                memfeatures = vision_feats
                memfeatures_pos = vision_pos_embeds

                outputs1, _, _, _ = point_net(imgs)
                prompt_labels = torch.cat(labels_choose).to(device)
                nearest_points = find_nearest_points(outputs1["pred_coords"].cpu(), points_choose)
                nearest_points_cat = torch.cat([nearest_points[i] for i in range(len(nearest_points))]).to(device)
                cell_nums = cell_nums.to(device)

                if nearest_points_cat.shape[0] == 0:
                    print("[skip] no prompts in batch")
                    if runtime_stats is not None:
                        runtime_stats["no_prompt_batch_count"] += 1
                        if runtime_stats.get("record_no_prompt_batches", False):
                            position = {
                                "epoch_index": int(epoch),
                                "image_loader_index": int(data_iter_step),
                                "crop_start_index": int(start_idx),
                                "crop_end_index": int(end_idx),
                                "global_crop_batch_index": int(runtime_stats["crop_batches_seen"]) - 1,
                            }
                            canonical = json.dumps(
                                position, sort_keys=True, separators=(",", ":")
                            ).encode("utf-8")
                            position["index_sha256"] = hashlib.sha256(canonical).hexdigest()
                            runtime_stats["no_prompt_batches"].append(position)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                batch_size = vision_feats[-1].size(1)
                if cfgs.context:
                    vision_feats, vision_pos_embeds = context_memory_attention(
                        context_memory_bank_list,
                        vision_feats,
                        vision_pos_embeds,
                        xs,
                        ys,
                        net,
                        feat_sizes,
                        cfgs.context_atten_k,
                    )

                if cfgs.texture:
                    if len(texture_memory_bank_list) == 0:
                        zero = torch.zeros(1, batch_size, net.hidden_dim, device=device)
                        vision_feats[-1] = vision_feats[-1] + zero
                        vision_pos_embeds[-1] = vision_pos_embeds[-1] + zero
                    else:
                        to_cat_memory = []
                        to_cat_memory_pos = []
                        to_cat_image_embed = []
                        for element in texture_memory_bank_list:
                            to_cat_memory.append(element[0].to(device, non_blocking=True).flatten(2).permute(2, 0, 1))
                            to_cat_memory_pos.append(element[1].to(device, non_blocking=True).flatten(2).permute(2, 0, 1))
                            to_cat_image_embed.append(element[3].to(device, non_blocking=True))

                        memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                        memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                        image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

                        vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(batch_size, -1, 64, 64)
                        vision_feats_temp = vision_feats_temp.reshape(batch_size, -1)
                        image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                        vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                        similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
                        similarity_scores = F.softmax(similarity_scores, dim=1)
                        sampled_indices = torch.topk(similarity_scores, batch_size, dim=1).indices.squeeze(1)

                        memory_stack_new = memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
                        memory = memory_stack_new.reshape(-1, memory_stack_new.size(2), memory_stack_new.size(3))
                        memory_pos_stack_new = memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
                        memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_new.size(2), memory_stack_new.size(3))
                        vision_feats[-1], vision_pos_embeds[-1] = net.memory_attention(
                            state="texture",
                            curr=[vision_feats[-1]],
                            curr_pos=[vision_pos_embeds[-1]],
                            memory=memory,
                            memory_pos=memory_pos,
                            num_obj_ptr_tokens=0,
                        )

                feats = [
                    feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
                ][::-1]
                image_embed = feats[-1]
                high_res_feats = feats[:-1]

                with torch.no_grad():
                    se, de = net.sam_prompt_encoder(
                        points=(nearest_points_cat, prompt_labels),
                        boxes=None,
                        masks=None,
                        batch_size=batch_size,
                    )

                (
                    low_res_multimasks,
                    iou_predictions,
                    _,
                    all_standard_low_res_masks,
                    all_standard_quality,
                ) = _decode_training_candidates(
                    cfgs,
                    net,
                    image_embed=image_embed,
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    cell_nums=cell_nums,
                    high_res_feats=high_res_feats,
                    runtime_stats=runtime_stats,
                )
                values, _ = torch.max(iou_predictions, dim=1)
                values_list = torch.split(values, cell_nums.tolist())
                mean_iou_predictions = [part.mean() for part in values_list]
                pred = _refine_sam_masks(cfgs, low_res_multimasks)

                points_split = torch.split(nearest_points_cat, cell_nums.tolist())
                pred_split = torch.split(pred, cell_nums.tolist())
                inst_pred_list = []
                for i in range(len(cell_nums)):
                    inst_pred = combine_mask(
                        ori_shape,
                        points_split[i],
                        pred_split[i],
                        values_list[i],
                    )
                    inst_pred_list.append(torch.from_numpy(inst_pred))
                high_res_multimasks = torch.stack(inst_pred_list, dim=0).to(torch.float32).unsqueeze(1).to(device)

                maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                    current_vision_feats=vision_feats,
                    feat_sizes=feat_sizes,
                    pred_masks_high_res=high_res_multimasks,
                    is_mask_from_pts=True,
                )
                maskmem_features = maskmem_features.to(device=device, non_blocking=True)
                maskmem_pos_enc = maskmem_pos_enc[0].to(device=device, non_blocking=True)

                if cfgs.texture:
                    if len(texture_memory_bank_list) < cfgs.texture_memory_bank_size:
                        for batch_idx in range(maskmem_features.size(0)):
                            texture_memory_bank_list.append(
                                [
                                    maskmem_features[batch_idx].unsqueeze(0).detach(),
                                    maskmem_pos_enc[batch_idx].unsqueeze(0).detach(),
                                    mean_iou_predictions[batch_idx],
                                    image_embed[batch_idx].reshape(-1).detach(),
                                ]
                            )
                    else:
                        for batch_idx in range(maskmem_features.size(0)):
                            bank_flat = [element[0].reshape(-1) for element in texture_memory_bank_list]
                            bank_flat = torch.stack(bank_flat)
                            bank_norm = F.normalize(bank_flat, p=2, dim=1)
                            current_similarity = torch.mm(bank_norm, bank_norm.t())
                            current_similarity_no_diag = current_similarity.clone()
                            diag_indices = torch.arange(current_similarity_no_diag.size(0))
                            current_similarity_no_diag[diag_indices, diag_indices] = float("-inf")
                            single_key = F.normalize(maskmem_features[batch_idx].reshape(-1), p=2, dim=0).unsqueeze(1)
                            similarity_scores = torch.mm(bank_norm, single_key).squeeze()
                            min_similarity_index = torch.argmin(similarity_scores)
                            max_similarity_index = torch.argmax(current_similarity_no_diag[min_similarity_index])
                            if similarity_scores[min_similarity_index] < current_similarity_no_diag[min_similarity_index][max_similarity_index]:
                                if mean_iou_predictions[batch_idx] > texture_memory_bank_list[max_similarity_index][2] - 0.1:
                                    texture_memory_bank_list.pop(max_similarity_index)
                                    texture_memory_bank_list.append(
                                        [
                                            maskmem_features[batch_idx].unsqueeze(0).detach(),
                                            maskmem_pos_enc[batch_idx].unsqueeze(0).detach(),
                                            mean_iou_predictions[batch_idx],
                                            image_embed[batch_idx].reshape(-1).detach(),
                                        ]
                                    )

                if cfgs.context:
                    if len(context_memory_bank_list) + batch_size <= cfgs.context_memory_bank_size:
                        for batch_idx in range(batch_size):
                            context_memory_bank_list.append(
                                [
                                    memfeatures[-1][:, batch_idx:batch_idx + 1, :].detach(),
                                    memfeatures_pos[-1][:, batch_idx:batch_idx + 1, :].detach(),
                                    xs[batch_idx],
                                    ys[batch_idx],
                                ]
                            )

                gt_inst_masks = inst_masks.to(torch.float32).to(device)
                if pred.shape[0] != gt_inst_masks.shape[0]:
                    print(
                        f"[skip] pred.shape[0]={pred.shape[0]} != "
                        f"gt_inst_masks.shape[0]={gt_inst_masks.shape[0]}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if runtime_stats is not None:
                        runtime_stats["shape_skips"] += 1
                    continue

                loss_dict = criterion(outputs1, targets, pred, values, gt_inst_masks, epoch)
                candidate_groups = None
                if str(getattr(cfgs, "warmstart_candidate_arm", "")).lower() == "c1":
                    if all_standard_low_res_masks is None or all_standard_quality is None:
                        raise RuntimeError("C1 requires the common four-candidate decoder path")
                    all_standard_masks = F.interpolate(
                        all_standard_low_res_masks,
                        size=(cfgs.out_size, cfgs.out_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    candidate_groups = {
                        "ordinary": {
                            "candidate_logits": all_standard_masks,
                            "quality_predictions": all_standard_quality,
                            "gt_masks": gt_inst_masks,
                            "alpha": 1.0,
                        },
                        "pms_residual": {
                            "candidate_logits": all_standard_masks.new_empty(
                                (0, 4, cfgs.out_size, cfgs.out_size)
                            ),
                            "quality_predictions": all_standard_quality.new_empty((0, 4)),
                            "gt_masks": gt_inst_masks.new_empty(
                                (0, cfgs.out_size, cfgs.out_size)
                            ),
                            "alpha": float(criterion.pms_loss_coef)
                            * float(criterion.pms_residual_mask_weight),
                        },
                        "pms_preservation": {
                            "candidate_logits": all_standard_masks.new_empty(
                                (0, 4, cfgs.out_size, cfgs.out_size)
                            ),
                            "quality_predictions": all_standard_quality.new_empty((0, 4)),
                            "gt_masks": gt_inst_masks.new_empty(
                                (0, cfgs.out_size, cfgs.out_size)
                            ),
                            "alpha": float(criterion.pms_preserve_loss_coef),
                        },
                    }

                pms_coef = getattr(criterion, "pms_loss_coef", 0.0)
                if getattr(cfgs, "use_pms", False) and pms_coef > 0 and len(b_coords_batch) == batch_size:
                    pos_counts = [int(bc.shape[0]) for bc in b_coords_batch]
                    neg_counts = [int(bn.shape[0]) for bn in b_neg_coords_batch]
                    n_pos = sum(pos_counts)
                    n_neg = sum(neg_counts)
                    total_b = n_pos + n_neg
                    zero = (outputs1["pred_coords"].sum() + outputs1["pred_logits"].sum()) * 0.0
                    loss_dict["loss_pms_focal"] = zero
                    loss_dict["loss_pms_dice"] = zero
                    loss_dict["loss_pms_iou"] = zero
                    loss_dict["loss_pms_preserve_focal"] = zero
                    loss_dict["loss_pms_preserve_dice"] = zero
                    loss_dict["loss_pms_preserve_iou"] = zero
                    loss_dict["loss_pms_object"] = zero

                    if total_b > 0:
                        per_image_coords = []
                        per_image_total = []
                        per_image_pos_mask = []
                        per_image_preserve_mask = []
                        if torch.is_tensor(b_preserve_counts_batch):
                            preserve_counts = [int(v.item()) for v in b_preserve_counts_batch]
                        else:
                            preserve_counts = [int(v) for v in b_preserve_counts_batch]

                        for i in range(batch_size):
                            pos_i = b_coords_batch[i].float()
                            neg_i = b_neg_coords_batch[i].float()
                            n_pos_i = int(pos_i.shape[0])
                            n_neg_i = int(neg_i.shape[0])
                            n_preserve_i = min(max(int(preserve_counts[i]), 0), n_pos_i)
                            n_residual_i = n_pos_i - n_preserve_i
                            cat_i = torch.cat([pos_i, neg_i], dim=0)
                            per_image_coords.append(cat_i)
                            per_image_total.append(int(cat_i.shape[0]))
                            per_image_pos_mask.append(
                                torch.cat(
                                    [
                                        torch.ones(n_pos_i, dtype=torch.float32),
                                        torch.zeros(n_neg_i, dtype=torch.float32),
                                    ]
                                )
                            )
                            per_image_preserve_mask.append(
                                torch.cat(
                                    [
                                        torch.zeros(n_residual_i, dtype=torch.float32),
                                        torch.ones(n_preserve_i, dtype=torch.float32),
                                        torch.zeros(n_neg_i, dtype=torch.float32),
                                    ]
                                )
                            )

                        b_coords_cat = torch.cat(per_image_coords).to(device).unsqueeze(1)
                        b_labels_cat = torch.ones(total_b, 1, dtype=torch.long, device=device)
                        b_cell_nums = torch.tensor(per_image_total, dtype=torch.long, device=device)
                        is_pos_cat = torch.cat(per_image_pos_mask).to(device)
                        is_preserve_cat = torch.cat(per_image_preserve_mask).to(device).bool()

                        with torch.no_grad():
                            b_se, b_de = net.sam_prompt_encoder(
                                points=(b_coords_cat, b_labels_cat),
                                boxes=None,
                                masks=None,
                                batch_size=batch_size,
                            )
                        (
                            b_low_res_masks,
                            b_iou_preds,
                            b_object_score_logits,
                            b_all_low_res_masks,
                            b_all_quality,
                        ) = _decode_training_candidates(
                            cfgs,
                            net,
                            image_embed=image_embed,
                            sparse_prompt_embeddings=b_se,
                            dense_prompt_embeddings=b_de,
                            cell_nums=b_cell_nums,
                            high_res_feats=high_res_feats,
                            runtime_stats=runtime_stats,
                        )
                        obj_logits = b_object_score_logits.squeeze(-1)
                        pms_object = F.binary_cross_entropy_with_logits(
                            obj_logits,
                            is_pos_cat,
                            reduction="mean",
                        )
                        loss_dict["loss_pms_object"] = (
                            pms_object * pms_coef * criterion.pms_object_weight
                        )

                        if n_pos > 0:
                            pos_idx_mask = is_pos_cat.bool()
                            b_iou_values, _ = torch.max(b_iou_preds, dim=1)
                            b_pred = _refine_sam_masks(cfgs, b_low_res_masks)
                            b_pred_pos = b_pred[pos_idx_mask]
                            b_iou_pos = b_iou_values[pos_idx_mask]
                            preserve_pos_mask = is_preserve_cat[pos_idx_mask]
                            residual_pos_mask = ~preserve_pos_mask
                            b_gt_masks_cat = torch.cat(
                                [bm for bm in b_gt_masks_batch if bm.shape[0] > 0]
                            ).to(device).float()
                            assert b_gt_masks_cat.shape[0] == int(pos_idx_mask.sum().item()), (
                                f"pos GT mask count {b_gt_masks_cat.shape[0]} != "
                                f"pos prompt count {int(pos_idx_mask.sum().item())}"
                            )
                            residual_coef = pms_coef * float(
                                getattr(criterion, "pms_residual_mask_weight", 1.0)
                            )
                            preserve_coef = float(
                                getattr(criterion, "pms_preserve_loss_coef", pms_coef)
                            )
                            if preserve_coef < 0:
                                preserve_coef = pms_coef

                            if candidate_groups is not None:
                                if b_all_low_res_masks is None or b_all_quality is None:
                                    raise RuntimeError(
                                        "C1 PMS prompts require the common four-candidate decoder path"
                                    )
                                b_all_masks = F.interpolate(
                                    b_all_low_res_masks,
                                    size=(cfgs.out_size, cfgs.out_size),
                                    mode="bilinear",
                                    align_corners=False,
                                )[pos_idx_mask]
                                b_all_quality_pos = b_all_quality[pos_idx_mask]
                                candidate_groups["pms_residual"] = {
                                    "candidate_logits": b_all_masks[residual_pos_mask],
                                    "quality_predictions": b_all_quality_pos[residual_pos_mask],
                                    "gt_masks": b_gt_masks_cat[residual_pos_mask],
                                    "alpha": residual_coef,
                                }
                                candidate_groups["pms_preservation"] = {
                                    "candidate_logits": b_all_masks[preserve_pos_mask],
                                    "quality_predictions": b_all_quality_pos[preserve_pos_mask],
                                    "gt_masks": b_gt_masks_cat[preserve_pos_mask],
                                    "alpha": preserve_coef,
                                }

                            if residual_pos_mask.any():
                                res_pred = b_pred_pos[residual_pos_mask]
                                res_gt = b_gt_masks_cat[residual_pos_mask]
                                res_iou = b_iou_pos[residual_pos_mask]
                                loss_dict["loss_pms_focal"] = (
                                    criterion.dice_loss(res_pred.unsqueeze(1), res_gt)
                                    * residual_coef
                                    * criterion.pms_focal_weight
                                )
                                loss_dict["loss_pms_dice"] = (
                                    criterion.focal_loss(res_pred.unsqueeze(1), res_gt.unsqueeze(1))
                                    * residual_coef
                                    * criterion.pms_dice_weight
                                )
                                loss_dict["loss_pms_iou"] = (
                                    criterion.iou_loss(res_pred.unsqueeze(1), res_gt.float(), res_iou)
                                    * residual_coef
                                    * criterion.pms_iou_weight
                                )

                            if preserve_pos_mask.any():
                                pre_pred = b_pred_pos[preserve_pos_mask]
                                pre_gt = b_gt_masks_cat[preserve_pos_mask]
                                pre_iou = b_iou_pos[preserve_pos_mask]
                                loss_dict["loss_pms_preserve_focal"] = (
                                    criterion.dice_loss(pre_pred.unsqueeze(1), pre_gt)
                                    * preserve_coef
                                    * criterion.pms_focal_weight
                                )
                                loss_dict["loss_pms_preserve_dice"] = (
                                    criterion.focal_loss(pre_pred.unsqueeze(1), pre_gt.unsqueeze(1))
                                    * preserve_coef
                                    * criterion.pms_dice_weight
                                )
                                loss_dict["loss_pms_preserve_iou"] = (
                                    criterion.iou_loss(pre_pred.unsqueeze(1), pre_gt.float(), pre_iou)
                                    * preserve_coef
                                    * criterion.pms_iou_weight
                                )

                if candidate_groups is not None:
                    collect_candidate_audit = bool(
                        runtime_stats is not None
                        and runtime_stats.get("collect_candidate_audit", True)
                    )
                    stainpms_loss = sum(loss for loss in loss_dict.values())
                    coverage_loss, quality_loss, group_audit = (
                        aggregate_candidate_prompt_groups(
                            candidate_groups,
                            temperature=float(cfgs.candidate_coverage_tau),
                            collect_audit=collect_candidate_audit,
                        )
                    )
                    weighted_coverage = (
                        coverage_loss * float(cfgs.candidate_coverage_coefficient)
                    )
                    weighted_quality = (
                        quality_loss * float(cfgs.candidate_quality_coefficient)
                    )
                    loss_dict["loss_candidate_coverage"] = weighted_coverage
                    loss_dict["loss_candidate_quality"] = weighted_quality
                    if collect_candidate_audit:
                        _record_candidate_loss_audit(
                            runtime_stats,
                            stainpms_loss=stainpms_loss,
                            coverage_loss=coverage_loss,
                            quality_loss=quality_loss,
                            weighted_coverage=weighted_coverage,
                            weighted_quality=weighted_quality,
                            group_audit=group_audit,
                        )

                losses = sum(loss for loss in loss_dict.values())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])

                if not torch.isfinite(losses):
                    nonfinite_keys = [
                        key for key, value in loss_dict.items() if not torch.isfinite(value).all()
                    ]
                    print(
                        f"[Train] non-finite loss at epoch {epoch} "
                        f"batch_step {data_iter_step} start_idx {start_idx}; "
                        f"components={nonfinite_keys}; skipping"
                    )
                    optimizer.zero_grad()
                    if runtime_stats is not None:
                        runtime_stats["nonfinite_loss_skips"] += 1
                    continue

                loss_dict_reduced = reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                for key, value in loss_dict_reduced.items():
                    log_info[key] = log_info.get(key, 0) + value.item()

                optimizer.zero_grad()
                losses.backward()

                if (
                    str(getattr(cfgs, "warmstart_candidate_arm", "")).lower()
                    in {"legacy", "c0", "c1"}
                    and runtime_stats is not None
                    and runtime_stats.get("capture_gradient_audit", True)
                ):
                    _record_gradient_audit(runtime_stats, point_net, net)

                trainable_params = [
                    p for group in optimizer.param_groups for p in group["params"] if p.requires_grad
                ]
                has_bad_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in trainable_params
                )
                if has_bad_grad:
                    print(f"[Train] non-finite gradients at epoch {epoch}; skipping")
                    optimizer.zero_grad()
                    if runtime_stats is not None:
                        runtime_stats["nonfinite_gradient_skips"] += 1
                    continue

                if cfgs.clip_grad > 0 and trainable_params:
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfgs.clip_grad)
                optimizer.step()
                if runtime_stats is not None:
                    runtime_stats["optimizer_steps"] += 1
                metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

                if (
                    max_optimizer_steps is not None
                    and runtime_stats is not None
                    and runtime_stats["optimizer_steps"] >= int(max_optimizer_steps)
                ):
                    stop_after_step_limit = True
                    break

            pbar.update()
            if stop_after_step_limit:
                break

    denominator = (
        int(runtime_stats["images_seen"])
        if runtime_stats is not None
        else len(train_loader)
    )
    return {key: value / max(1, denominator) for key, value in log_info.items()}


def validation_on_epoch(
    cfgs,
    args,
    val_loader,
    epoch,
    point_net,
    point_encoder,
    net: nn.Module,
    load,
    iou_threshold,
    memory_bank_list,
    device,
):
    point_net.eval()
    net.eval()

    feat_sizes = [(64, 64), (32, 32), (16, 16)]
    score_lists = {
        "pq": [],
        "dq": [],
        "sq": [],
        "aji": [],
        "aji_p": [],
        "dice2": [],
        "dice1": [],
        "_records": [],
    }
    margin = 7

    with tqdm(total=len(val_loader), desc="Validation round", unit="batch", leave=False) as pbar:
        for _, (img_seg, inst_maps, type_maps, gt_points, labels, bi_masks, ori_shape, file_inds, name) in enumerate(val_loader):
            images_seg = img_seg.to(device)
            inst_maps = inst_maps.numpy()

            all_masks = []
            all_boxes = []
            all_scores = []
            all_inds = []
            all_points = []
            all_points_scores = []
            all_points_class = []
            processed_boxes = []
            candidate_records = []
            point_id_map = {}
            next_id = 0
            context_memory_bank_list = []

            crop_boxes = crop_with_overlap(
                images_seg[0],
                cfgs.crop_size,
                cfgs.crop_size,
                cfgs.overlap,
                load,
            ).tolist()

            for crop_box in crop_boxes:
                x1, y1, x2, y2 = crop_box
                img = images_seg[..., y1:y2, x1:x2].to(device)

                with torch.no_grad():
                    pd_points, pd_scores, pd_classes, _, _, _, _ = predict(
                        point_net,
                        img,
                        ori_shape=np.array((y2 - y1, x2 - x1)),
                        filtering=args.test.filtering,
                        nms_thr=args.test.nms_thr,
                    )

                if len(pd_points) == 0:
                    processed_boxes.append(crop_box)
                    continue

                pd_points[:, 0] += x1
                pd_points[:, 1] += y1

                keep_new = np.ones(len(pd_points), dtype=bool)
                for prev_box in processed_boxes:
                    px1, py1, px2, py2 = prev_box
                    keep_new &= ~(
                        (pd_points[:, 0] >= px1 + 1)
                        & (pd_points[:, 0] <= px2 - 1)
                        & (pd_points[:, 1] >= py1 + 1)
                        & (pd_points[:, 1] <= py2 - 1)
                    )
                processed_boxes.append(crop_box)

                pd_points = pd_points[keep_new]
                pd_scores = pd_scores[keep_new]
                pd_classes = pd_classes[keep_new]
                if len(pd_points) == 0:
                    continue

                all_points.append(pd_points)
                all_points_scores.append(pd_scores)
                all_points_class.append(pd_classes)

                current_points = np.vstack(all_points)
                current_scores = np.concatenate(all_points_scores)
                current_classes = np.concatenate(all_points_class)
                current_points, current_scores, current_classes = point_nms(
                    current_points,
                    current_scores,
                    current_classes,
                    args.test.nms_thr,
                )

                current_inds = []
                for point in current_points:
                    point_tuple = tuple(point.tolist())
                    if point_tuple not in point_id_map:
                        point_id_map[point_tuple] = next_id
                        next_id += 1
                    current_inds.append(point_id_map[point_tuple])
                current_inds = torch.tensor(current_inds).long()

                prompt_points = torch.from_numpy(current_points).unsqueeze(1)
                keep = (
                    (prompt_points[..., 0] >= x1)
                    & (prompt_points[..., 0] < x2)
                    & (prompt_points[..., 1] >= y1)
                    & (prompt_points[..., 1] < y2)
                ).squeeze(1)
                if keep.sum() == 0:
                    continue

                sub_prompt_points = (
                    prompt_points[keep] - torch.as_tensor([x1, y1])
                ).to(device).float()
                sub_prompt_labels = torch.ones(
                    sub_prompt_points.size(0),
                    1,
                    dtype=torch.int,
                    device=device,
                )
                keep_np = keep.cpu().numpy()

                with torch.no_grad():
                    pred, values, iou_predictions, vision_feats, image_embed = inference(
                        net,
                        point_encoder,
                        img,
                        memory_bank_list,
                        sub_prompt_points,
                        sub_prompt_labels,
                        feat_sizes,
                        context_memory_bank_list,
                        x1,
                        y1,
                        True,
                        cfgs,
                        device,
                    )

                    if cfgs.tta:
                        pred, values, iou_predictions = _tta_average(
                            net,
                            point_encoder,
                            img,
                            memory_bank_list,
                            sub_prompt_points,
                            sub_prompt_labels,
                            feat_sizes,
                            context_memory_bank_list,
                            x1,
                            y1,
                            cfgs,
                            device,
                            pred,
                            values,
                            iou_predictions,
                        )

                    inst_pred = combine_mask(ori_shape, sub_prompt_points, pred, values)
                    high_res_multimasks = torch.from_numpy(inst_pred.astype(float)).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                    maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                        current_vision_feats=vision_feats,
                        feat_sizes=feat_sizes,
                        pred_masks_high_res=high_res_multimasks,
                        is_mask_from_pts=True,
                    )
                    maskmem_features = maskmem_features.to(device=device, non_blocking=True)
                    maskmem_pos_enc = maskmem_pos_enc[0].to(device=device, non_blocking=True)

                    if cfgs.texture:
                        if len(memory_bank_list) < cfgs.texture_memory_bank_size:
                            for batch_idx in range(maskmem_features.size(0)):
                                memory_bank_list.append(
                                    [
                                        maskmem_features[batch_idx].unsqueeze(0),
                                        maskmem_pos_enc[batch_idx].unsqueeze(0),
                                        iou_predictions,
                                        image_embed[batch_idx].reshape(-1).detach(),
                                    ]
                                )
                        else:
                            for batch_idx in range(maskmem_features.size(0)):
                                bank_flat = [element[0].reshape(-1).to(device) for element in memory_bank_list]
                                bank_flat = torch.stack(bank_flat)
                                bank_norm = F.normalize(bank_flat, p=2, dim=1)
                                current_similarity = torch.mm(bank_norm, bank_norm.t())
                                current_similarity_no_diag = current_similarity.clone()
                                diag_indices = torch.arange(current_similarity_no_diag.size(0))
                                current_similarity_no_diag[diag_indices, diag_indices] = float("-inf")
                                single_key = F.normalize(maskmem_features[batch_idx].reshape(-1), p=2, dim=0).unsqueeze(1)
                                similarity_scores = torch.mm(bank_norm, single_key).squeeze()
                                min_similarity_index = torch.argmin(similarity_scores)
                                max_similarity_index = torch.argmax(current_similarity_no_diag[min_similarity_index])
                                if similarity_scores[min_similarity_index] < current_similarity_no_diag[min_similarity_index][max_similarity_index]:
                                    if iou_predictions > memory_bank_list[max_similarity_index][2] - 0.1:
                                        memory_bank_list.pop(max_similarity_index)
                                        memory_bank_list.append(
                                            [
                                                maskmem_features[batch_idx].unsqueeze(0),
                                                maskmem_pos_enc[batch_idx].unsqueeze(0),
                                                iou_predictions,
                                                image_embed[batch_idx].reshape(-1).detach(),
                                            ]
                                        )

                    masks = mask_process_eval(
                        current_classes[keep_np],
                        current_inds[keep],
                        crop_box,
                        ori_shape,
                        sub_prompt_points,
                        pred,
                        values,
                    )

                    for mask_data in masks:
                        bx1, by1, bx2, by2 = mask_data["bbox"]
                        sx1, sy1, sx2, sy2 = crop_box
                        ori_h, ori_w = _ori_hw(ori_shape)
                        edge_penalized = False
                        if (
                            (bx1 > margin and abs(bx1 - sx1) <= margin)
                            or (abs(bx2 - ori_h) > margin and abs(bx2 - sx2) <= margin)
                            or (by1 > margin and abs(by1 - sy1) <= margin)
                            or (abs(by2 - ori_w) > margin and abs(by2 - sy2) <= margin)
                        ):
                            assembly_score = mask_data["predicted_iou"] * 0.3
                            edge_penalized = True
                        else:
                            assembly_score = mask_data["predicted_iou"]

                        all_scores.append(assembly_score)
                        all_masks.append(mask_data["segmentation"][:ori_h, :ori_w])
                        all_boxes.append(mask_data["bbox"])
                        all_inds.append(mask_data["inds"])
                        candidate_records.append(
                            {
                                "candidate_index": len(candidate_records),
                                "bbox": mask_data["bbox"],
                                "crop_box": [int(v) for v in crop_box],
                                "predicted_iou": float(mask_data["predicted_iou"]),
                                "assembly_score": float(assembly_score),
                                "stability_score": float(mask_data["stability_score"]),
                                "point": mask_data["point"],
                                "categories": mask_data["categories"],
                                "inds": mask_data["inds"],
                                "edge_penalized": bool(edge_penalized),
                            }
                        )

            if getattr(cfgs, "dump_eval_artifacts_dir", ""):
                pred_inst_map, selected_records = _assemble_instance_map(
                    all_boxes,
                    all_scores,
                    all_masks,
                    all_inds,
                    inst_maps[0].shape,
                    iou_threshold,
                    all_records=candidate_records,
                    return_records=True,
                )
            else:
                pred_inst_map = _assemble_instance_map(
                    all_boxes,
                    all_scores,
                    all_masks,
                    all_inds,
                    inst_maps[0].shape,
                    iou_threshold,
                )
                selected_records = []

            dump_dir = getattr(cfgs, "dump_baseline_masks_dir", None)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                image_name = _safe_image_name(name)
                out_path = os.path.join(dump_dir, f"{image_name}.npy")
                new_cov = pred_inst_map.astype(np.int32)
                if bool(getattr(cfgs, "coverage_accumulate", False)) and os.path.exists(out_path):
                    new_cov = _accumulate_coverage(
                        np.load(out_path).astype(np.int32),
                        new_cov,
                    )
                np.save(out_path, new_cov)

            artifact_dir = getattr(cfgs, "dump_eval_artifacts_dir", "") or ""
            if artifact_dir:
                image_name = _safe_image_name(name)
                _dump_eval_artifacts(
                    artifact_dir,
                    image_name,
                    inst_maps[0],
                    pred_inst_map,
                    candidate_records,
                    selected_records,
                )

            image_name = _safe_image_name(name)
            _append_metric_scores(
                inst_maps[0],
                pred_inst_map,
                score_lists,
                evaluator_mode=getattr(cfgs, "evaluator_mode", "legacy_skip"),
                sample_id=image_name,
            )

            if cfgs.vis:
                namecat = "Test_" + "_".join(str(item) for item in name) + "_"
                vis_inst_image(
                    images_seg,
                    torch.from_numpy(pred_inst_map).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device),
                    torch.tensor(inst_maps).unsqueeze(0).to(device),
                    os.path.join(cfgs.path_helper["sample_path"], namecat + "epoch_" + str(epoch) + ".jpg"),
                    reverse=False,
                    points=None,
                )

            pbar.update()

    metrics = _mean_metric_tuple(score_lists)
    metrics_output_dir = str(getattr(cfgs, "metrics_output_dir", "") or "")
    if metrics_output_dir:
        epoch_dir = os.path.join(metrics_output_dir, f"epoch_{int(epoch):04d}")
        eval_manifest_payload = getattr(test_loader.dataset, "manifest", None) or {}
        write_evaluation_outputs(
            score_lists["_records"],
            epoch_dir,
            context={
                "dataset": str(getattr(cfgs, "dataset", "")),
                "epoch": int(epoch),
                "evaluator_mode": str(
                    getattr(cfgs, "evaluator_mode", "legacy_skip")
                ),
                "match_iou": 0.5,
                "aggregation_unit": "complete_reconstructed_image",
                "crop_overlap": int(getattr(cfgs, "overlap", 0)),
                "point_nms_threshold": int(args.test.nms_thr),
                "test_filtering": bool(args.test.filtering),
                "train_manifest": str(getattr(cfgs, "train_manifest", "") or ""),
                "eval_manifest": str(getattr(cfgs, "eval_manifest", "") or ""),
                "eval_manifest_sha256": eval_manifest_payload.get("manifest_sha256"),
                "eval_protocol_id": eval_manifest_payload.get("protocol_id"),
            },
        )
    return metrics


def _tta_average(
    net,
    point_encoder,
    img,
    memory_bank_list,
    sub_prompt_points,
    sub_prompt_labels,
    feat_sizes,
    context_memory_bank_list,
    x1,
    y1,
    cfgs,
    device,
    pred,
    values,
    iou_predictions,
):
    side = cfgs.crop_size - 1

    points_h = sub_prompt_points.clone()
    points_h[..., 0] = side - points_h[..., 0]
    pred_h, values_h, iou_h, _, _ = inference(
        net,
        point_encoder,
        torch.flip(img, dims=[3]),
        memory_bank_list,
        points_h,
        sub_prompt_labels,
        feat_sizes,
        context_memory_bank_list,
        x1,
        y1,
        False,
        cfgs,
        device,
    )
    pred_h = torch.flip(pred_h, dims=[2])

    points_v = sub_prompt_points.clone()
    points_v[..., 1] = side - points_v[..., 1]
    pred_v, values_v, iou_v, _, _ = inference(
        net,
        point_encoder,
        torch.flip(img, dims=[2]),
        memory_bank_list,
        points_v,
        sub_prompt_labels,
        feat_sizes,
        context_memory_bank_list,
        x1,
        y1,
        False,
        cfgs,
        device,
    )
    pred_v = torch.flip(pred_v, dims=[1])

    pred = (pred + pred_h + pred_v) / 3
    values = (values + values_h + values_v) / 3
    iou_predictions = (iou_predictions + iou_h + iou_v) / 3
    return pred, values, iou_predictions


def inference(
    net,
    point_encoder,
    img,
    memory_bank_list,
    sub_prompt_points,
    sub_prompt_labels,
    feat_sizes,
    context_memory_bank_list,
    x1,
    y1,
    flag,
    cfgs,
    device,
):
    feats, _ = point_encoder(img)
    backbone_out, _ = net.forward_image(img, feats)
    _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
    batch_size = vision_feats[-1].size(1)

    memfeatures = vision_feats
    memfeatures_pos = vision_pos_embeds

    if cfgs.context:
        vision_feats, vision_pos_embeds = context_memory_attention(
            context_memory_bank_list,
            vision_feats,
            vision_pos_embeds,
            [x1],
            [y1],
            net,
            feat_sizes,
            cfgs.context_atten_k,
        )

    if cfgs.texture:
        if len(memory_bank_list) == 0:
            zero = torch.zeros(1, batch_size, net.hidden_dim, device=device)
            vision_feats[-1] = vision_feats[-1] + zero
            vision_pos_embeds[-1] = vision_pos_embeds[-1] + zero
        else:
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []
            for element in memory_bank_list:
                to_cat_memory.append(element[0].to(device, non_blocking=True).flatten(2).permute(2, 0, 1))
                to_cat_memory_pos.append(element[1].to(device, non_blocking=True).flatten(2).permute(2, 0, 1))
                to_cat_image_embed.append(element[3].to(device, non_blocking=True))

            memory_stack_ori = torch.stack(to_cat_memory, dim=0)
            memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
            image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

            vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(batch_size, -1, 64, 64)
            vision_feats_temp = vision_feats_temp.reshape(batch_size, -1)
            image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
            vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
            similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
            similarity_scores = F.softmax(similarity_scores, dim=1)
            sampled_indices = torch.topk(similarity_scores, batch_size, dim=1).indices.squeeze(1)

            memory_stack_new = memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
            memory = memory_stack_new.reshape(-1, memory_stack_new.size(2), memory_stack_new.size(3))
            memory_pos_stack_new = memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
            memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_new.size(2), memory_stack_new.size(3))
            vision_feats[-1], vision_pos_embeds[-1] = net.memory_attention(
                state="texture",
                curr=[vision_feats[-1]],
                curr_pos=[vision_pos_embeds[-1]],
                memory=memory,
                memory_pos=memory_pos,
                num_obj_ptr_tokens=0,
            )

    feats = [
        feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
    ][::-1]
    image_embed = feats[-1]
    high_res_feats = feats[:-1]

    if flag and cfgs.context and len(context_memory_bank_list) < cfgs.context_memory_bank_size:
        context_memory_bank_list.append(
            [memfeatures[-1].detach(), memfeatures_pos[-1].detach(), x1, y1]
        )

    se, de = net.sam_prompt_encoder(
        points=(sub_prompt_points, sub_prompt_labels),
        boxes=None,
        masks=None,
        batch_size=batch_size,
    )
    low_res_multimasks, iou_predictions, _, _ = net.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=net.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=se,
        dense_prompt_embeddings=de,
        multimask_output=False,
        repeat_image=False,
        cell_nums=torch.as_tensor([sub_prompt_points.shape[0]], device=device),
        high_res_features=high_res_feats,
    )
    pred = _refine_sam_masks(cfgs, low_res_multimasks)
    values, _ = torch.max(iou_predictions, dim=1)
    mean_iou = values.mean()
    return pred, values, mean_iou, vision_feats, image_embed


def mask_process_eval(
    cell_types,
    sub_inds,
    crop_box,
    ori_shape,
    points,
    pred,
    iou_predictions,
    mask_threshold=0.0,
    stability_score_offset=1.0,
    box_nms_thresh=1.0,
    pred_iou_thresh=0.0,
    stability_score_thresh=0.0,
):
    if pred.shape[0] == 0:
        return []
    orig_h, orig_w = _ori_hw(ori_shape)
    mask_data = MaskData(
        masks=pred,
        iou_preds=iou_predictions,
        points=points,
        categories=cell_types,
        inds=sub_inds,
    )

    if pred_iou_thresh > 0.0:
        keep_mask = mask_data["iou_preds"] > pred_iou_thresh
        mask_data.filter(keep_mask)

    mask_data["stability_score"] = calculate_stability_score(
        mask_data["masks"],
        mask_threshold,
        stability_score_offset,
    )
    if stability_score_thresh > 0.0:
        keep_mask = mask_data["stability_score"] >= stability_score_thresh
        mask_data.filter(keep_mask)

    mask_data["masks"] = mask_data["masks"] > mask_threshold
    mask_data["boxes"] = batched_mask_to_box(mask_data["masks"])
    mask_data["masks"] = uncrop_masks(mask_data["masks"], crop_box, orig_h, orig_w)
    mask_data["rles"] = mask_to_rle_pytorch(mask_data["masks"])
    del mask_data["masks"]

    keep_by_nms = batched_nms(
        mask_data["boxes"].float(),
        mask_data["iou_preds"],
        torch.zeros_like(mask_data["boxes"][:, 0]),
        iou_threshold=box_nms_thresh,
    )
    mask_data.filter(keep_by_nms)

    mask_data["boxes"] = uncrop_boxes_xyxy(mask_data["boxes"], crop_box)
    mask_data["points"] = uncrop_points(mask_data["points"], crop_box)
    mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]

    masks = []
    for idx in range(len(mask_data["segmentations"])):
        masks.append(
            {
                "segmentation": mask_data["segmentations"][idx],
                "bbox": mask_data["boxes"][idx].tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "stability_score": mask_data["stability_score"][idx].item(),
                "point": mask_data["points"][idx].tolist(),
                "categories": mask_data["categories"][idx].tolist(),
                "inds": mask_data["inds"][idx].tolist(),
            }
        )
    return masks


def combine_mask(
    ori_shape,
    points,
    pred,
    iou_predictions,
    mask_threshold=0.0,
    box_nms_thresh=1.0,
):
    if pred.shape[0] == 0:
        return np.zeros(pred.shape[-2:], dtype=float)

    cell_types = np.ones(points.shape[0], dtype=np.int64)
    sub_inds = torch.arange(points.shape[0], dtype=torch.int64, device=points.device)
    mask_data = MaskData(
        masks=pred,
        iou_preds=iou_predictions,
        points=points,
        categories=cell_types,
        inds=sub_inds,
    )
    mask_data["masks"] = mask_data["masks"] > mask_threshold
    mask_data["boxes"] = batched_mask_to_box(mask_data["masks"])
    mask_data["rles"] = mask_to_rle_pytorch(mask_data["masks"])
    del mask_data["masks"]

    keep_by_nms = batched_nms(
        mask_data["boxes"].float(),
        mask_data["iou_preds"],
        torch.zeros_like(mask_data["boxes"][:, 0]),
        iou_threshold=box_nms_thresh,
    )
    mask_data.filter(keep_by_nms)
    mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]

    masks = []
    for idx in range(len(mask_data["segmentations"])):
        masks.append(
            {
                "segmentation": mask_data["segmentations"][idx],
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "inds": mask_data["inds"][idx].tolist(),
            }
        )

    all_masks = []
    all_scores = []
    all_inds = []
    crop_h, crop_w = _ori_hw(ori_shape)
    for mask_item in masks:
        all_scores.append(mask_item["predicted_iou"])
        all_masks.append(mask_item["segmentation"][:crop_h, :crop_w])
        all_inds.append(mask_item["inds"])

    all_scores = torch.as_tensor(all_scores)
    all_inds = np.asarray(all_inds)
    unique_inds, counts = np.unique(all_inds, return_counts=True)
    keep_prior = np.ones(len(all_inds), dtype=bool)
    for i in np.where(counts > 1)[0]:
        inds = np.where(all_inds == unique_inds[i])[0]
        inds = np.delete(inds, np.argmax(all_scores[inds]))
        keep_prior[inds] = False

    pred_map = np.zeros((pred.shape[1], pred.shape[2]), dtype=float)
    for ind in np.where(keep_prior)[0]:
        if pred_map[all_masks[ind]].all() == 0:
            pred_map[all_masks[ind]] = ind + 1
    return pred_map


def crop_with_overlap(img, split_width, split_height, overlap, load):
    def start_points(size, split_size, overlap_pixels):
        points = [0]
        counter = 1
        stride = 256 - overlap_pixels
        while True:
            point = stride * counter
            if point + split_size >= size:
                if split_size != size:
                    points.append(size - split_size)
                break
            points.append(point)
            counter += 1
        return points

    _, img_h, img_w = img.shape
    x_points = start_points(img_w, split_width, overlap)
    y_points = start_points(img_h, split_height, overlap)

    crop_boxes = []
    if load == "sequence":
        for x in x_points:
            for y in y_points:
                crop_boxes.append([x, y, min(x + split_width, img_w), min(y + split_height, img_h)])
    elif load == "unsequence":
        flag = True
        for x in x_points:
            y_iter = y_points if flag else np.flip(y_points)
            for y in y_iter:
                crop_boxes.append([x, y, min(x + split_width, img_w), min(y + split_height, img_h)])
            flag = not flag
    elif load in ("clockwise", "unclockwise"):
        top = 0
        bottom = len(y_points) - 1
        left = 0
        right = len(x_points) - 1
        while top <= bottom or left <= right:
            if top <= bottom:
                for y in range(left, right + 1):
                    crop_boxes.append([x_points[top], y_points[y], min(x_points[top] + split_width, img_w), min(y_points[y] + split_height, img_h)])
                top += 1
            if left <= right:
                for x in range(top, bottom + 1):
                    crop_boxes.append([x_points[x], y_points[right], min(x_points[x] + split_width, img_w), min(y_points[right] + split_height, img_h)])
                right -= 1
            if top <= bottom:
                for y in np.flip(range(left, right + 1)):
                    crop_boxes.append([x_points[bottom], y_points[y], min(x_points[bottom] + split_width, img_w), min(y_points[y] + split_height, img_h)])
                bottom -= 1
            if left <= right:
                for x in np.flip(range(top, bottom + 1)):
                    crop_boxes.append([x_points[x], y_points[left], min(x_points[x] + split_width, img_w), min(y_points[left] + split_height, img_h)])
                left += 1
        if load == "unclockwise":
            crop_boxes = crop_boxes[::-1]
    else:
        raise ValueError(f"Unsupported crop load order: {load}")
    return np.asarray(crop_boxes)


def context_memory_attention(context_memory_bank_list, feats, feats_pos, xs, ys, net, feat_sizes, k):
    batch_size = feats[-1].size(1)
    device = feats[-1].device
    if len(context_memory_bank_list) == 0:
        zero = torch.zeros(1, batch_size, net.hidden_dim, device=device)
        feats[-1] = feats[-1] + zero
        feats_pos[-1] = feats_pos[-1] + zero
        return feats, feats_pos

    memory_list = [[] for _ in range(batch_size)]
    for element in context_memory_bank_list:
        maskmem_features = element[0].to(device, non_blocking=True)
        maskmem_pos_enc = element[1].to(device, non_blocking=True)
        x, y = element[2], element[3]
        for i in range(batch_size):
            distance = math.sqrt((x - xs[i]) ** 2 + (y - ys[i]) ** 2)
            memory_list[i].append([maskmem_features, maskmem_pos_enc, distance])

    for sub_memory_list in memory_list:
        sub_memory_list.sort(key=lambda item: item[2])

    for i in range(min(k, len(memory_list[0]))):
        memory = torch.stack([sublist[i][0] for sublist in memory_list]).transpose(0, 1).squeeze(2)
        memory_pos = torch.stack([sublist[i][1] for sublist in memory_list]).transpose(0, 1).squeeze(2)
        feats[-1], feats_pos[-1] = net.memory_attention(
            state="context",
            curr=feats[-1],
            curr_pos=feats_pos[-1],
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=0,
        )
    return feats, feats_pos

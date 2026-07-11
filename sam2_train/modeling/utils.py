import random
import torch.distributed as dist

import numpy as np
import scipy.spatial as S
import torchvision.transforms as T

from datetime import datetime, timedelta
import errno
import os
import time
from collections import defaultdict, deque
from torchvision.transforms.functional import hflip, vflip

import torch
import torch.distributed as dist


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                #eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                eta_string = str(timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        #total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        total_time_str = str(timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


def get_tp(
        pred_points,
        pred_scores,
        gd_points,
        thr=12,
        return_index=False
):
    sorted_pred_indices = np.argsort(-pred_scores)
    sorted_pred_points = pred_points[sorted_pred_indices]

    unmatched = np.ones(len(gd_points), dtype=bool)
    dis = S.distance_matrix(sorted_pred_points, gd_points)

    for i in range(len(pred_points)):
        min_index = dis[i, unmatched].argmin()
        if dis[i, unmatched][min_index] <= thr:
            unmatched[np.where(unmatched)[0][min_index]] = False

        if not np.any(unmatched):
            break
    
    #这里看一下unmatch的点吧
    if return_index:
        return sum(~unmatched), np.where(unmatched)[0],unmatched
    else:
        return sum(~unmatched),unmatched


def point_nms(points, scores, classes, nms_thr=-1):
    _reserved = np.ones(len(points), dtype=bool)
    dis_matrix = S.distance_matrix(points, points)
    np.fill_diagonal(dis_matrix, np.inf)

    for idx in np.argsort(-scores):
        if _reserved[idx]:
            _reserved[dis_matrix[idx] <= nms_thr] = False

    points = points[_reserved]
    scores = scores[_reserved]
    classes = classes[_reserved]

    return points, scores, classes


def set_seed(args):
    seed = args.seed
    # seed = args.seed + get_rank()

    # Set random seed for PyTorch
    torch.manual_seed(seed)

    # Set random seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set random seed for NumPy
    np.random.seed(seed)

    # Set random seed for random module
    random.seed(seed)

    # Set random seed for CuDNN if available
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pre_processing(img):
    trans = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return trans(img).unsqueeze(0)

from run.utils import *
@torch.no_grad()
def predict(
        model,
        image,
        nms_thr=-1,
        ori_shape=None,
        filtering=False,
        prompt_score_mode="objectness",
        return_candidate_trace=False,
):
    ori_h, ori_w = ori_shape
    outputs,_,_,_ = model(image)

    points = outputs['pred_coords'][0].cpu().numpy()
    scores = outputs['pred_logits'][0].softmax(-1).cpu().numpy()
    source_indices = np.arange(len(points), dtype=np.int64)

    ori_points = points.copy()
    ori_scores = scores.copy()

    classes = np.argmax(scores, axis=-1)

    np.clip(points[:, 0], a_min=0, a_max=ori_w - 1, out=points[:, 0])
    np.clip(points[:, 1], a_min=0, a_max=ori_h - 1, out=points[:, 1])
    valid_flag = classes < (scores.shape[-1] - 1)

    points = points[valid_flag]
    source_indices = source_indices[valid_flag]
    foreground_scores = scores[:, 0]
    if prompt_score_mode == "objectness":
        ranking_scores = foreground_scores
        quality_scores = np.full_like(foreground_scores, np.nan, dtype=np.float32)
    else:
        if "pred_quality_logits" not in outputs:
            raise ValueError(f"prompt_score_mode={prompt_score_mode} requires a PromptCredit quality head")
        quality_scores = torch.sigmoid(outputs["pred_quality_logits"][0]).cpu().numpy()
        if prompt_score_mode == "objectness_x_quality":
            ranking_scores = foreground_scores * quality_scores
        elif prompt_score_mode == "quality":
            ranking_scores = quality_scores
        else:
            raise ValueError(f"Unknown prompt_score_mode: {prompt_score_mode}")
    scores = ranking_scores[valid_flag]
    classes = classes[valid_flag]
    foreground_scores = foreground_scores[valid_flag]
    quality_scores = quality_scores[valid_flag]

    mask = outputs['pred_masks'][0, 0].to(torch.float32).cpu().numpy() > 0

    if filtering:
        valid_flag = mask[points.astype(int)[:, 1], points.astype(int)[:, 0]]
        points = points[valid_flag]
        scores = scores[valid_flag]
        classes = classes[valid_flag]
        source_indices = source_indices[valid_flag]
        foreground_scores = foreground_scores[valid_flag]
        quality_scores = quality_scores[valid_flag]

    # if len(points) and nms_thr > 0:
    #     points, scores, classes = point_nms(points, scores, classes, nms_thr)

    if return_candidate_trace:
        trace = {
            "proposal_indices": source_indices,
            "objectness_scores": foreground_scores,
            "quality_scores": quality_scores,
            "ranking_scores": scores,
        }
        return points, scores, classes, mask, outputs['pred_masks'],ori_points,ori_scores,trace
    return points, scores, classes, mask, outputs['pred_masks'],ori_points,ori_scores


def collate_fn(batch):
    # PMS additions:
    #   x[11] = b_coords (positive):  list of (M_p, 2) per crop
    #   x[12] = b_weights:            list of (M_p,) per crop
    #   x[13] = b_gt_masks:           list of (M_p, H, W) per crop
    #   x[14] = b_neg_coords:         list of (M_n, 2) per crop (Phase 10)
    #   x[15] = b_preserve_counts:    list of preserve-positive counts per crop
    # Backwards-compatible: legacy len(x) falls back to empty tensors.
    images,inst_masks,points_choose,labels_choose,cell_nums, points_all, labels_all, bi_masks,ori_shape,xs,ys = [[] for _ in range(11)]
    b_coords_all = []
    b_weights_all = []
    b_gt_masks_all = []
    b_neg_coords_all = []
    b_preserve_counts_all = []
    for x in batch:
        images.extend(x[0])
        inst_masks.extend(x[1])
        points_choose.extend(x[2])
        labels_choose.extend(x[3])

        points_all.extend(x[4])
        labels_all.extend(x[5])
        cell_nums.extend(x[6])
        bi_masks.extend(x[7])
        ori_shape.extend(x[8])
        xs.extend(x[9])
        ys.extend(x[10])
        n_crops = len(x[0])
        if len(x) >= 13:
            b_coords_all.extend(x[11])
            b_weights_all.extend(x[12])
        else:
            b_coords_all.extend([torch.empty(0, 2, dtype=torch.float32)] * n_crops)
            b_weights_all.extend([torch.empty(0, dtype=torch.float32)] * n_crops)
        if len(x) >= 14:
            b_gt_masks_all.extend(x[13])
        else:
            H_p, W_p = x[0][0].shape[-2:] if n_crops > 0 else (256, 256)
            b_gt_masks_all.extend([torch.empty(0, H_p, W_p, dtype=torch.uint8)] * n_crops)
        if len(x) >= 15:
            b_neg_coords_all.extend(x[14])
        else:
            b_neg_coords_all.extend([torch.empty(0, 2, dtype=torch.float32)] * n_crops)
        if len(x) >= 16:
            b_preserve_counts_all.extend([int(v) for v in x[15]])
        else:
            b_preserve_counts_all.extend([0] * n_crops)
    return (
        torch.stack(images), torch.cat(inst_masks), points_choose, labels_choose,
        points_all, labels_all, torch.as_tensor(cell_nums), torch.stack(bi_masks),
        ori_shape, xs, ys, b_coords_all, b_weights_all, b_gt_masks_all,
        b_neg_coords_all, torch.as_tensor(b_preserve_counts_all, dtype=torch.long),
    )


from sam2_train.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    calculate_stability_score,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
from torchvision.ops.boxes import batched_nms

import torch.nn as nn
@torch.inference_mode()
def inference(
        model: nn.Module,
        image: torch.Tensor,
        crop_box: np.ndarray,
        ori_size: tuple,
        prompt_points: torch.Tensor,
        prompt_labels: torch.Tensor,
        prompt_cell_types: torch.Tensor,
        points_per_batch: int = 256,
        mask_threshold: float = .0,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 1.0,
        min_mask_region_area: int = 0,
        inds=None,
        tta=False,
        bi_masks=None,
):
    orig_h, orig_w = ori_size

    # Generate masks for this crop in batches
    mask_data = MaskData()
    for (points, labels, cell_types, sub_inds) in batch_iterator(points_per_batch, prompt_points, prompt_labels,
                                                              prompt_cell_types, inds):
        
        outputs ,semantic_embeddings= model(
            image,
            points,
            labels,
            torch.as_tensor([len(points)]).to(points.device),
            bi_masks
        )

        masks = outputs["pred_masks"]
        iou_preds = outputs["pred_ious"]

        # Serialize predictions and store in MaskData
        batch_data = MaskData(
            masks=masks,
            iou_preds=iou_preds,
            points=points,
            categories=cell_types,
            inds=sub_inds
        )
        del masks

        # Filter by predicted IoU
        if pred_iou_thresh > 0.0:
            keep_mask = batch_data["iou_preds"] > pred_iou_thresh
            batch_data.filter(keep_mask)

        # Calculate stability score
        batch_data["stability_score"] = calculate_stability_score(
            batch_data["masks"], mask_threshold, stability_score_offset
        )
        if stability_score_thresh > 0.0:
            keep_mask = batch_data["stability_score"] >= stability_score_thresh
            batch_data.filter(keep_mask)

        # Threshold masks and calculate boxes
        batch_data["masks"] = batch_data["masks"] > mask_threshold
        batch_data["boxes"] = batched_mask_to_box(batch_data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(batch_data["boxes"], crop_box, [0, 0, orig_w, orig_h], atol=7)

        if bi_masks!=None and (not torch.all(keep_mask)) :
            batch_data.filter(keep_mask)

        # Compress to RLE
        batch_data["masks"] = uncrop_masks(batch_data["masks"], crop_box, orig_h, orig_w)
        batch_data["rles"] = mask_to_rle_pytorch(batch_data["masks"])
        del batch_data["masks"]

        mask_data.cat(batch_data)
        del batch_data

    # Remove duplicates within this crop.
    keep_by_nms = batched_nms(
        mask_data["boxes"].float(),
        mask_data["iou_preds"],
        torch.zeros_like(mask_data["boxes"][:, 0]),  # apply cross categories
        iou_threshold=box_nms_thresh
    )
    mask_data.filter(keep_by_nms)

    # Return to the original image frame
    mask_data["boxes"] = uncrop_boxes_xyxy(mask_data["boxes"], crop_box)
    mask_data["points"] = uncrop_points(mask_data["points"], crop_box)
    mask_data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(mask_data["rles"]))])
    mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]

    # Write mask records
    curr_anns = []
    for idx in range(len(mask_data["segmentations"])):
        ann = {
            "segmentation": mask_data["segmentations"][idx],
            "area": area_from_rle(mask_data["rles"][idx]),
            "bbox": mask_data["boxes"][idx].tolist(),
            "predicted_iou": mask_data["iou_preds"][idx].item(),
            "point_coords": [mask_data["points"][idx].tolist()],
            "stability_score": mask_data["stability_score"][idx].item(),
            "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            'categories': mask_data['categories'][idx].tolist(),
            'inds': mask_data['inds'][idx].tolist()
        }
        curr_anns.append(ann)

    return curr_anns,outputs["pred_masks"],semantic_embeddings

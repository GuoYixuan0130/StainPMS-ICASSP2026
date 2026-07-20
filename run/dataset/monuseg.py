import os

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io
import scipy.io as sio

from run.dataset.manifest import load_dataset_manifest
from stainpms.candidate import compute_b_candidates_oncrop, compute_baseline_center_candidates


class MONUSEG(Dataset):
    def __init__(
        self,
        args,
        cfgs,
        data_path,
        load,
        mode='train',
        manifest_path=None,
        data_split=None,
        verify_manifest_hashes=False,
    ):
        self.data_path = data_path
        self.mode = mode
        self.data_split = data_split or mode
        if self.data_split == 'train':
            self.image_root = data_path + '/train_12/images'
            self.label_root = data_path + '/train_12/labels'
        elif self.data_split == 'test':
            self.image_root = data_path + '/test/images'
            self.label_root = data_path + '/test/labels'
        else:
            raise ValueError(f"Unsupported MoNuSeg data split: {self.data_split}")
        self.manifest = None
        self.records = None
        if manifest_path:
            self.manifest, self.records = load_dataset_manifest(
                manifest_path,
                expected_dataset="monuseg",
                require_labels=True,
                verify_hashes=bool(verify_manifest_hashes),
            )
            self.paths = [os.path.basename(record["image_path"]) for record in self.records]
            self.sample_names = [record["sample_id"] for record in self.records]
            print(
                f"[MONUSEG manifest] mode={mode} split={self.data_split} "
                f"protocol={self.manifest.get('protocol_id')} n={len(self.records)} "
                f"sha256={self.manifest.get('manifest_sha256')}"
            )
        else:
            self.paths = sorted(os.listdir(self.image_root))
            self.sample_names = [os.path.splitext(path)[0] for path in self.paths]
        self.crop_size = args.crop_size
        self.overlap = args.overlap
        self.load = load

        self.num_mask_per_img = 150
        self.num_classes = 1

        self.transform = A.Compose(
          [getattr(A, tf_dict.pop('type'))(**tf_dict) for tf_dict in cfgs.data.get(mode).transform]
          + [ToTensorV2()], p=1)

        # PMS fine-tune support.
        # NOTE the parameter naming inversion: in this class `args` = the
        # argparse Namespace and `cfgs` = the mmengine Config.
        stain_cfg = getattr(cfgs, "criterion", {})
        self.stain_top_k = int(getattr(stain_cfg, "stain_top_k", 20))
        self.stain_min_distance = int(getattr(stain_cfg, "stain_min_distance", 12))
        self.stain_open_disk = int(getattr(stain_cfg, "stain_open_disk", 2))
        self.stain_sigma = float(getattr(stain_cfg, "stain_sigma", 1.0))
        self.stain_baseline_dilate_radius = int(getattr(stain_cfg, "stain_baseline_dilate_radius", 5))
        self.stain_merge_aware = bool(getattr(stain_cfg, "stain_merge_aware", False))
        self.stain_merge_min_distance = int(getattr(stain_cfg, "stain_merge_min_distance", 6))
        self.stain_merge_num_peaks = int(getattr(stain_cfg, "stain_merge_num_peaks", 3))
        self.hed_alpha = float(getattr(stain_cfg, "hed_alpha", 1.0))
        self.hed_beta = float(getattr(stain_cfg, "hed_beta", 0.0))
        self.hed_gamma = float(getattr(stain_cfg, "hed_gamma", 0.0))
        self.pms_enabled = (
            mode == "train"
            and getattr(args, "use_pms", False)
            and float(getattr(stain_cfg, "pms_loss_coef", 0.0)) > 0.0
        )
        self.pms_self_bootstrap = bool(getattr(args, "pms_self_bootstrap", False))
        self.pms_gt_match_radius = int(getattr(stain_cfg, "pms_gt_match_radius", 8))
        self.pms_baseline_prompts = bool(getattr(stain_cfg, "pms_baseline_prompts", False))
        self.pms_preserve_max_prompts = int(getattr(stain_cfg, "pms_preserve_max_prompts", 0))
        self._need_b = self.pms_enabled
        self.baseline_masks_dir = getattr(args, "baseline_masks_dir", "") or ""
        self._baseline_cache = {}
        if self._need_b and self.baseline_masks_dir:
            for name in self.sample_names:
                npy_path = os.path.join(self.baseline_masks_dir, name + ".npy")
                if os.path.exists(npy_path):
                    self._baseline_cache[name] = np.load(npy_path).astype(np.int32)
            if self.pms_enabled and self.pms_self_bootstrap:
                print("[MONUSEG PMS] self-bootstrap coverage cache active; "
                      "the initial online cache will be populated before epoch 0.")
            else:
                print(f"[MONUSEG PMS] loaded {len(self._baseline_cache)}/{len(self.paths)} "
                      f"precomputed baseline masks from {self.baseline_masks_dir}")
            if self.pms_enabled and self.pms_baseline_prompts:
                max_msg = "all" if self.pms_preserve_max_prompts <= 0 else str(self.pms_preserve_max_prompts)
                print(f"[MONUSEG PMS] coverage-preservation prompts enabled; max_per_crop={max_msg}")

    def reload_baseline_masks(self):
        """Re-read baseline_masks_dir/*.npy into _baseline_cache in place.

        Used by main.py's iterative refresh path after the current model has
        overwritten the .npy files. No-op when PMS is not enabled.
        """
        if not (self._need_b and self.baseline_masks_dir):
            return 0
        n_old = len(self._baseline_cache)
        self._baseline_cache.clear()
        for name in self.sample_names:
            npy_path = os.path.join(self.baseline_masks_dir, name + ".npy")
            if os.path.exists(npy_path):
                self._baseline_cache[name] = np.load(npy_path).astype(np.int32)
        n_new = len(self._baseline_cache)
        print(f"[MONUSEG PMS] reloaded {n_new}/{len(self.paths)} baseline masks "
              f"from {self.baseline_masks_dir} (was {n_old})")
        return n_new

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        """Get the images"""
        path = self.paths[index]
        if self.records is not None:
            record = self.records[index]
            image_path = record["image_path"]
            mask_path = record["label_path"]
            name_no_ext = record["sample_id"]
        else:
            image_path = os.path.join(self.image_root, path)
            name_no_ext = os.path.splitext(path)[0]
            mask_path = os.path.join(self.label_root, name_no_ext + '.mat')

        img = io.imread(image_path)[..., :3]
        mask = load_maskfile(mask_path)

        # PMS: stack precomputed baseline mask as 3rd mask channel so
        # albumentations augments it together with image + GT (keeps alignment).
        baseline_attached = False
        if self.mode == 'train' and self._need_b and self._baseline_cache:
            baseline = self._baseline_cache.get(name_no_ext)
            if baseline is not None:
                if baseline.shape != mask.shape[:2]:
                    raise ValueError(
                        f"baseline mask shape {baseline.shape} != GT shape {mask.shape[:2]} for {name_no_ext}"
                    )
                mask = np.concatenate([mask, baseline.astype(mask.dtype)[..., None]], axis=-1)
                baseline_attached = True

        if self.mode == 'train':
            imgs = []
            inst_map_alls = []
            prompt_points_alls = []
            segment_labels_alls = []
            prompt_points_lists = []
            prompt_labels_alls = []
            cell_nums = []
            binary_masks = []
            ori_shapes = []
            xs = []
            ys = []
            b_coords_list = []
            b_weights_list = []
            b_gt_masks_list = []
            b_neg_coords_list = []
            b_preserve_counts_list = []

            img = img.transpose(2, 0, 1)
            mask = mask.transpose(2, 0, 1)

            crop_boxes = crop_with_overlap(
                img,
                self.crop_size,
                self.crop_size,
                self.overlap,
                self.load,
            ).tolist()
            for idx, crop_box in enumerate(crop_boxes):
                x1, y1, x2, y2 = crop_box
                img_c = img[..., y1:y2, x1:x2].transpose(1, 2, 0)
                mask_c = mask[..., y1:y2, x1:x2].transpose(1, 2, 0)

                res = self.transform(image=img_c, mask=mask_c)
                img_c, mask_c = list(res.values())

                ori_shape = mask_c.shape[:2]
                inst_map, type_map = mask_c[..., 0], mask_c[..., 1]
                unique_pids = np.unique(inst_map)[1:]  # remove zero

                cell_num = len(unique_pids)

                if cell_num:
                    
                    chosen_pids = unique_pids
                    inst_maps_all = []

                    prompt_points_all = []
                    prompt_labels_all = []
                    segment_labels_all = []
                    for pid in chosen_pids:
                        mask_single_cell = torch.eq(inst_map, pid)

                        inst_maps_all.append(mask_single_cell)
                        coords = torch.argwhere(mask_single_cell)
                        rand_idx = torch.randint(0, coords.shape[0], (1,))
                        center = coords[rand_idx.item()]
                        pt = center[None, [1, 0]]  # Adjust order to [y, x]
                        prompt_points_all.append(pt)
                        prompt_labels_all.append(type_map[pt[0, 1], pt[0, 0]] - 1)
                        segment_labels_all.append(1)

                    prompt_points_all = torch.stack(prompt_points_all, dim=0)
                    prompt_labels_all = torch.as_tensor(prompt_labels_all)
                    segment_labels_all = torch.as_tensor(segment_labels_all)
                    inst_map_all = torch.stack(inst_maps_all, dim=0)

                else:
                    prompt_points_all = torch.empty(0, 1, 2)
                    prompt_labels_all = torch.zeros(prompt_points_all.squeeze(1).shape[:1])
                    segment_labels_all = torch.zeros(0)
                    inst_map_all = torch.empty(0, 256, 256)

                binary_tensor = (inst_map_all).to(torch.uint8)
                binary_mask = torch.any(binary_tensor, dim=0).to(torch.uint8)

                imgs.append(img_c.to(torch.float32))
                inst_map_alls.append(inst_map_all.long())
                prompt_points_alls.append(prompt_points_all)
                segment_labels_alls.append(segment_labels_all.unsqueeze(1))
                prompt_points_lists.append(prompt_points_all.squeeze(1))
                prompt_labels_alls.append(prompt_labels_all)
                cell_nums.append(cell_num)
                binary_masks.append(binary_mask)
                ori_shapes.append(torch.as_tensor(ori_shape))
                xs.append(x1)
                ys.append(y1)

                # PMS: compute B candidates on the augmented crop. In
                # self-bootstrap PMS, never fall back to raw H peaks before a
                # previous-epoch coverage map has been generated and attached.
                requires_bootstrap_map = self.pms_enabled and self.pms_self_bootstrap
                can_compute_b = (
                    self._need_b
                    and cell_num > 0
                    and (not requires_bootstrap_map or baseline_attached)
                )
                if can_compute_b:
                    crop_baseline = mask_c[..., 2] if baseline_attached else None
                    gt_r = self.pms_gt_match_radius
                    coords_np, weights_np, inst_ids_np = compute_b_candidates_oncrop(
                        img_c, inst_map,
                        baseline_inst_map=crop_baseline,
                        baseline_dilate_radius=self.stain_baseline_dilate_radius,
                        top_k=self.stain_top_k,
                        min_distance=self.stain_min_distance,
                        open_disk=self.stain_open_disk,
                        sigma=self.stain_sigma,
                        gt_match_radius=gt_r,
                        return_gt_inst_ids=True,
                        keep_negative=True,
                        merge_aware=self.stain_merge_aware,
                        merge_min_distance=self.stain_merge_min_distance,
                        merge_num_peaks=self.stain_merge_num_peaks,
                        hed_alpha=self.hed_alpha,
                        hed_beta=self.hed_beta,
                        hed_gamma=self.hed_gamma,
                    )
                    pos_mask_np = inst_ids_np > 0
                    pos_coords = coords_np[pos_mask_np]
                    pos_weights = weights_np[pos_mask_np]
                    pos_inst_ids = inst_ids_np[pos_mask_np]
                    neg_coords = coords_np[~pos_mask_np]
                    preserve_count = 0

                    if (self.pms_baseline_prompts
                            and self.pms_enabled
                            and crop_baseline is not None):
                        bl_coords, bl_weights, bl_inst_ids = compute_baseline_center_candidates(
                            crop_baseline, inst_map,
                            gt_match_radius=self.pms_gt_match_radius,
                        )
                        max_keep = int(self.pms_preserve_max_prompts)
                        if max_keep > 0 and len(bl_coords) > max_keep:
                            bl_coords = bl_coords[:max_keep]
                            bl_inst_ids = bl_inst_ids[:max_keep]
                            bl_weights = bl_weights[:max_keep]
                        if len(bl_coords) > 0:
                            pos_coords = np.concatenate([pos_coords, bl_coords], axis=0)
                            pos_inst_ids = np.concatenate([pos_inst_ids, bl_inst_ids], axis=0)
                            preserve_count = len(bl_coords)
                            n_total = len(pos_coords)
                            pos_weights = np.full(n_total, 1.0 / n_total, dtype=np.float32)

                    b_coords_list.append(torch.from_numpy(pos_coords))
                    b_weights_list.append(torch.from_numpy(pos_weights))
                    b_neg_coords_list.append(torch.from_numpy(neg_coords))
                    b_preserve_counts_list.append(int(preserve_count))

                    inst_map_np = inst_map if isinstance(inst_map, np.ndarray) else inst_map.numpy()
                    H_c, W_c = inst_map_np.shape
                    if len(pos_inst_ids) > 0:
                        masks_np = np.stack(
                            [(inst_map_np == int(iid)).astype(np.uint8)
                             for iid in pos_inst_ids], axis=0
                        )
                    else:
                        masks_np = np.empty((0, H_c, W_c), dtype=np.uint8)
                    b_gt_masks_list.append(torch.from_numpy(masks_np))
                else:
                    b_coords_list.append(torch.empty(0, 2, dtype=torch.float32))
                    b_weights_list.append(torch.empty(0, dtype=torch.float32))
                    b_neg_coords_list.append(torch.empty(0, 2, dtype=torch.float32))
                    b_preserve_counts_list.append(0)
                    H_c = inst_map.shape[-2] if hasattr(inst_map, 'shape') else 256
                    W_c = inst_map.shape[-1] if hasattr(inst_map, 'shape') else 256
                    b_gt_masks_list.append(torch.empty(0, H_c, W_c, dtype=torch.uint8))

            return (
                imgs, inst_map_alls, prompt_points_alls, segment_labels_alls,
                prompt_points_lists, prompt_labels_alls, cell_nums, binary_masks,
                ori_shapes, xs, ys, b_coords_list, b_weights_list,
                b_gt_masks_list, b_neg_coords_list, b_preserve_counts_list,
            )
            
        else:
            res = self.transform(image=img, mask=mask)
            img, mask = list(res.values())

            ori_shape = mask.shape[:2]
            inst_map, type_map = mask[..., 0], mask[..., 1]
            unique_pids = np.unique(inst_map)[1:]  # remove zero

            cell_num = len(unique_pids)

            if cell_num:
                
                chosen_pids = unique_pids
                inst_maps_all = []

                prompt_points_all = []
                prompt_labels_all = []
                segment_labels_all = []
                for pid in chosen_pids:
                    mask_single_cell = torch.eq(inst_map, pid)

                    inst_maps_all.append(mask_single_cell)
                    coords = torch.argwhere(mask_single_cell)
                    center = coords.float().mean(dim=0)
                    center = center.round().long()
                    if mask_single_cell[center[0], center[1]] == 0:
                        # If the center point is on the background, find the nearest foreground point
                        dists = torch.sqrt(((coords - center) ** 2).sum(dim=1).float())
                        closest_idx = dists.argmin()
                        center = coords[closest_idx]
                    pt = center[None, [1, 0]]  # Adjust order to [y, x]
                    prompt_points_all.append(pt)
                    prompt_labels_all.append(type_map[pt[0, 1], pt[0, 0]] - 1)
                    segment_labels_all.append(1)

                prompt_points_all = torch.stack(prompt_points_all, dim=0)
                prompt_labels_all = torch.as_tensor(prompt_labels_all)
                segment_labels_all = torch.as_tensor(segment_labels_all)
                inst_map_all = torch.stack(inst_maps_all, dim=0)
            else:
                prompt_points_all = torch.empty(0, 1, 2)
                prompt_labels_all = torch.zeros(prompt_points_all.squeeze(1).shape[:1])
                inst_map_all = torch.empty(0, 256, 256)
            
            binary_tensor = (inst_map_all).to(torch.uint8)
            binary_mask = torch.any(binary_tensor, dim=0).to(torch.uint8)
            
            return img.to(torch.float32),inst_map, type_map.squeeze(0),prompt_points_all.squeeze(1), prompt_labels_all, binary_mask ,torch.as_tensor(ori_shape),index,name_no_ext

def load_maskfile(mask_path: str):
    inst_map = sio.loadmat(mask_path)['inst_map']
    type_map = (inst_map.copy() > 0).astype(float)

    mask = np.stack([inst_map, type_map], axis=-1)
    return mask

def crop_with_overlap(
        img,
        split_width,
        split_height,
        overlap,
        load
):
    def start_points(
            size,
            split_size,
            overlap
    ):
        points = [0]
        counter = 1
        stride = 256 - overlap
        while True:
            pt = stride * counter
            if pt + split_size >= size:
                if split_size == size:
                    break
                points.append(size - split_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points

    _, img_h, img_w = img.shape

    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)

    crop_boxes = []
    if load == 'sequence':
        for x in X_points:
            for y in Y_points:
                crop_boxes.append([x, y, min(x + split_width, img_w), min(y + split_height, img_h)])
    elif load == 'unsequence':
        flag = True
        for x in X_points:
            if flag:
                for y in Y_points:
                    crop_boxes.append([x, y, min(x + split_width, img_w), min(y + split_height, img_h)])
            else:   
                for y in np.flip(Y_points):
                    crop_boxes.append([x, y, min(x + split_width, img_w), min(y + split_height, img_h)])
            flag = not flag
    elif load == 'clockwise':
        top = 0
        down = len(Y_points)-1
        left = 0
        right = len(X_points)-1
        while top <= down or left <= right:
            if top <= down:
                for y in range(left, right+1):
                    crop_boxes.append([X_points[top], Y_points[y], min(X_points[top] + split_width, img_w), min(Y_points[y] + split_height, img_h)])
                top += 1
            if left <= right:
                for x in range(top, down+1):
                    crop_boxes.append([X_points[x], Y_points[right], min(X_points[x] + split_width, img_w), min(Y_points[right] + split_height, img_h)])
                right -= 1
            if top <= down:
                for y in np.flip(range(left, right+1)):
                    crop_boxes.append([X_points[down], Y_points[y], min(X_points[down] + split_width, img_w), min(Y_points[y] + split_height, img_h)])
                down -= 1
            if left <= right:
                for x in np.flip(range(top, down+1)):
                    crop_boxes.append([X_points[x], Y_points[left], min(X_points[x] + split_width, img_w), min(Y_points[left] + split_height, img_h)])
                left += 1

    elif load == 'unclockwise':
        top = 0
        down = len(Y_points)-1
        left = 0
        right = len(X_points)-1
        while top <= down or left <= right:
            if top <= down:
                for y in range(left, right+1):
                    crop_boxes.append([X_points[top], Y_points[y], min(X_points[top] + split_width, img_w), min(Y_points[y] + split_height, img_h)])
                top += 1
            if left <= right:
                for x in range(top, down+1):
                    crop_boxes.append([X_points[x], Y_points[right], min(X_points[x] + split_width, img_w), min(Y_points[right] + split_height, img_h)])
                right -= 1
            if top <= down:
                for y in np.flip(range(left, right+1)):
                    crop_boxes.append([X_points[down], Y_points[y], min(X_points[down] + split_width, img_w), min(Y_points[y] + split_height, img_h)])
                down -= 1
            if left <= right:
                for x in np.flip(range(top, down+1)):
                    crop_boxes.append([X_points[x], Y_points[left], min(X_points[x] + split_width, img_w), min(Y_points[left] + split_height, img_h)])
                left += 1
        crop_boxes = crop_boxes[::-1]
    return np.asarray(crop_boxes)

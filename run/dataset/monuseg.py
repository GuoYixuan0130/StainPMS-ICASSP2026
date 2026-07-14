import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io
import scipy.io as sio

from stainpms.candidate import compute_b_candidates_oncrop, compute_baseline_center_candidates
from resimixpms.manifests import (
    ManifestPreflightError,
    load_allowed_image_names,
    load_crop_plan,
    load_crop_records,
    validate_manifest_patient_isolation,
)
from resimixpms.runtime import ResiMixAugmentor
from resimixpms.transplant import mask_medoid
from resimixpms.coverage import StaticCoverageCache


def _parse_int_set(value):
    if not value:
        return set()
    return {int(item.strip()) for item in str(value).split(",") if item.strip()}


def _safe_relative_name(name):
    candidate = Path(str(name).replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ManifestPreflightError(f"unsafe manifest image name: {name}")
    return candidate


def _resolve_manifest_image_name(image_root, name):
    """Resolve only the one manifest-named image, never enumerate a directory."""
    root = Path(image_root).resolve()
    relative = _safe_relative_name(name)
    candidates = [relative]
    if not relative.suffix:
        candidates.extend(relative.with_suffix(ext) for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"))
    existing = []
    for candidate in candidates:
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ManifestPreflightError(f"manifest image escapes image root: {name}") from exc
        if resolved.is_file():
            existing.append(resolved)
    if len(existing) != 1:
        raise ManifestPreflightError(
            f"manifest image must resolve to exactly one file ({name}); found {len(existing)}"
        )
    return str(existing[0].relative_to(root)).replace("\\", "/")


def _path_stem(path):
    return Path(path).stem


class MONUSEG(Dataset):
    def __init__(self, args, cfgs, data_path, load, mode='train', source_split=None):
        self.data_path = data_path
        self.mode = mode
        self.source_split = source_split or mode
        if self.source_split not in ("train", "test"):
            raise ValueError(f"unsupported MONUSEG source split: {self.source_split}")
        self._runtime_args = args
        self._mm_args = cfgs
        self.data_identity = str(getattr(args, "data_identity", "") or "")
        self._enforce_formal_root_isolation()
        self.crop_size = args.crop_size
        self.overlap = args.overlap
        self.load = load
        self.image_root, self.label_root = self._split_roots(self.source_split)
        self.paths = self._load_manifest_paths(self.source_split)

        train_crop_manifest = str(getattr(args, "train_crop_manifest", "") or "")
        self._train_crop_schedule = None
        if train_crop_manifest:
            train_crop_records, self._train_crop_schedule = load_crop_plan(
                train_crop_manifest,
                allowed_image_names=self.paths if self.source_split == "train" else None,
                expected_crop_size=self.crop_size if self.data_identity == "monuseg_lite" else None,
                expected_overlap=self.overlap if self.data_identity == "monuseg_lite" else None,
                expected_load=self.load if self.data_identity == "monuseg_lite" else None,
                expected_epochs=10 if self.data_identity == "monuseg_lite" else None,
            )
            self._train_crop_records_by_stem = self._records_by_stem(
                train_crop_records, self.paths if self.source_split == "train" else None
            )
        else:
            self._train_crop_records_by_stem = defaultdict(list)
        self._eval_crop_records_by_stem = self._crop_records_by_stem(
            str(getattr(args, "eval_crop_manifest", "") or ""),
            self.paths if self.source_split == "test" else None,
        )
        self._active_eval_crop_records = None
        self._active_eval_crop_schedule = None
        self._test_patch_records = []
        if mode == "test" and self.source_split == "test" and self._eval_crop_records_by_stem:
            by_stem = {_path_stem(path): path for path in self.paths}
            self._test_patch_records = []
            for records in self._eval_crop_records_by_stem.values():
                for record in records:
                    stem = _path_stem(record["image_name"])
                    if stem not in by_stem:
                        raise ManifestPreflightError(
                            f"evaluation crop references image outside test manifest: {record['image_name']}"
                        )
                    sample = dict(record)
                    sample["path"] = by_stem[stem]
                    self._test_patch_records.append(sample)
        elif mode == "test" and self.source_split == "train":
            # Static coverage must use test-time transforms while touching only
            # manifest-admitted training files and, for MoNuSeg-Lite, only the
            # frozen training crops.
            self._active_eval_crop_records = self._train_crop_records_by_stem
            self._active_eval_crop_schedule = self._train_crop_schedule

        self.num_mask_per_img = 150
        self.num_classes = 1

        self.resimix_enabled = bool(
            mode == "train" and getattr(args, "resimix_enabled", False)
        )
        self._build_transforms(cfgs)

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
        self.coverage_manifest_path = str(getattr(args, "coverage_manifest", "") or "")
        self._static_coverage_cache = None
        self._baseline_cache = {}
        if self._need_b and self.baseline_masks_dir:
            if self.coverage_manifest_path:
                manifest_path = Path(self.coverage_manifest_path)
                if manifest_path.name != "coverage_manifest.json":
                    raise ValueError("--coverage_manifest must name coverage_manifest.json")
                self._static_coverage_cache = StaticCoverageCache.open(manifest_path.parent)
                expected_ids = {_path_stem(path) for path in self.paths}
                if set(self._static_coverage_cache.image_ids) != expected_ids:
                    raise RuntimeError("static coverage image set does not exactly match the training manifest")
                for name in sorted(expected_ids):
                    self._baseline_cache[name] = self._static_coverage_cache.load(name)
            else:
                for p in self.paths:
                    name = _path_stem(p)
                    npy_path = os.path.join(self.baseline_masks_dir, name + ".npy")
                    if not os.path.isfile(npy_path):
                        if not self.pms_self_bootstrap:
                            raise FileNotFoundError(
                                f"missing immutable static coverage cache for manifest image {p}: {npy_path}"
                            )
                        continue
                    self._baseline_cache[name] = np.load(npy_path).astype(np.int32)
            if self.pms_enabled and self.pms_self_bootstrap:
                print("[MONUSEG PMS] self-bootstrap coverage cache active; "
                      "the initial online cache will be populated before epoch 0.")
            else:
                if len(self._baseline_cache) != len(self.paths):
                    raise RuntimeError("static coverage cache is incomplete; refusing partial-cache training")
                print(f"[MONUSEG PMS] loaded {len(self._baseline_cache)}/{len(self.paths)} "
                      f"precomputed baseline masks from {self.baseline_masks_dir}")
            if self.pms_enabled and self.pms_baseline_prompts:
                max_msg = "all" if self.pms_preserve_max_prompts <= 0 else str(self.pms_preserve_max_prompts)
                print(f"[MONUSEG PMS] coverage-preservation prompts enabled; max_per_crop={max_msg}")

        self._resimix_epoch = 0
        self._resimix_augmentor = None
        if self.resimix_enabled:
            config_path = str(getattr(args, "resimix_config", "") or "")
            if not config_path:
                raise ValueError("ResiMix dataset requires --resimix_config")
            self._resimix_augmentor = ResiMixAugmentor(config_path)
            if not self.coverage_manifest_path:
                raise ValueError("ResiMix dataset requires a sealed coverage manifest")
            self._resimix_augmentor.validate_formal_bindings(
                dataset=self.data_identity,
                train_manifest=str(getattr(args, "train_manifest", "") or ""),
                train_crop_manifest=str(getattr(args, "train_crop_manifest", "") or "") or None,
                coverage_manifest=self.coverage_manifest_path,
            )

    def _enforce_formal_root_isolation(self):
        """Keep MoNuSeg-Lite on frozen train material even outside the driver."""
        if self.data_identity != "monuseg_lite":
            return
        train_image = str(getattr(self._runtime_args, "train_image_root", "") or "")
        train_label = str(getattr(self._runtime_args, "train_label_root", "") or "")
        if not train_image or not train_label:
            raise ManifestPreflightError("MoNuSeg-Lite requires explicit admitted train image/label roots")
        if not str(getattr(self._runtime_args, "train_manifest", "") or ""):
            raise ManifestPreflightError("MoNuSeg-Lite requires its frozen train manifest")
        if not str(getattr(self._runtime_args, "train_crop_manifest", "") or ""):
            raise ManifestPreflightError("MoNuSeg-Lite requires its frozen training-crop manifest")
        if self.source_split != "test":
            return
        test_image = str(getattr(self._runtime_args, "test_image_root", "") or "")
        test_label = str(getattr(self._runtime_args, "test_label_root", "") or "")
        if not test_image or not test_label:
            raise ManifestPreflightError("MoNuSeg-Lite development roots must be explicit train roots")
        if Path(train_image).resolve() != Path(test_image).resolve() or Path(train_label).resolve() != Path(test_label).resolve():
            raise ManifestPreflightError("MoNuSeg-Lite may not use an official-test root")
        if not str(getattr(self._runtime_args, "test_manifest", "") or ""):
            raise ManifestPreflightError("MoNuSeg-Lite requires its frozen six-image development manifest")
        if not str(getattr(self._runtime_args, "eval_crop_manifest", "") or ""):
            raise ManifestPreflightError("MoNuSeg-Lite requires its frozen twelve-patch evaluation manifest")

    def _split_roots(self, split):
        image_root = str(getattr(self._runtime_args, f"{split}_image_root", "") or "")
        label_root = str(getattr(self._runtime_args, f"{split}_label_root", "") or "")
        if bool(image_root) != bool(label_root):
            raise ManifestPreflightError(
                f"{split}_image_root and {split}_label_root must be supplied together"
            )
        if image_root:
            return image_root, label_root
        if split == "train":
            return self.data_path + "/train_12/images", self.data_path + "/train_12/labels"
        if split == "test":
            return self.data_path + "/test/images", self.data_path + "/test/labels"
        raise ValueError(f"unsupported MONUSEG source split: {split}")

    def _manifest_for_mode(self, split):
        return str(getattr(self._runtime_args, f"{split}_manifest", "") or "")

    def _load_manifest_paths(self, split):
        image_root, label_root = self._split_roots(split)
        manifest = self._manifest_for_mode(split)
        if self.data_identity and not manifest:
            raise ManifestPreflightError(
                f"{self.data_identity} requires an explicit {split}_manifest; directory enumeration is forbidden"
            )
        if manifest:
            if self.data_identity == "tnbc":
                allowed_value = getattr(self._runtime_args, f"{split}_allowed_patient_ids", "")
                if not allowed_value:
                    allowed_value = getattr(self._runtime_args, "allowed_patient_ids", "")
                allowed = _parse_int_set(allowed_value)
                forbidden = _parse_int_set(getattr(self._runtime_args, "forbidden_patient_ids", "9,10,11"))
                validate_manifest_patient_isolation(manifest, allowed, forbidden)
            names = load_allowed_image_names(manifest)
            paths = [_resolve_manifest_image_name(image_root, name) for name in names]
        else:
            paths = sorted(os.listdir(image_root))
        stems = [_path_stem(path) for path in paths]
        if len(stems) != len(set(stems)):
            raise ManifestPreflightError("manifest image names must have unique filename stems")
        for path in paths:
            label_path = Path(label_root) / Path(path).with_suffix(".mat")
            if not label_path.is_file():
                raise ManifestPreflightError(f"missing label for manifest image {path}: {label_path}")
        return paths

    def _crop_records_by_stem(self, manifest, allowed_paths):
        if not manifest:
            return defaultdict(list)
        return self._records_by_stem(load_crop_records(manifest), allowed_paths)

    def _records_by_stem(self, records, allowed_paths):
        result = defaultdict(list)
        allowed_stems = {_path_stem(path) for path in allowed_paths} if allowed_paths else None
        for record in records:
            stem = _path_stem(record["image_name"])
            if allowed_stems is not None and stem not in allowed_stems:
                raise ManifestPreflightError(
                    f"crop manifest references image outside the corresponding image manifest: {record['image_name']}"
                )
            result[stem].append(record)
        return result

    def _build_transforms(self, cfgs):
        specs = [dict(item) for item in cfgs.data.get(self.mode).transform]

        def make(spec_list):
            built = []
            for spec in spec_list:
                params = dict(spec)
                transform_type = params.pop("type")
                built.append(getattr(A, transform_type)(**params))
            return built

        # Keep the original one-Compose path for every normal crop.  This is
        # essential for pixel-exact Static-PMS / ResiMix warm-up equivalence:
        # splitting Albumentations would consume random state differently even
        # before the epoch-2 augmentation is allowed to run.
        self.transform = A.Compose(make(specs) + [ToTensorV2()], p=1)
        self._resimix_post_transform = None
        self._resimix_normalize_mean = None
        self._resimix_normalize_std = None
        self._resimix_normalize_scale = None
        if not self.resimix_enabled:
            return
        normalize_specs = [spec for spec in specs if spec.get("type") == "Normalize"]
        if len(normalize_specs) != 1:
            raise ValueError("ResiMix requires exactly one terminal Normalize transform")
        self._resimix_post_transform = A.Compose(make(normalize_specs) + [ToTensorV2()], p=1)
        normalize = normalize_specs[0]
        self._resimix_normalize_mean = np.asarray(normalize.get("mean", (0.485, 0.456, 0.406)), dtype=np.float32)
        self._resimix_normalize_std = np.asarray(normalize.get("std", (0.229, 0.224, 0.225)), dtype=np.float32)
        self._resimix_normalize_scale = float(normalize.get("max_pixel_value", 255.0))

    def set_epoch(self, epoch):
        self._resimix_epoch = int(epoch)

    def consume_resimix_events(self):
        if self._resimix_augmentor is None:
            return []
        return self._resimix_augmentor.consume_events()

    def use_train_split_for_evaluation(self):
        """Switch a test-transform view to train records without enumerating directories."""
        self.source_split = "train"
        self.image_root, self.label_root = self._split_roots("train")
        self.paths = self._load_manifest_paths("train")
        self._test_patch_records = []
        self._active_eval_crop_records = self._train_crop_records_by_stem
        self._active_eval_crop_schedule = self._train_crop_schedule

    def _resimix_restore_rgb(self, normalized_image):
        """Invert the formal terminal Normalize solely for a successful transplant.

        The original composed output is retained unchanged when no synthetic
        nucleus is accepted, preserving the canonical Static-PMS tensor path.
        """
        if self._resimix_normalize_mean is None:
            raise RuntimeError("ResiMix normalization metadata is unavailable")
        if torch.is_tensor(normalized_image):
            values = normalized_image.detach().cpu().numpy()
        else:
            values = np.asarray(normalized_image)
        if values.ndim != 3 or values.shape[0] != 3:
            raise ValueError("ResiMix expects a CHW normalized image tensor")
        rgb = values.transpose(1, 2, 0) * self._resimix_normalize_std + self._resimix_normalize_mean
        return np.rint(np.clip(rgb * self._resimix_normalize_scale, 0.0, self._resimix_normalize_scale)).astype(np.uint8)

    def evaluation_crop_boxes(self, image_name, image_shape, default_boxes):
        if self._active_eval_crop_schedule is not None:
            return self._active_eval_crop_schedule.select_boxes(
                image_name, default_boxes, union=True
            )
        records = self._active_eval_crop_records
        if not records:
            return default_boxes
        stem = _path_stem(image_name)
        selected = records.get(stem, [])
        if not selected:
            raise ManifestPreflightError(f"no frozen training crop for {image_name}")
        height, width = int(image_shape[0]), int(image_shape[1])
        boxes = []
        for record in selected:
            x, y = int(record["x"]), int(record["y"])
            crop_w, crop_h = int(record["width"]), int(record["height"])
            if x + crop_w > width or y + crop_h > height:
                raise ManifestPreflightError(f"frozen crop lies outside {image_name}")
            boxes.append([x, y, x + crop_w, y + crop_h])
        return boxes

    def reload_baseline_masks(self):
        """Re-read baseline_masks_dir/*.npy into _baseline_cache in place.

        Used by main.py's iterative refresh path after the current model has
        overwritten the .npy files. No-op when PMS is not enabled.
        """
        if not (self._need_b and self.baseline_masks_dir):
            return 0
        if self._static_coverage_cache is not None:
            raise RuntimeError("immutable static coverage cannot be refreshed")
        n_old = len(self._baseline_cache)
        self._baseline_cache.clear()
        for p in self.paths:
            name = _path_stem(p)
            npy_path = os.path.join(self.baseline_masks_dir, name + ".npy")
            if not os.path.isfile(npy_path):
                raise FileNotFoundError(f"missing refreshed coverage map for {p}: {npy_path}")
            self._baseline_cache[name] = np.load(npy_path).astype(np.int32)
        n_new = len(self._baseline_cache)
        print(f"[MONUSEG PMS] reloaded {n_new}/{len(self.paths)} baseline masks "
              f"from {self.baseline_masks_dir} (was {n_old})")
        return n_new

    def __len__(self):
        return len(self._test_patch_records) if self._test_patch_records else len(self.paths)

    def __getitem__(self, index):

        """Get the images"""
        patch_record = None
        if self.mode == "test" and self._test_patch_records:
            patch_record = self._test_patch_records[index]
            path = patch_record["path"]
        else:
            path = self.paths[index]

        image_path = os.path.join(self.image_root, path)
        mask_path = os.path.join(self.label_root, str(Path(path).with_suffix('.mat')))

        img = io.imread(image_path)[..., :3]
        mask = load_maskfile(mask_path)

        output_name = _path_stem(path)
        if patch_record is not None:
            x, y = int(patch_record["x"]), int(patch_record["y"])
            width, height = int(patch_record["width"]), int(patch_record["height"])
            if x + width > img.shape[1] or y + height > img.shape[0]:
                raise ManifestPreflightError(f"evaluation patch lies outside {path}")
            img = img[y:y + height, x:x + width]
            mask = mask[y:y + height, x:x + width]
            output_name = f"{output_name}__x{x}_y{y}_w{width}_h{height}"

        # PMS: stack precomputed baseline mask as 3rd mask channel so
        # albumentations augments it together with image + GT (keeps alignment).
        baseline_attached = False
        if self.mode == 'train' and self._need_b and self._baseline_cache:
            name_no_ext = _path_stem(path)
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
            frozen_crops = self._train_crop_records_by_stem.get(_path_stem(path), [])
            if self._train_crop_schedule is not None:
                crop_boxes = self._train_crop_schedule.select_boxes(
                    path, crop_boxes, epoch=self._resimix_epoch
                )
            elif frozen_crops:
                crop_boxes = []
                for record in frozen_crops:
                    x, y = int(record["x"]), int(record["y"])
                    width, height = int(record["width"]), int(record["height"])
                    if width != self.crop_size or height != self.crop_size:
                        raise ManifestPreflightError(
                            f"training crop must be {self.crop_size}x{self.crop_size}: {record}"
                        )
                    if x + width > img.shape[2] or y + height > img.shape[1]:
                        raise ManifestPreflightError(f"frozen training crop lies outside {path}: {record}")
                    crop_boxes.append([x, y, x + width, y + height])
            for idx, crop_box in enumerate(crop_boxes):
                x1, y1, x2, y2 = crop_box
                img_c = img[..., y1:y2, x1:x2].transpose(1, 2, 0)
                mask_c = mask[..., y1:y2, x1:x2].transpose(1, 2, 0)

                synthetic_instance_id = None
                synthetic_event_index = None
                # Always execute the historical Compose exactly once.  During
                # warm-up (epochs 0--1), non-selected attempts, and rejected
                # proposals, its output is returned untouched, so ResiMix is
                # not an accidental change to the natural augmentation path.
                res = self.transform(image=img_c, mask=mask_c)
                img_c, mask_c = list(res.values())
                if self.resimix_enabled and self._resimix_augmentor.enabled_for_epoch(self._resimix_epoch):
                    raw_mask_c = mask_c.detach().cpu().numpy() if torch.is_tensor(mask_c) else np.asarray(mask_c)
                    if raw_mask_c.ndim != 3 or raw_mask_c.shape[-1] != 3:
                        raise ValueError("ResiMix requires instance/type/static-coverage mask channels")
                    augmentation = self._resimix_augmentor.augment(
                        self._resimix_restore_rgb(img_c),
                        raw_mask_c[..., 0],
                        raw_mask_c[..., 1],
                        raw_mask_c[..., 2],
                        epoch=self._resimix_epoch,
                        sample_key=f"{_path_stem(path)}:{idx}:{x1}:{y1}",
                    )
                    synthetic_instance_id = augmentation.synthetic_instance_id
                    synthetic_event_index = augmentation.event_index
                    if synthetic_instance_id is not None:
                        raw_mask_c = np.stack(
                            [augmentation.instance_map, augmentation.type_map, augmentation.coverage_map], axis=-1
                        )
                        res = self._resimix_post_transform(image=augmentation.image, mask=raw_mask_c)
                        img_c, mask_c = list(res.values())

                ori_shape = mask_c.shape[:2]
                inst_map, type_map = mask_c[..., 0], mask_c[..., 1]
                synthetic_medoid_xy = None
                if synthetic_instance_id is not None:
                    inst_np = inst_map.detach().cpu().numpy() if torch.is_tensor(inst_map) else np.asarray(inst_map)
                    synthetic_mask = inst_np == int(synthetic_instance_id)
                    if not synthetic_mask.any():
                        raise RuntimeError("synthetic instance disappeared after ResiMix post-transform")
                    medoid_y, medoid_x = mask_medoid(synthetic_mask)
                    synthetic_medoid_xy = np.asarray([medoid_x, medoid_y], dtype=np.float32)
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

                    # A successful synthetic transplant is deliberately an
                    # uncovered residual positive.  De-duplicate only natural
                    # residual H-peak prompts within six pixels, then retain
                    # the synthetic medoid before any preserve prompts.
                    if synthetic_medoid_xy is not None:
                        if len(pos_coords):
                            distances = np.linalg.norm(pos_coords - synthetic_medoid_xy[None, :], axis=1)
                            keep_natural = distances > 6.0
                            pos_coords = pos_coords[keep_natural]
                            pos_inst_ids = pos_inst_ids[keep_natural]
                        pos_coords = np.concatenate(
                            [pos_coords, synthetic_medoid_xy.reshape(1, 2)], axis=0
                        ).astype(np.float32, copy=False)
                        pos_inst_ids = np.concatenate(
                            [pos_inst_ids, np.asarray([synthetic_instance_id], dtype=np.int32)], axis=0
                        )
                        pos_weights = np.full(
                            len(pos_coords), 1.0 / max(1, len(pos_coords)), dtype=np.float32
                        )
                        self._resimix_augmentor.mark_prompt_added(synthetic_event_index)

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
            
            return img.to(torch.float32),inst_map, type_map.squeeze(0),prompt_points_all.squeeze(1), prompt_labels_all, binary_mask ,torch.as_tensor(ori_shape),index,output_name

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

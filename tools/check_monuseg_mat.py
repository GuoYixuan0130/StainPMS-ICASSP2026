from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.io import loadmat
from skimage import io


def count_xml_regions(xml_path):
    root = ET.parse(xml_path).getroot()
    count = 0
    for region in root.iter("Region"):
        vertices = region.find("Vertices")
        if vertices is not None and len(vertices.findall("Vertex")) >= 3:
            count += 1
    return count


def check_split(root, split):
    root = Path(root)
    img_dir = root / split / "images"
    xml_dir = root / split / "xml"
    label_dir = root / split / "labels"

    tif_files = sorted(img_dir.glob("*.tif"))

    print(f"\nChecking {split}: {len(tif_files)} images")

    bad = 0

    for img_path in tif_files:
        name = img_path.stem
        xml_path = xml_dir / f"{name}.xml"
        mat_path = label_dir / f"{name}.mat"

        if not xml_path.exists():
            print("[Missing XML]", xml_path)
            bad += 1
            continue

        if not mat_path.exists():
            print("[Missing MAT]", mat_path)
            bad += 1
            continue

        img = io.imread(img_path)
        mat = loadmat(mat_path)

        if "inst_map" not in mat:
            print("[No inst_map]", mat_path, mat.keys())
            bad += 1
            continue

        inst_map = mat["inst_map"]

        xml_count = count_xml_regions(xml_path)
        mat_count = int(inst_map.max())

        shape_ok = img.shape[:2] == inst_map.shape
        ids = np.unique(inst_map)
        ids = ids[ids > 0]
        contiguous_ok = len(ids) == mat_count

        print(
            name,
            "| image:", img.shape[:2],
            "| mat:", inst_map.shape,
            "| xml regions:", xml_count,
            "| mat instances:", mat_count,
            "| shape_ok:", shape_ok,
            "| contiguous_ok:", contiguous_ok,
        )

        if not shape_ok or not contiguous_ok or abs(xml_count - mat_count) > 0:
            bad += 1

    print(f"\n{split} finished. suspicious files: {bad}")


if __name__ == "__main__":
    check_split("data/monuseg", "train_12")
    check_split("data/monuseg", "test")
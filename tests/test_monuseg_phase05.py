import argparse
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from tools.audit_monuseg_xml_labels import audit_xml_label, polygon_self_intersects
from tools.audit_tcga_metadata import audit_metadata
from tools.build_monuseg_manifests import build_manifests


class MonusegPhase05Tests(unittest.TestCase):
    def test_extended7_offline_gdc_metadata_is_ordered_and_complete(self):
        config = json.loads(
            Path("configs/manifests/monuseg_release_v1.json").read_text(encoding="utf-8")
        )
        hits = []
        for row in reversed(config["extended7"]):
            case = row["case"]
            tss = case.split("-")[1]
            hits.append(
                {
                    "submitter_id": case,
                    "project": {"project_id": row["tcga_project"], "name": "Synthetic"},
                    "primary_site": row["primary_site"],
                    "disease_type": row["disease"],
                    "tissue_source_site": {
                        "code": tss,
                        "name": row["tissue_source_site"]["name"],
                        "project": row["tissue_source_site"]["project"],
                        "bcr_id": row["tissue_source_site"]["bcr_id"],
                    },
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            response = Path(tmp) / "gdc.json"
            response.write_text(json.dumps({"data": {"hits": hits}}), encoding="utf-8")
            report = audit_metadata(
                argparse.Namespace(
                    release_config="configs/manifests/monuseg_release_v1.json",
                    offline_response=str(response),
                    save_raw_response="",
                )
            )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            [record["case"] for record in report["records"]],
            [row["case"] for row in config["extended7"]],
        )

    def test_xml_regions_expose_legacy_merged_identity(self):
        xml = b"""<Annotations><Annotation><Regions>
        <Region Id='1'><Vertices>
          <Vertex X='1' Y='1'/><Vertex X='3' Y='1'/>
          <Vertex X='3' Y='3'/><Vertex X='1' Y='3'/>
        </Vertices></Region>
        <Region Id='2'><Vertices>
          <Vertex X='6' Y='6'/><Vertex X='8' Y='6'/>
          <Vertex X='8' Y='8'/><Vertex X='6' Y='8'/>
        </Vertices></Region>
        </Regions></Annotation></Annotations>"""
        legacy = np.zeros((10, 10), dtype=np.int32)
        legacy[1:4, 1:4] = 1
        legacy[6:9, 6:9] = 1
        report, candidate = audit_xml_label(xml, legacy)
        self.assertEqual(report["xml_region_count"], 2)
        self.assertEqual(report["candidate_effective_instance_count"], 2)
        self.assertEqual(report["legacy_instance_count"], 1)
        self.assertEqual(
            report["legacy_ids_best_matched_by_multiple_xml_regions"]["1"], [1, 2]
        )
        self.assertEqual(report["legacy_disconnected_instance_ids"], {"1": 2})
        self.assertEqual(np.unique(candidate).tolist(), [0, 1, 2])

    def test_polygon_self_intersection(self):
        bow_tie = np.asarray([(1, 1), (4, 4), (1, 4), (4, 1)], dtype=float)
        square = np.asarray([(1, 1), (4, 1), (4, 4), (1, 4)], dtype=float)
        self.assertTrue(polygon_self_intersects(bow_tie))
        self.assertFalse(polygon_self_intersects(square))

    def test_release_builder_materializes_30_7_14_and_ignores_test_png(self):
        config = json.loads(
            Path("configs/manifests/monuseg_release_v1.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_zip = root / "train.zip"
            test_zip = root / "test.zip"
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            train_rows = config["classic30"] + config["extended7"]
            with zipfile.ZipFile(train_zip, "w") as archive:
                for row in train_rows:
                    sample = row["sample_id"]
                    archive.writestr(f"Tissue Images/{sample}.tif", sample.encode())
                    archive.writestr(f"Annotations/{sample}.xml", b"<Annotations/>")
                    (images / f"{sample}.png").write_bytes(b"prepared-" + sample.encode())
                    (labels / f"{sample}.mat").write_bytes(b"legacy-" + sample.encode())
            with zipfile.ZipFile(test_zip, "w") as archive:
                for row in config["test14_expected_identities"]:
                    sample = row["sample_id"]
                    archive.writestr(f"Tissue Images/{sample}.tif", sample.encode())
                    archive.writestr(f"Annotations/{sample}.png", b"must-not-be-opened")
                    archive.writestr(f"Masks/{sample}.tif", b"must-not-be-opened")
            output = root / "out"
            report = build_manifests(
                argparse.Namespace(
                    release_config="configs/manifests/monuseg_release_v1.json",
                    train_archive=str(train_zip),
                    test_archive=str(test_zip),
                    prepared_image_root=str(images),
                    legacy_label_root=str(labels),
                    downloaded_at_utc="2026-07-21T00:00:00Z",
                    organ_info="",
                    official_converter="",
                    output_dir=str(output),
                )
            )
            self.assertEqual(report["counts"], {
                "classic30": 30,
                "extended7": 7,
                "download37": 37,
                "test14": 14,
            })
            self.assertEqual(report["train_test_isolation"]["status"], "isolated")
            self.assertFalse(report["test_access_attestation"]["opened_annotation_members"])
            test_manifest = json.loads(
                (output / "monuseg_test14_identity_v1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(test_manifest["record_count"], 14)
            self.assertTrue(
                all("label_path" not in record for record in test_manifest["records"])
            )


if __name__ == "__main__":
    unittest.main()

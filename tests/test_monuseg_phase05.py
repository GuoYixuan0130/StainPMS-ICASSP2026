import argparse
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import savemat
from skimage import io as skio

from tools.audit_monuseg_xml_labels import (
    audit_manifest,
    audit_xml_label,
    polygon_intersection_diagnostics,
    polygon_self_intersects,
)
from tools.audit_tcga_metadata import audit_metadata
from tools.build_monuseg_manifests import build_manifests
from tools.summarize_monuseg_xml_audit import render_summary


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
        closed_square = np.asarray(
            [(1, 1), (4, 1), (4, 4), (1, 4), (1, 1)], dtype=float
        )
        self.assertTrue(polygon_self_intersects(bow_tie))
        self.assertFalse(polygon_self_intersects(square))
        self.assertFalse(polygon_self_intersects(closed_square))
        self.assertEqual(
            polygon_intersection_diagnostics(bow_tie)["proper_crossing_count"], 1
        )
        touching = np.asarray([(0, 0), (2, 0), (2, 2), (0, 2), (2, 0)], dtype=float)
        self.assertEqual(
            polygon_intersection_diagnostics(touching)["proper_crossing_count"], 0
        )
        self.assertGreater(
            polygon_intersection_diagnostics(touching)["nonadjacent_touch_count"], 0
        )

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

    def test_local_source_tree_manifest_is_explicitly_provisional(self):
        config = json.loads(
            Path("configs/manifests/monuseg_release_v1.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_images = root / "source_images"
            source_xml = root / "source_xml"
            prepared = root / "prepared"
            labels = root / "labels"
            for directory in (source_images, source_xml, prepared, labels):
                directory.mkdir()
            for row in config["classic30"] + config["extended7"]:
                sample = row["sample_id"]
                (source_images / f"{sample}.tif").write_bytes(b"source-" + sample.encode())
                (source_xml / f"{sample}.xml").write_bytes(b"<Annotations/>")
                (prepared / f"{sample}.tif").write_bytes(b"prepared-" + sample.encode())
                (labels / f"{sample}.mat").write_bytes(b"legacy-" + sample.encode())
            output = root / "out"
            report = build_manifests(
                argparse.Namespace(
                    release_config="configs/manifests/monuseg_release_v1.json",
                    train_archive="",
                    train_source_image_root=str(source_images),
                    train_source_xml_root=str(source_xml),
                    test_archive="",
                    prepared_image_root=str(prepared),
                    legacy_label_root=str(labels),
                    downloaded_at_utc="",
                    organ_info="",
                    official_converter="",
                    output_dir=str(output),
                )
            )
            self.assertEqual(report["status"], "partial_provenance")
            self.assertEqual(report["test14_identity_record_count"], 0)
            self.assertIn(
                "official_training_archive_byte_identity_missing",
                report["unresolved_provenance"],
            )
            self.assertFalse((output / "monuseg_test14_identity_v1.json").exists())
            manifest = json.loads(
                (output / "monuseg_challenge30_v1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["status"], "local_source_tree_snapshot_archive_identity_pending"
            )

    def test_xml_audit_accepts_hash_verified_local_source_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_image = root / "source.tif"
            source_xml = root / "source.xml"
            prepared = root / "prepared.tif"
            label = root / "legacy.mat"
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            skio.imsave(source_image, image, check_contrast=False)
            skio.imsave(prepared, image, check_contrast=False)
            source_xml.write_text(
                "<Annotations><Region Id='1'><Vertex X='1' Y='1'/>"
                "<Vertex X='3' Y='1'/><Vertex X='3' Y='3'/></Region></Annotations>",
                encoding="utf-8",
            )
            savemat(label, {"inst_map": np.zeros((8, 8), dtype=np.int32)})
            import hashlib

            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": "monuseg",
                        "role": "training_pool",
                        "access_policy": "train_annotations_allowed",
                        "source_archive": {"kind": "local_training_source_tree_snapshot"},
                        "records": [
                            {
                                "sample_id": "TCGA-AA-0000-01Z-00-DX1",
                                "subset": "classic30",
                                "source_image_member": source_image.name,
                                "source_image_path": str(source_image),
                                "source_image_sha256": digest(source_image),
                                "source_xml_member": source_xml.name,
                                "source_xml_path": str(source_xml),
                                "source_xml_sha256": digest(source_xml),
                                "image_path": str(prepared),
                                "image_sha256": digest(prepared),
                                "label_path": str(label),
                                "label_sha256": digest(label),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = audit_manifest(
                argparse.Namespace(manifest=str(manifest_path), regenerated_label_root="")
            )
        self.assertEqual(report["source_access_mode"], "local_source_tree")
        self.assertEqual(report["aggregates"]["download37"]["image_count"], 1)

    def test_key_findings_summary_lists_only_samples_requiring_review(self):
        report = {
            "status": "complete_with_xml_anomalies",
            "source_access_mode": "local_source_tree",
            "aggregates": {
                "classic30": {key: 0 for key in [
                    "xml_empty_region_count", "xml_invalid_vertex_region_count",
                    "xml_out_of_bounds_region_count", "xml_self_intersection_region_count",
                    "xml_nonadjacent_path_touch_region_count",
                    "xml_disconnected_raster_region_count", "candidate_fully_occluded_region_count",
                    "candidate_effective_instance_count", "legacy_instance_count",
                    "exact_instance_map_equal_image_count", "legacy_disconnected_image_count",
                ]}
            },
            "classic30_reported_count_comparison": {
                "reported_classic30_nuclei": 21623,
                "audited_classic30_xml_regions": 1,
                "delta": -21622,
            },
            "samples": [
                {
                    "sample_id": "clean",
                    "subset": "classic30",
                    "label_audit": {
                        "candidate_effective_instance_count": 1,
                        "legacy_instance_count": 1,
                        "candidate_fully_occluded_region_count": 0,
                        "xml_empty_region_count": 0,
                        "xml_out_of_bounds_region_count": 0,
                        "xml_self_intersection_region_count": 0,
                        "xml_nonadjacent_path_touch_region_count": 0,
                        "xml_disconnected_raster_region_count": 0,
                        "legacy_disconnected_instance_ids": {},
                    },
                },
                {
                    "sample_id": "review",
                    "subset": "extended7",
                    "label_audit": {
                        "candidate_effective_instance_count": 1,
                        "legacy_instance_count": 2,
                        "candidate_fully_occluded_region_count": 0,
                        "xml_empty_region_count": 0,
                        "xml_out_of_bounds_region_count": 0,
                        "xml_self_intersection_region_count": 0,
                        "xml_nonadjacent_path_touch_region_count": 0,
                        "xml_disconnected_raster_region_count": 0,
                        "legacy_disconnected_instance_ids": {},
                        "xml_region_count": 1,
                    },
                },
            ],
        }
        text = render_summary(report)
        self.assertIn("review", text)
        self.assertNotIn('"sample_id": "clean"', text)


if __name__ == "__main__":
    unittest.main()

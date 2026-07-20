import argparse
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools import extract_tnbc_p1_8


class TnbcPhase05Tests(unittest.TestCase):
    def test_selective_extract_never_opens_closed_patient_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / extract_tnbc_p1_8.OFFICIAL_FILENAME
            with zipfile.ZipFile(archive_path, "w") as archive:
                for patient, count in extract_tnbc_p1_8.EXPECTED_IMAGES_PER_PATIENT.items():
                    for index in range(1, count + 1):
                        archive.writestr(
                            f"TNBC/Slide_{patient:02d}/{index}.png",
                            f"slide-{patient}-{index}".encode(),
                        )
                        archive.writestr(
                            f"TNBC/GT_{patient:02d}/{index}.png",
                            f"gt-{patient}-{index}".encode(),
                        )
                archive.writestr("TNBC/Slide_09/1.png", b"sealed-slide")
                archive.writestr("TNBC/GT_09/1.png", b"sealed-gt")
            md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()  # noqa: S324
            output_root = root / "allowed"
            with patch.object(
                extract_tnbc_p1_8, "OFFICIAL_SIZE_BYTES", archive_path.stat().st_size
            ), patch.object(extract_tnbc_p1_8, "OFFICIAL_MD5", md5):
                report = extract_tnbc_p1_8.extract_allowed(
                    argparse.Namespace(
                        archive=str(archive_path),
                        downloaded_at_utc="2026-07-21T00:00:00Z",
                        output_root=str(output_root),
                    )
                )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(report["records"]["slide"]), 37)
            self.assertEqual(len(report["records"]["gt"]), 37)
            self.assertFalse((output_root / "Slide_09").exists())
            self.assertFalse((output_root / "GT_09").exists())
            self.assertFalse(
                report["access_attestation"]["closed_patient_member_content_opened"]
            )


if __name__ == "__main__":
    unittest.main()

import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from tools.download_phase05_sources import (
    attachment_filename,
    build_session,
    parse_registered_asset,
    register_existing_asset,
)


class Phase05DownloadTests(unittest.TestCase):
    def test_attachment_filename_variants(self):
        self.assertEqual(
            attachment_filename('attachment; filename="Training Data.zip"'),
            "Training Data.zip",
        )
        self.assertEqual(
            attachment_filename("attachment; filename*=UTF-8''Training%20Data.zip"),
            "Training Data.zip",
        )

    def test_attachment_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsafe"):
            attachment_filename('attachment; filename="../archive.zip"')

    def test_retry_policy_covers_transient_get_failures(self):
        session = build_session(retries=3, backoff_seconds=0.0)
        self.addCleanup(session.close)
        adapter = session.get_adapter("https://example.org")
        retry = adapter.max_retries
        self.assertEqual(retry.total, 3)
        self.assertEqual(retry.connect, 3)
        self.assertEqual(retry.read, 3)
        self.assertIn(503, retry.status_forcelist)
        self.assertIn("GET", retry.allowed_methods)

    def test_registered_source_hashes_without_copying(self):
        path = Path("/approved/Training Data.zip")
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "open", return_value=BytesIO(b"raw archive bytes")),
            patch.object(Path, "resolve", return_value=path),
        ):
            record = register_existing_asset("monuseg_train", path)
        self.assertEqual(record["acquisition"], "manual_browser_upload")
        self.assertEqual(record["filename"], "Training Data.zip")
        self.assertEqual(record["size_bytes"], len(b"raw archive bytes"))

    def test_register_existing_argument_requires_known_asset_and_path(self):
        asset, path = parse_registered_asset("monuseg_train=/tmp/train.zip")
        self.assertEqual(asset, "monuseg_train")
        self.assertEqual(path, Path("/tmp/train.zip"))
        with self.assertRaisesRegex(ValueError, "ASSET=PATH"):
            parse_registered_asset("unknown=/tmp/train.zip")


if __name__ == "__main__":
    unittest.main()

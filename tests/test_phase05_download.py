import unittest

from tools.download_phase05_sources import attachment_filename


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


if __name__ == "__main__":
    unittest.main()

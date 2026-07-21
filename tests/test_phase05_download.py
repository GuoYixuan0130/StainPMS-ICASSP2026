import unittest

from tools.download_phase05_sources import attachment_filename, build_session


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


if __name__ == "__main__":
    unittest.main()

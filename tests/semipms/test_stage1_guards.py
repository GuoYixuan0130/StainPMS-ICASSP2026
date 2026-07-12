import unittest

from semipms.guards import ImageRecord
from semipms.stage1_guards import CLOSED_STAGE1_PATIENTS, Stage1AccessGuard


class Stage1GuardTest(unittest.TestCase):
    def test_hidden_train_gt_is_post_training_only(self):
        guard = Stage1AccessGuard()
        train = ImageRecord(1, "01_1", "image", "label", "x")
        dev = ImageRecord(7, "07_1", "image", "label", "y")
        with self.assertRaises(PermissionError):
            guard.allow_hidden_train_audit_read(train)
        guard.freeze_training_configuration()
        with self.assertRaises(PermissionError):
            guard.allow_hidden_train_audit_read(train)
        guard.allow_development_label_read(dev)
        guard.mark_training_finished()
        guard.allow_hidden_train_audit_read(train)
        self.assertEqual((guard.hidden_train_label_reads, guard.development_label_reads), (1, 1))

    def test_closed_patient_set_is_separate_from_development(self):
        self.assertEqual(CLOSED_STAGE1_PATIENTS, frozenset({9, 10, 11}))
        guard = Stage1AccessGuard()
        with self.assertRaises(PermissionError):
            guard.allow_development_label_read(ImageRecord(9, "09_1", "image", "label", "z"))


if __name__ == "__main__":
    unittest.main()

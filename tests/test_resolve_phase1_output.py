import json
import unittest

from tools.resolve_phase1_output import resolve_output


class _File:
    def __init__(self, exists=True):
        self._exists = exists

    def is_file(self):
        return self._exists


class _Directory:
    def __init__(self, name, files=()):
        self.name = name
        self.files = set(files)

    def __truediv__(self, name):
        return _File(name in self.files)

    def __str__(self):
        return self.name


class _Summary:
    def __init__(self, parent, payload):
        self.parent = parent
        self.payload = payload

    def read_text(self, encoding="utf-8"):
        return json.dumps(self.payload)

    def __lt__(self, other):
        return str(self.parent) < str(other.parent)


class _Root:
    def __init__(self, summaries):
        self.summaries = summaries

    def glob(self, pattern):
        assert pattern == "*/summary.json"
        return self.summaries

    def __str__(self):
        return "diagnostics"


def _payload(*, status="complete", records=37):
    return {
        "dataset": "monuseg",
        "status": status,
        "manifest": {"processed_record_count": records},
    }


class ResolvePhase1OutputTests(unittest.TestCase):
    def test_resolves_only_complete_matching_output(self):
        valid = _Directory("optimized_full", ("gt_instances.csv", "images.json"))
        smoke = _Directory("smoke")
        root = _Root(
            [
                _Summary(smoke, _payload(status="smoke_only_partial", records=1)),
                _Summary(valid, _payload()),
            ]
        )
        resolved = resolve_output(
            root,
            dataset="monuseg",
            processed_records=37,
            required_files=("gt_instances.csv", "images.json"),
        )
        self.assertIs(resolved, valid)

    def test_rejects_ambiguous_outputs(self):
        root = _Root(
            [
                _Summary(_Directory("a"), _payload()),
                _Summary(_Directory("b"), _payload()),
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "expected exactly one"):
            resolve_output(
                root,
                dataset="monuseg",
                processed_records=37,
                required_files=(),
            )


if __name__ == "__main__":
    unittest.main()

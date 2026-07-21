"""TNBC manifest-only dataset adapter for the existing StainPMS crop path.

TNBC prepared labels use the same ``inst_map`` MAT representation as the
current MoNuSeg loader.  This adapter changes only manifest identity and log
labelling; it deliberately does not discover files or define a test split.
"""

from run.dataset.monuseg import MONUSEG


class TNBC(MONUSEG):
    """Use the shared PNG/TIFF + ``inst_map.mat`` crop implementation safely."""

    manifest_dataset = "tnbc"
    manifest_log_name = "TNBC"

    def __init__(self, *args, **kwargs):
        requested = kwargs.pop("manifest_dataset", self.manifest_dataset)
        if str(requested).lower() != self.manifest_dataset:
            raise ValueError(
                "TNBC loader accepts only manifests declaring dataset='tnbc'"
            )
        super().__init__(*args, manifest_dataset=self.manifest_dataset, **kwargs)

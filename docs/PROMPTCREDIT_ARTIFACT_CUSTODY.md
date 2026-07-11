# PromptCredit artifact custody

Date: 2026-07-11

## Stage 0 original recovery

The original PC-Stage 0 output was found in the user-owned, untracked staging
directory `temple/stage0_tnbc_router_train_six`.  It was copied, never moved or
edited, to:

```text
logs/promptcredit/stage0/af0fc4a_original/
```

The copy is explicitly labelled `recovered_original`.  Its six file hashes,
sizes, and preserved source modification times are in
`artifact_manifest.json` beside the copy.  In particular, the original report
SHA256 remains
`6f99fbf15efb85b8b24faef9c682b39175b619473b4598c8fc1d8bce1893be1e`.
The recovered report records code SHA
`af0fc4a7a5c64e9456c93b7ddf035a50313c72b9`, frozen checkpoint SHA
`44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781`, and
the fixed-six-image checksum
`d172ba88ebe645c4abd1a4bf78f8b7b66da60a58ba093b153754611f3bdf2ea6`.

No reproduction was needed and no artifact-incident report is warranted.

## Preserved protocol-invalid Stage 1 run

The earlier 100-step smoke at code SHA
`09c768f07f800da462ae93877f535c1d884214f7` is retained unchanged at:

```text
logs/promptcredit/stage1_smoke_invalid/09c768f/
```

Its disposition is **MECHANICALLY SUCCESSFUL, PROTOCOL-INVALID**.  Its
automatic `PASS` is not a formal scientific conclusion: train-mode dropout
affected evaluations and those evaluations consumed the shared RNG stream.
`artifact_manifest.json` in that directory lists the SHA256 of every preserved
file, including the original report
`80d752ff86d5055ac4c7539b4deea0a571455a6311c5afa68a9d06b64648691a`.

`temple/` remains a user-owned staging area and is not a formal sole artifact
location.  It must not be staged, deleted, renamed, or otherwise modified.

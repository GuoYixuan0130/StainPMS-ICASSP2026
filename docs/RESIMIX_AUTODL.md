# ResiMix-PMS AutoDL 执行

仅在 `research/resimix_pms` 的已提交干净 worktree 执行。driver 会先校验 canonical SHA、checkpoint、TNBC 1--8 manifest、MoNuSeg-Lite 原始 checksum，再执行单测、一次静态 coverage、donor bank、smoke、两臂各一次训练与报告。

先从已发布分支创建独立 AutoDL worktree（不能从任何退役路线续接）：

```bash
cd /root/autodl-tmp/projects/StainPMS-ICASSP2026
git fetch origin research/resimix_pms:refs/remotes/origin/research/resimix_pms
git merge-base --is-ancestor 2a1348cb7a1158a6f77aae2f92c168f9552d8068 refs/remotes/origin/research/resimix_pms
git worktree add -b research/resimix_pms /root/autodl-tmp/projects/StainPMS-ICASSP2026-resimix refs/remotes/origin/research/resimix_pms
```

先只读查看冻结 MoNuSeg-Lite JSON 的可用 JSON pointer：

```bash
cd /root/autodl-tmp/projects/StainPMS-ICASSP2026-resimix
python tools/inspect_resimix_frozen_bundle.py \
  --bundle /root/autodl-tmp/projects/StainPMS-ICASSP2026/.setpms/logs/setpms/stage1_dual_dev/20260714_172429_6a40ce194788
```

复制 `configs/resimixpms_stage1.example.json` 到工作区外的私有路径，填写 TNBC 的既有 1--6 / 7--8 manifest 和上一步显示的四个冻结 pointer。MoNuSeg-Lite 的 `test_*_root` 必须与 `train_*_root` 相同；这会阻止访问 official test。

正式执行：

```bash
python tools/run_resimix_stage1.py \
  --spec /root/autodl-tmp/resimixpms_stage1.json \
  --artifact-root /root/autodl-tmp/projects/StainPMS-ICASSP2026/logs/resimixpms/stage1_dual_dev
```

不要把生成的 `logs/`、coverage、donor payload、checkpoint 或 report 加入 Git。任何 checksum、manifest、数据隔离、标签或 NaN 失败都会停止；不应改 selector、重选 patch 或重跑调参。

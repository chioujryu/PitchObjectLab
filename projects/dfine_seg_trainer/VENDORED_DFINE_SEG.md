# Vendored D-FINE-seg Snapshot

- Upstream: https://github.com/ArgoHA/D-FINE-seg
- Commit: `0a0f0a12511568857922924854a63d06e1ae0fbd`
- License: Apache-2.0, preserved at `vendor/D-FINE-seg/LICENSE`

The wrapper keeps the upstream code under `vendor/D-FINE-seg` and runs it through Hydra using a generated runtime config. The local patch in `src/dl/dataset.py` adds extra Albumentations transforms consumed by `config/dfine_seg_train.yaml`.

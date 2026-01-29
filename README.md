# TD3B (TR2-D2 Peptide) — Minimal Release

This repository is a slimmed-down, anonymized release focused on the **multi‑target TD3B** training path. It keeps only the code needed to run `launch_multi_target.sh` plus the `baselines/` folder.

## Repository layout

```
TD3B/
  launch_multi_target.sh     # Main launcher for multi-target TD3B training
  finetune_multi_target.py   # Main training script
  finetune_utils.py          # Shared training utilities (merged)
  diffusion.py               # TR2-D2 diffusion model
  roformer.py                # RoFormer backbone wrapper
  noise_schedule.py          # Noise schedules
  peptide_mcts.py            # Base MCTS implementation
  td3b/                      # TD3B modules (losses, scoring, oracle, MCTS)
  scoring/                   # Scoring models/utilities
  tokenizer/                 # Tokenizer vocab/splits
  utils/                     # Misc utilities
  configs/                   # Configs (including finetune_config.py)
  baselines/                 # Baseline runners and scripts
```

## Launch multi-target TD3B training

1) Open `launch_multi_target.sh` and replace all `To Be Added` placeholders with your local paths.

   Required fields:
   - `BASE_PATH`
   - `PRETRAINED_CHECKPOINT`
   - `TRAIN_CSV`
   - `VAL_CSV` (optional but recommended)
   - `ORACLE_CKPT`
   - `ORACLE_TR2D2_CHECKPOINT`
   - `ORACLE_TOKENIZER_VOCAB`
   - `ORACLE_TOKENIZER_SPLITS`

2) Run:

```bash
bash launch_multi_target.sh
```

That script builds the full training command and starts `finetune_multi_target.py` with the configured options.

## Notes

- All absolute paths have been replaced with `To Be Added` for anonymous release. Replace them before running.
- The code expects the pretrained checkpoint, tokenizer files, and oracle checkpoint to be available on disk.


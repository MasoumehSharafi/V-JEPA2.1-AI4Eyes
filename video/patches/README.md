# Patches applied to the cloned vjepa2 checkout

`scripts/setup_vjepa.sh` applies these on top of the pinned commit
(`204698b45b3712590f06245fbfba32d3be539812`) with `git apply`. Vendoring a
patch instead of a full file copy keeps the diff against upstream visible
and small.

## `0001-vit-base-pretrained-encoder-save-best.patch`

Adds, to `app/vjepa_2_1/train.py`:
- `vit_base` support in the embed-dim dispatch (stock code only recognized
  large/giant/gigantic).
- A `pretrained_encoder` config field: initializes the encoder from an
  existing EMA-encoder checkpoint instead of from scratch -- this is what
  `scripts/pretrain_ssl.sh` relies on for domain-adaptive continued
  pretraining from the official V-JEPA2.1 release weights.
- A `frame_sample_rate` data option (mutually exclusive with `fps`) and
  `persistent_workers` passthrough.
- A `save_best` / `best.pth.tar` / `best_loss.txt` mechanism intended to
  track the lowest training loss seen so far.

**Known bug, reproduced faithfully (this is the exact diff that produced
the checkpoints referenced in ../README.md's results table):** the
`save_best` block and the `save_every_freq` block both ended up indented
one level too shallow -- outside the epoch loop instead of inside the
`if (epoch + 1) % CHECKPOINT_FREQ == 0` branch. In practice this means
`best.pth.tar` and any `save_every_freq` checkpoints are only ever written
once, right after the *final* epoch, using that epoch's loss -- not
continuously throughout training as the code reads like it should. It
does not crash (the final epoch always satisfies the `CHECKPOINT_FREQ`
check via `epoch == num_epochs - 1`, so `epoch_loss` is always defined by
the time the misplaced block runs), it just doesn't do real per-checkpoint
best-tracking.

If you want genuine per-epoch best-tracking, re-indent the `if save_best
and epoch_loss < best_loss:` block (and the `if save_every_freq > 0:`
block after it) to sit inside the `if (epoch + 1) % CHECKPOINT_FREQ == 0`
block, matching `save_checkpoint(epoch + 1, latest_path)`'s indentation.
We left the patch as-is rather than silently fixing someone else's vendored
training script.

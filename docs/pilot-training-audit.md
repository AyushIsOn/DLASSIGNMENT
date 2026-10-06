# Training audit, 2026-10-06

Audit by Codex from source; no GPU run and no external independent reviewer.

## Present and usable

- `_tokenize_chat_row` in `training/train_qlora.py` renders complete chat and
  generation-prefix templates, verifies prefix identity, masks prefix labels with
  -100 and rejects empty supervised targets or oversize examples. The training
  collator uses label_pad_token_id=-100. This is the right contract, but a real
  locked-tokenizer sample check is still required on the new dataset.
- BF16 LoRA starts from the locked base snapshot. Dataset/review hashes are in run
  identity; checkpoints with another run identity are not resumed.
- Metrics are printed and appended per step. SIGINT/SIGTERM request stopping at
  an optimizer-step boundary and a checkpoint save. The callback seals and packs
  a recovery archive. Validation loss, not test loss, selects the best checkpoint.
- The existing production gate requires 500/100/100 records. It remains unchanged.

## Blockers before a small pilot

1. A pilot dataset and manifest schema must be separate from production approval;
   the current 24 examples do not satisfy either the proposed plan or production.
   Source-family grouping and near-duplicate checks must join the current exact
   context/question checks. A valid hash alone does not establish source review.
2. Current evaluation/save interval is 25 optimizer steps. A short pilot ending
   earlier could finish without periodic validation or a best-checkpoint comparison.
   The pilot must compute an interval that guarantees evaluation/checkpoints,
   including the final step, and test this path.
3. Budget timing begins after model loading and tokenization. The callback's
   wall-time limit is a soft boundary checked at step end; a forward/backward pass,
   evaluation or checkpoint packing may overrun. It is not a hard instance-billing
   limit. Include an external watchdog/reserve and a recovery test before paid use.
4. Early interruptions during model initialization are not handled by the later
   signal callback. Abrupt termination/SIGKILL cannot guarantee a new checkpoint.
   Keep completed checkpoints on durable storage and verify cross-path restore.
5. `run_input` does not itself freeze a new raw-evaluation rubric. Bind pilot data,
   rubric, inference prompt format and training settings before comparison.
6. Do not reuse old test answers for tuning. Source families must be assigned before
   expansion; separate source pages do not alone establish topic independence.

## Next verification

Before loading model weights: data/manifest/split checks and tokenizer-only smoke.
On GPU: brief label/loss/throughput check, one save/resume interruption exercise,
and a bounded fresh pilot with a validation-based selection rule. Preserve the old
adapter and use a distinct output directory. No claim about full-H100/H200 utilization
or quality benefit follows from this source audit.

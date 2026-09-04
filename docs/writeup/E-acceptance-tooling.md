# Acceptance tooling: what we check before handing anything over

WP-E. Source: `bench/accept.sh`, and the checkpoint-selection analysis reproduced below.

## `accept.sh`

`~/hackathon-shared/bench/accept.sh [--fast] [--tag NAME] [--submission] <repo-or-branch> <ckpt>`

Clones the branch into a throwaway directory and runs the organisers' `eval.py` exactly as the
notebook does (severities 0-1 in 10 steps, grace 1000, full eval set) against that clean clone.
The point is that it tests **what ships**, not what happens to be lying around in a working tree.

Design decisions worth recording:

- **`PYTHONPATH`, not `pip install -e`.** Five instances share one Python environment with a
  single editable install; installing from the acceptance script would silently repoint every
  other instance's `embedding` package. See `E-bench-design.md`.
- **Preflight on free GPU memory**, defaulting to 8000 MiB with an `ACCEPT_NEED_MIB` override,
  because `eval.py` embeds at a hard-coded batch of 1024 and OOMs mid-run otherwise. It fails in
  one readable line instead of 200 lines of traceback. The default is sized for the transformer;
  the header documents the architecture split so nobody assumes a set encoder needs 8 GB.
- **`--fast`** stops `eval.py` once the area is printed and `auc_vs_severity.json` is on disk,
  skipping the trailing t-SNE, which is a very slow CPU job over the whole eval set. The area is
  identical either way. It is **not** the default: this script's original purpose is "does the
  grader's script run to completion", and stopping early no longer verifies that. `--fast` for
  per-candidate numbers, full run for the final submission check.
- **Smoke runs mark themselves.** With `ACCEPT_EVAL_PT` set (a small slice, for testing the script
  itself) it prints a warning that the area is not a result, and records `smoke_test: true` in the
  JSON. An early version claimed "full eval set" regardless, which was misleading.

Each run writes `runs/official_<tag>_<ts>.json`, which `ablation_table.py` joins onto the bench
row as the official-area column. No number is transcribed by hand.

## The checkpoint-selection hazard

The organisers' notebook selects its checkpoint as `sorted(glob("checkpoints/*.pth"))[-1]`.

`.` is `0x2E` and `_` is `0x5F`, so for a common stem:

```
robust_tagging_encoder_..._190000.pth
robust_tagging_encoder_..._201234.pth
robust_tagging_encoder_..._201234_bestauc.pth
robust_tagging_encoder_..._201234_last.pth   <- sorted(...)[-1] picks THIS
```

A `_last.pth` sibling sorts after both `.pth` and `_bestauc.pth`. So a training loop that also
writes a final-epoch checkpoint causes the notebook to silently grade the **final-epoch** model
instead of the best-validation one. Verified in both Python and shell sort.

`accept.sh` therefore checks three things:

1. **Clone side.** More than one `checkpoints/*.pth` on the branch is a hard failure, naming which
   file the notebook would pick. With `--submission`, *zero* is also a failure: the notebook's
   `latest_ckpt = ckpts[-1] if ckpts else None` yields `None` and the next cell asserts, so a
   branch shipping none is unusable to the organisers.
2. **Working tree.** This is the check that actually bites, because the notebook runs in a working
   directory, not a clone -- and the repo's `.gitignore` excludes `checkpoints/` and `*.pth`, so a
   fresh clone carries none by construction. It compares the checkpoint being accepted against
   what `sorted(glob(...))[-1]` would auto-select in that tree and warns when they differ. When
   they match it says so, so a clean tree is positively confirmed rather than merely not warned
   about.
3. **Subdirectories.** `checkpoints/smokes/` and `checkpoints/aux/` are outside the notebook's
   non-recursive glob and can never be auto-selected, so they are listed with a total size rather
   than failed -- the submitter should still see what is being handed over.

## The certification that shipped

Two runs, both on fresh clones of `submission-pma0-meanpt`:

```
PASS  commit=3ac262f  ckpt=rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth  area=0.8780   (full run)
PASS  commit=41d45a9  ckpt=rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth  area=0.8795   (--fast)
```

Guard trace on the shipped tip:

```
accept: fresh clone at 41d45a9
accept: eval.py will build preproc=PFPreProcessorMeanPt (preproc_type: PFPreProcessorMeanPt), encoder=PMAEncoder
accept: preproc matches --expect-preproc (PFPreProcessorMeanPt)
accept: encoder matches --expect-encoder (PMAEncoder)
accept: clone ships exactly one checkpoint: rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth
accept: scored checkpoint is byte-identical to the shipped one (sha256 2f34628ccd78...)
area under AUC-vs-severity curve: 0.8795
(No degradation) area under curve: 0.8201
```

**Why two runs.** Run 1 certified `3ac262f`; a README commit then moved the tip to `41d45a9`. The
tip is what the organisers receive, so a PASS for `3ac262f` would have certified a commit that does
not ship -- the same defect as the shipped-vs-scored checkpoint gap, relocated from the checkpoint
to the commit. Run 1 establishes that the code runs to completion; run 2 establishes that the
shipped hash passes every guard. That the two are the same code was proven at the git object level,
not asserted: `git diff --name-status` listed only `README.md`, and `models.py`, `preprocs.py`,
`train_config.yaml`, `data_config_eval.yaml`, `eval.py` and the checkpoint were identical blobs.

The corollary was applied afterwards: the results pair was NOT committed to the branch README,
because a commit on top of a certified tip makes the shipped tip uncertified again. It went into
`RULING.md` instead.

## Silent failures cluster where two objects share a schema

The guards are not uniformly valuable, and the reason is worth stating.

A **readout** mismatch is caught for free: `eval.py` calls `load_state_dict` strictly, so building
the wrong readout produces a key mismatch and a loud error. Verified on the a-pma branch, where the
PMA readout comes from a defaulted `readout='pma'` kwarg that `eval.py` never passes -- the shipped
checkpoint loads strictly into `eval.py`'s exact construction, so the default *is* the final choice
as the brief requires.

A **preprocessor** mismatch is not caught at all: `PFPreProcessor`, `PFPreProcessorAbsPt`,
`PFPreProcessorMaxPt` and `PFPreProcessorMeanPt` register identical `state_dict` keys (verified
pairwise), so a checkpoint trained with one loads silently into another and yields a plausible,
wrong area. Nothing downstream can detect it.

So `--expect-preproc` earns its place and `--expect-encoder` is mostly belt-and-braces. The general
form: **the dangerous silent failures are where two objects share a schema**, because that is
exactly where the runtime has nothing to check against. The root cause here is that the checkpoint
does not record which preprocessor produced it; `eval.py` reads only the keys it needs, so
`train.py` could store `preproc_type` and `encoder_class` in the checkpoint dict without breaking
the contract, and that would remove the class of error rather than guarding one call site. Future
work, not a change made tonight.

## An operational lesson worth carrying

A run of `accept.sh` died with `syntax error near unexpected token '('` at a line that was
perfectly valid, while `bash -n` passed. The cause was editing the script **while a copy was
executing**: bash reads a script incrementally by byte offset, so an in-place rewrite makes the
running process resume mid-token. The failure surfaced at the very end, after `eval.py` had done
all its work.

The hazard is shell-specific. CPython reads and compiles an entire source file before executing a
line, so editing a running `.py` is harmless (the exception being modules imported later). The
rule: **never write to a path a live shell process is executing** -- kill it first, or write to a
new filename. During convergence, shared scripts are frozen while anyone's run is in flight.

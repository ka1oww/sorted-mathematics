# Project agent memory

Project-intrinsic notes that should travel with the code. README.md covers what
the project is and DATA-PLAN.md is the data contract; neither is repeated here.

## Sharp edges

- **The frozen corpus is not reproducible from `src/merge.py` at head.**
  `data/private/questions.jsonl` holds 1,451 labelled rows and was frozen before
  the Q-prefix duplicate collapsing landed; re-running the merge now yields
  1,525. Every accuracy number in README.md and RESULTS.md is quoted against the
  frozen file, so do not regenerate it without also regenerating the split, both
  models and RESULTS.md, and updating the numbers everywhere.
- **A model outlives its split, and that is how a fake accuracy happens.** A
  classifier trained on an earlier split scores far too well on a later one,
  because rows that were training data have become test data. Retrain after any
  change to `src/split.py`, its seed, or the corpus.
  `tools/report_results.py` refuses to score a model file older than
  `data/private/test.jsonl` for this reason, so a "not measurable" row in
  RESULTS.md means a stale artefact rather than a missing one.
- `tools/measure_grouping_key_impact.py` needs `data/private/school_aliases.json`
  to report the school residual. Absent, that section is a no-op rather than an
  error, so a zero there can mean "no alias map" rather than "no residual".

## Running things

- Tests use synthetic fixtures only: `python3 -m pytest` needs no private data.
- `conftest.py` at the root is the one import shim. Modules import their
  siblings by bare name because they run as `python3 src/x.py`; tools do their
  own explicit insert. Do not add per-file `sys.path` blocks.
- Every writer goes through `assert_inside_project` in `src/paths.py`, and the
  two source roots are overridable with `EMBER_SOURCE_ROOT` and `TRIPLEMATH_PDF`.

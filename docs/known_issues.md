# Known Issues

Pre-existing test failures observed on 2026-08-27, verified as present at
commit `f7a3249` (i.e. they predate that day's changes). Recorded here so
they are tracked rather than rediscovered on each run.

The **root** `tests/` suite is fully green (5767 passed). Everything below is
confined to sub-project suites.

---

## veritas-rag — 1 failure

`tests/test_ingestion.py::TestExtractors::test_extract_pdf_roundtrip`

```
ModuleNotFoundError: No module named 'pypdf'
```

**Cause:** purely environmental. `pypdf==5.1.0` **is** already declared in
`veritas-rag/requirements.txt` (line 6) — it simply was not installed in the
image the suite ran under.

**Fix:** nothing to change in the repo. Install the declared dependencies
(`pip install -r veritas-rag/requirements.txt`) before running this suite.
Optionally guard the test with `pytest.importorskip("pypdf")` so a partial
environment skips rather than fails.

---

## suite-cast — 31 failures

Two distinct causes:

**1. Tests reference functions that do not exist (`ImportError`).**

```
ImportError: cannot import name 'length_of_stay_bucket' from 'app.features'
```

Affected: `tests/test_features.py::TestLengthOfStayBucket` and several
sibling classes in the same file.

**Cause:** the tests were written against an intended API that was never
implemented (or was later removed) in `app/features.py`.

**Fix:** either implement the missing helpers in `app/features.py`, or remove
the tests if the functionality was deliberately dropped. Decide which before
writing new code against that module.

**2. Model output shape mismatch.**

```
tests/test_model.py::test_predict_demand_output_keys[confidence]
```

`predict_demand` does not return the `confidence` key the test expects.

**Fix:** confirm whether `confidence` belongs in the response contract. If it
does, add it to `predict_demand`; if not, update the test and the API docs
together so they cannot drift apart again.

---

## forge-guard — 7 failures

`tests/test_validators.py`, including:

- `TestClampToRange::test_value_above_max_clamped`
- `TestZscoreOutlier::test_detects_outlier`

**Cause:** not yet diagnosed. These are assertion failures rather than import
errors, so the functions exist but do not behave as the tests expect — worth
checking whether the tests or the implementation drifted.

**Fix:** investigate `app/validators.py` against `tests/test_validators.py`.
Because these are behavioural rather than environmental, one side is wrong and
it matters which: a clamp or outlier check that misbehaves would let bad sensor
readings through.

---

## Not affected

`quake-net` passes in full (359 tests). All test files added on 2026-08-27
pass in every sub-project.

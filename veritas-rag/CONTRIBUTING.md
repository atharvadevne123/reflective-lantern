# Contributing to Veritas-Rag

1. `make install` then `make test` — all 108 tests must pass.
2. `make lint` must exit clean (`make format` autofixes).
3. `make eval` is the anti-hallucination gate: hallucination rate must stay
   0.0 and the adversarial pass rate 1.0. A change that trades correctness
   for fluency will be rejected here.
4. Every behavioral change needs a test; new retrieval or scoring logic
   needs an evaluation case.
5. Conventional commits: feat / fix / test / docs / chore / ci / refactor.

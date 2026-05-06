# Offline Wheels Status

The large offline wheel directories in this local copy were intentionally cleared to reduce disk usage during code analysis. SciWorld is the exception after the 2026-04-20 GRPO environment review.

Directories that may still be empty:
- wheels
- wheels_py312
- wheels_verl_py312
- wheels_babyai
- wheels_alfworld

Repopulated directories:
- wheels_sciworld: refilled on 2026-04-20, then regenerated on 2026-04-21 with Python 3.12 runtime wheels for `fastapi`, `uvicorn[standard]`, `scienceworld`, and transitive dependencies.

What remains:
- All requirements files in this directory.
- A status README inside each original wheel directory.

Interpretation guidance:
- Reviewers should not treat missing wheel artifacts in still-empty directories as evidence that the H200 setup path is incomplete.
- For SciWorld specifically, the local Python 3.12 wheel artifacts are now present under `wheels_sciworld`.

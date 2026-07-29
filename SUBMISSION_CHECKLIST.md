# TempoRun submission checklist

## Ready in this folder

- [x] Full retrieval source code
- [x] OmniShotCut wrapper needed to regenerate shot manifests
- [x] README method description and repository structure
- [x] Tested hardware, driver, CUDA, Python, and package versions
- [x] Direct-install instructions with `uv`
- [x] Automated Qwen model download with revisions, sizes, and SHA-256 hashes
- [x] Automated OmniShotCut model download with revision, sizes, and hashes
- [x] Input video, task, and shot-manifest schemas
- [x] No machine-specific data path in the submitted launcher
- [x] Script descriptions and required execution order
- [x] Whole-pipeline command from raw V3C clips
- [x] Official default parameter table
- [x] Output schema and local validation script
- [x] Seed and determinism notes
- [x] Known limitations and troubleshooting notes
- [x] Model files, caches, dataset, and generated artifacts excluded from Git

## Must be completed by the team before sending the link

- [x] Fill team name in `README.md`: `messiuuu`
- [ ] Fill contact name/email in `README.md`
- [x] Create the remote GitHub/GitLab/Bitbucket repository
- [x] Make the final Git commit or release tag: `v1.0.0`
- [x] Fill repository URL and official commit/tag in `README.md`
- [ ] Confirm every external model URL is reachable from a clean machine
- [ ] Run a clean-clone smoke test on the intended BTC hardware
- [ ] Run `scripts/validate_submission.py` on the reproduced final output
- [ ] If the repository is private, grant the organizers access

Do not upload the V3C dataset, competition task files, model weights, caches, or
generated embeddings unless the organizers explicitly request them.

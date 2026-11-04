# PhotonicsLab Wiki publishing and editorial contract

This repository's `docs/wiki/*.md` files are the reviewable source for the separate native GitHub Wiki. After merge and initializing an empty wiki if required, run `bash scripts/publish-github-wiki.sh --dry-run` to review; publish with `bash scripts/publish-github-wiki.sh` using configured secure Git credentials. The script preserves independent older pages and does not automatically delete stale material.

ArchWiki-inspired guides should contain scope, prerequisites, reproducible command or Python example, physical assumptions, numerical validity warnings, error diagnosis and source references. **Modern 0.6 versus 2021 legacy**, implemented versus proposed, and idealized versus experimentally calibrated are mandatory distinctions.

Guide/ACP indexing: include model name, version/commit, unit conventions, dependency, learning level, executable path, physical limits, validation status and canonical file. An agent must not infer secure QKD or experimentally confirmed entanglement from simulation of clicks, timing correlations or BB84 toy bits.
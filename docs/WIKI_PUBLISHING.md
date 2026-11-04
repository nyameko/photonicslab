# PhotonicsLab Wiki publishing and editorial contract

This repository's `docs/wiki/*.md` files are the reviewable source for the separate native GitHub Wiki. After merge and initializing an empty wiki if required, run `bash scripts/publish-github-wiki.sh --dry-run` to review; publish with `bash scripts/publish-github-wiki.sh` using configured secure Git credentials. The script preserves independent older pages and does not automatically delete stale material.

ArchWiki-inspired guides should contain scope, prerequisites, reproducible command or Python example, physical assumptions, numerical validity warnings, error diagnosis and source references. **Modern 0.6 versus 2021 legacy**, implemented versus proposed, and idealized versus experimentally calibrated are mandatory distinctions.

Guide/ACP indexing: include model name, version/commit, unit conventions, dependency, learning level, executable path, physical limits, validation status and canonical file. An agent must not infer secure QKD or experimentally confirmed entanglement from simulation of clicks, timing correlations or BB84 toy bits.

## Automatic Wiki publication from `dev`

After the reviewed `.github/workflows/publish-github-wiki.yml` PR is merged to `dev`, every pushed `docs/wiki/*.md` change on `dev` triggers validation of the Guide index and publication of managed Markdown files to this repository's native Wiki. The workflow uses built-in short-lived GitHub Actions `GITHUB_TOKEN` with `contents: write`, `gh auth setup-git` and a dry-run before publication, with no LLM provider API token or personal GitHub PAT in normal operation.

**First-time GitHub setup:** in repository Settings → General → Features enable Wikis; open Wiki, create and save an initial `Home` page. Until the first page exists, the Wiki Git remote does not exist. If GitHub Actions refuses a Wiki push, investigate org/repository token permissions first. The workflow also declares `workflow_dispatch`, but GitHub's UI may not show this until its file exists on the default branch; automatic `push: dev` works independently.

For manual publication, run `gh auth login`, `gh auth setup-git`, check out `dev`, then use `bash scripts/publish-github-wiki.sh --dry-run` followed by `bash scripts/publish-github-wiki.sh`. The publisher preserves unrelated native Wiki pages. Confirm Wiki Home/sidebar after each publication and distinguish alpha/development source from live production functionality. Revert inaccurate docs through another reviewed `dev` PR; use native Wiki Git history if direct Wiki content needs restoration.

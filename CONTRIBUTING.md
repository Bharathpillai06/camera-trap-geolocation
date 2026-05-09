# Contributing

Thanks for your interest in contributing! This project follows the
[Imageomics repository conventions](https://imageomics.github.io/Imageomics-Guide/github-guide/repo-guide/).

## Branches

- Primary branch: `main`. Do not commit directly to `main`.
- Use the pattern `category/issue-NN/short-description` for branches:
  - `category` ∈ {`feature`, `bugfix`, `experiment`}
  - `issue-NN` references the GitHub issue (use `no-ref` if there isn't one)
  - Example: `feature/issue-12/add-pyproj-fallback`

## Commits

Follow the [Conventional Commits](https://www.conventionalcommits.org/)
specification. Common types:

- `feat:` new functionality
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code change that neither fixes a bug nor adds a feature
- `test:` adding or updating tests
- `chore:` tooling, build, dependency bumps

Example: `feat: add pyproj-based Vincenty projection as alternative to spherical earth`

## Pull requests

- One PR per logical change. Don't bundle unrelated work.
- Reference the related issue in the PR description.
- All new code in `src/` should have a unit test in `tests/`.
- Update `HISTORY.md` with a one-line summary of user-facing changes.

## Code style

- Python 3.10+ type hints encouraged for new functions.
- Keep CLI scripts in `scripts/` thin — push reusable logic into
  `src/camera_trap_geolocation/`.

## Reporting issues

Please include:

- Python and OS version
- The full command line you ran
- A representative input filename or sample image (if license permits)
- The full error traceback

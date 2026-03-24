# Dependency & pip Hardening

## 1. Enforce pip inside virtualenvs (system-wide)

Configure pip globally so that it refuses to run outside an activated virtualenv:

```bash
sudo tee /etc/pip.conf >/dev/null <<'EOF'
[global]
require-virtualenv = true
EOF

sudo tee /etc/profile.d/pip-require-venv.sh >/dev/null <<'EOF'
export PIP_REQUIRE_VIRTUALENV=true
unset PIP_NO_REQUIRE_VIRTUALENV
EOF
```

Reload your shell (or log back in) and verify enforcement:

```bash
python3 -m pip install requests
# ^ fails until a virtualenv is active
```

## 2. Strict dependency management

Direct dependencies are pinned in `requirements.in`, and `requirements.lock` contains the fully resolved dependency graph with hashes (generated via `pip-compile --generate-hashes`). Never edit the lock file by hand.

Common workflow:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip pip-tools
# modify requirements.in if you need new/updated direct deps
pip-compile --generate-hashes -o requirements.lock requirements.in
```

After editing `requirements.in`, recompile the lock **and rebuild the wheel cache** so local installs never hit PyPI directly:

```bash
pip-compile --generate-hashes -o requirements.lock requirements.in
./scripts/build_wheelhouse.sh
```

To install for development/CI (wheel cache is mandatory):

```bash
source .venv/bin/activate
./scripts/check_deps.sh
python -m pip check
```

`scripts/check_deps.sh` automates the strict install + `pip check` combo, enforces an active virtualenv, and refuses to use the network—everything must come from `vendor/wheels`.

## 3. Local offline wheel cache

`./scripts/build_wheelhouse.sh` resolves every dependency in `requirements.lock`, downloads the exact hashed wheels, and drops them into `vendor/wheels`. This directory is ignored by git but should be refreshed (and re-distributed to CI agents) whenever the lock changes. Because installs now run with `--no-index --find-links vendor/wheels`, the build never talks to PyPI at install time.

## 4. Dependency auditing & SBOM

Run `./scripts/audit_deps.sh` after any dependency bump. It bootstraps a small tooling venv, executes `pip-audit -r requirements.lock`, and emits an up-to-date CycloneDX SBOM at `security/sbom.json`. Commit blockers: the audit must be clean, and the SBOM should be attached to releases or fed into scanners.

## 5. CI enforcement

Ensure CI always runs:

```bash
./scripts/check_deps.sh
./scripts/audit_deps.sh
```

Never invoke bare `pip install <pkg>` in CI or automation; rely on the lock file and hashes so supply-chain attacks are caught early.

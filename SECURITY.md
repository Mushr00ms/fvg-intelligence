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

To install for development/CI:

```bash
source .venv/bin/activate
python -m pip install --require-hashes --only-binary :all: -r requirements.lock
python -m pip check
```

`scripts/check_deps.sh` automates the strict install + `pip check` combo and will exit unless a virtualenv is active.

## 3. CI enforcement

Ensure CI always runs:

```bash
./scripts/check_deps.sh
```

Never invoke bare `pip install <pkg>` in CI or automation; rely on the lock file and hashes so supply-chain attacks are caught early.

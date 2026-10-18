.DEFAULT_GOAL := help

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
APP_ID := com.qmew.OrbitalEngineer
FPK_DIR := flatpak
MANIFEST := $(FPK_DIR)/$(APP_ID).yaml
DIST_DIR := $(FPK_DIR)/dist
VENDOR_DIR := $(FPK_DIR)/python3-deps
WHEELS := $(VENDOR_DIR)/wheels
# GNOME 49 uses CPython 3.13. Override both tags for another architecture/runtime.
PYVER_TAG ?= 313
PLAT_TAG ?= manylinux_2_27_x86_64 manylinux_2_28_x86_64 manylinux_2_17_x86_64 manylinux2014_x86_64
STATE_DIR ?= $(FPK_DIR)/.flatpak-builder
BUILD_DIR ?= $(FPK_DIR)/build
ARGS ?=

.PHONY: help all venv install dev pipx-install run server metrics test test-unit \
        wheel vendor-wheels flatpak-check flatpak-build flatpak-install \
        flatpak-run flatpak-server clean dev-clean distclean

help:
	@echo "Setup:     venv install (engine) dev (GTK UI + development tools) pipx-install"
	@echo "Run:       server run metrics (start server before run; ARGS=... supported)"
	@echo "Tests:     test (requires OpenCL) test-unit (CPU only)"
	@echo "Packaging: wheel vendor-wheels flatpak-check flatpak-build flatpak-install"
	@echo "Flatpak:   flatpak-server flatpak-run (start server first)"
	@echo "Cleanup:   clean dev-clean distclean"

all: wheel

venv: $(VENV)/bin/python

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(VENV_PYTHON) -m pip install -e .

dev: venv
	$(VENV_PYTHON) -m pip install -e '.[dev,gtk]'

pipx-install:
	pipx install '.[gtk]'

run: venv
	$(VENV_PYTHON) -m orbitalengineer $(ARGS)

server: venv
	$(VENV_PYTHON) -m orbitalengineer.ipc.server $(ARGS)

metrics: venv
	$(VENV_PYTHON) -m ui_metrics.metrics_app $(ARGS)

test: venv
	$(VENV_PYTHON) -m pytest $(ARGS)

test-unit: venv
	$(VENV_PYTHON) -m pytest src/orbitalengineer/twobody $(ARGS)

# Always rebuild so source and kernel changes cannot leave a stale wheel.
wheel: venv
	rm -rf build $(DIST_DIR)
	$(VENV_PYTHON) -m pip install 'build>=1.2'
	$(VENV_PYTHON) -m build --wheel --outdir $(DIST_DIR)

# Download into a fresh staging directory to avoid mixing incompatible wheels.
vendor-wheels: venv
	rm -rf $(WHEELS).tmp
	mkdir -p $(WHEELS).tmp
	$(VENV_PYTHON) -m pip download -r $(VENDOR_DIR)/requirements.txt \
		--dest $(WHEELS).tmp --only-binary=:all: \
		$(foreach platform,$(PLAT_TAG),--platform $(platform)) --implementation cp --python-version $(PYVER_TAG)
	rm -rf $(WHEELS)
	mv $(WHEELS).tmp $(WHEELS)

flatpak-check:
	appstreamcli validate --no-net --pedantic $(FPK_DIR)/data/$(APP_ID).metainfo.xml
	desktop-file-validate $(FPK_DIR)/data/$(APP_ID).desktop
	flatpak-builder --show-manifest $(MANIFEST) > /dev/null

flatpak-build: flatpak-check wheel vendor-wheels
	flatpak-builder --force-clean --state-dir=$(STATE_DIR) $(BUILD_DIR) $(MANIFEST)

flatpak-install: flatpak-build
	flatpak-builder --user --install --finish-only --state-dir=$(STATE_DIR) $(BUILD_DIR) $(MANIFEST)

flatpak-run:
	flatpak run $(APP_ID) $(ARGS)

flatpak-server:
	flatpak run --command=python3 $(APP_ID) -m orbitalengineer.ipc.server $(ARGS)

clean:
	rm -rf build $(DIST_DIR) src/*.egg-info

dev-clean: clean
	rm -rf $(VENV)

distclean: clean
	rm -rf $(WHEELS) $(WHEELS).tmp $(STATE_DIR) $(BUILD_DIR)


# ---------- config ----------
APP_ID     := com.qmew.OrbitalEngineer
FPK_DIR    := flatpak
MANIFEST   := $(FPK_DIR)/$(APP_ID).yaml

PYVER_TAG  := 312                    # GNOME 47 → CPython 3.12
PLAT_TAG   := manylinux_2_17_x86_64  # (= manylinux2014 x86_64)

VENV       := .venv
DIST_DIR   := $(FPK_DIR)/dist
VENDOR_DIR := $(FPK_DIR)/python3-deps
WHEELS     := $(VENDOR_DIR)/wheels
REQ        := $(VENDOR_DIR)/requirements.txt

STATE_DIR  := /tmp/fb-state
BUILD_DIR  := /tmp/build-dir

.PHONY: dev pipx-install all wheel $(WHEELS) vendor-wheels flatpak-check flatpak-build flatpak-run clean distclean

dev: $(VENV)/bin/activate
	$(VENV)/bin/pip install -e .[dev,gtk]

venv: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python -m venv $(VENV)
	$(VENV)/bin/pip install -U pip

pipx-install:
	pipx install .[gtk]

run: dev
	$(VENV)/bin/python 

dev-clean:
	rm -rf $(VENV) $(DIST_DIR)

wheel: $(DIST_DIR)/.built

$(DIST_DIR)/.built: pyproject.toml
	python3 -m pip install -U build
	python3 -m build --wheel -o $(DIST_DIR)
	touch $@

vendor-wheels: $(WHEELS)

$(WHEELS): 
	@echo ">> Downloading wheels -> $(WHEELS)"
	rm -rf $(WHEELS)
	mkdir -p $(WHEELS)
	python3 -m pip download --no-cache-dir -r $(REQ) -d $(WHEELS) \
	  --only-binary=:all: --prefer-binary \
	  --platform $(PLAT_TAG) --implementation cp --python-version $(PYVER_TAG)

flatpak-check:
	appstreamcli validate --pedantic $(FPK_DIR)/data/com.qmew.OrbitalEngineer.metainfo.xml

flatpak-build: wheel vendor-wheels
	flatpak-builder \
		--user \
		--install \
		--force-clean \
		--state-dir=$(STATE_DIR) \
		$(BUILD_DIR) $(MANIFEST)

flatpak-run:
	flatpak run $(APP_ID)

distclean: clean
 	rm -rf $(DIST_DIR) *.egg-info build

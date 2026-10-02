export PATH := $(HOME)/.local/bin:$(PATH)
export CARGO_TARGET_DIR := $(abspath target)

ENGINE_BUILD := work/engine-build
ENGINE_OUT   := target/engine
ENGINE_VERSION ?= 0.1.0

.PHONY: help setup engine app check-prereqs

help:
	@echo "  make setup           install JS deps and check the toolchain"
	@echo "  make engine          build the native Metal engine"
	@echo "  make app             build, bundle, and write dist/edge0-*-arm64.app"

check-prereqs:
	bash scripts/check-prereqs.sh

setup: check-prereqs
	cd app && yarn install --frozen-lockfile

engine:
	cmake -S crates/engine-native -B $(ENGINE_BUILD) -DCMAKE_BUILD_TYPE=Release -DE0N_VERSION=$(ENGINE_VERSION)
	cmake --build $(ENGINE_BUILD) --parallel 8
	@mkdir -p $(ENGINE_OUT)
	rm -f $(ENGINE_OUT)/libedge0_engine_native.dylib $(ENGINE_OUT)/libmlx.dylib
	cp -f $(ENGINE_BUILD)/out/libedge0_engine_native.dylib $(ENGINE_OUT)/
	cp -f $(ENGINE_BUILD)/out/libmlx.dylib $(ENGINE_OUT)/ 2>/dev/null || \
	  cp -f $(ENGINE_BUILD)/mlx/libmlx.dylib $(ENGINE_OUT)/
	cp -f $(ENGINE_BUILD)/out/mlx.metallib $(ENGINE_OUT)/ 2>/dev/null || \
	  cp -f $(ENGINE_BUILD)/mlx/mlx/backend/metal/kernels/mlx.metallib $(ENGINE_OUT)/
	codesign --force --sign - $(ENGINE_OUT)/libmlx.dylib $(ENGINE_OUT)/mlx.metallib \
	  $(ENGINE_OUT)/libedge0_engine_native.dylib
	@echo "engine: $(ENGINE_OUT)/libedge0_engine_native.dylib"

app: setup
	cargo build --release -p edge0-cli -p edge0d -p edge0-engine
	$(MAKE) ENGINE_VERSION=$$(awk '$$0 == "[workspace.package]" { in_pkg=1; next } /^\[/ { in_pkg=0 } in_pkg && $$1 == "version" { gsub(/"/, "", $$3); print $$3; exit }' Cargo.toml) engine
	bash scripts/package-macos.sh

#!/bin/sh
# Preserve missing-tool outcomes as CTest skips.
for tool in uv git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "not executed: $tool not found"
        exit 77
    fi
done
exec uv run --no-project python -I "$@"

"""``python -m edge0`` entry point (same surface as the ``edge0``
console script)."""

from edge0.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

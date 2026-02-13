#!/usr/bin/env python
from __future__ import annotations

import sys

from rna3d_local.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["research-sync-literature", *sys.argv[1:]]))

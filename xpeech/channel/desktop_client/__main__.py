from __future__ import annotations

import argparse

from .app import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Xpeech desktop client.")
    parser.add_argument("--dev", action="store_true", help="Use the Vite dev server for the frontend.")
    args = parser.parse_args()
    run(dev=args.dev)

if __name__ == "__main__":
    main()

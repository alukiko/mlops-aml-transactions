from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


CONVENTIONAL_COMMIT = re.compile(
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(\([a-z0-9._/-]+\))?(!)?: .+",
    re.IGNORECASE,
)


def commit_subjects(revision_range: str) -> list[str]:
    result = subprocess.run(
        ["git", "log", "--no-merges", "--format=%s", revision_range],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Conventional Commit subjects.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--range", dest="revision_range", help="Git revision range, for example base..head.")
    source.add_argument("--message-file", type=Path, help="Commit message file supplied by a commit-msg hook.")
    args = parser.parse_args()

    if args.message_file:
        lines = args.message_file.read_text(encoding="utf-8").splitlines()
        subjects = [lines[0].strip() if lines else ""]
    else:
        subjects = commit_subjects(args.revision_range)

    invalid = [subject for subject in subjects if not CONVENTIONAL_COMMIT.fullmatch(subject)]
    if invalid:
        print("Invalid Conventional Commit subject(s):", file=sys.stderr)
        for subject in invalid:
            print(f"  - {subject}", file=sys.stderr)
        print("Expected format: type(optional-scope): description", file=sys.stderr)
        return 1

    print(f"Validated {len(subjects)} Conventional Commit subject(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

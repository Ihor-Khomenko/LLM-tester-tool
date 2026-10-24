"""Shared range selection helpers (CLI + legacy interactive scripts)."""

from __future__ import annotations

from typing import Optional, Tuple


def parse_range_spec(spec: str, data_len: int) -> Optional[Tuple[int, int]]:
    """Parse a range string into half-open [start, end).

    Accepted forms (matching the original scripts):
      - ``all`` / ``0``  → entire dataset
      - ``N``            → first N items  [0, N)
      - ``start-end``    → [start, end)
    """
    user_input = (spec or "").strip().lower()
    if user_input in ("0", "all", "*"):
        return 0, data_len
    if "-" in user_input:
        try:
            start_str, end_str = user_input.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start < 0 or end <= start or end > data_len:
                return None
            return start, end
        except ValueError:
            return None
    try:
        n = int(user_input)
        if n <= 0 or n > data_len:
            return None
        return 0, n
    except ValueError:
        return None


def interactive_select_range(data_len: int, label: str = "sets") -> Tuple[int, int]:
    """Prompt the user for a range (legacy script behaviour). Exits on ``00``."""
    while True:
        user_input = input(
            f"How many {label} should be tested? "
            "(0 for all, 00 to exit, N for first N, start-end for a range): "
        ).strip()
        if user_input == "00":
            print("Exiting program.")
            raise SystemExit(0)
        parsed = parse_range_spec(user_input, data_len)
        if parsed is None:
            print(
                "Invalid input. Please enter 0, 00, a positive integer, "
                "or a valid range like 10-20."
            )
            continue
        start, end = parsed
        if start == 0 and end == data_len:
            print(f"Running ALL {end} {label}...")
        elif start == 0:
            print(f"Running first {end} {label}...")
        else:
            print(f"Running {label} {start} to {end - 1}...")
        return start, end

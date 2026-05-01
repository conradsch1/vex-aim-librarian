#!/usr/bin/env python3
"""
Presenter walkthrough for Celeste Librarian (robot + GPT).

Run without hardware to print the demo script::

    python3 celeste_librarian_demo.py

Print the GPT personality preamble::

    python3 celeste_librarian_demo.py --personality

With hardware: from the ``vex-aim-tools`` repo, start ``simple_cli`` with both repos on
``PYTHONPATH`` (see README), then::

    runfsm('CelesteLibrarian')
    start

Use the numbered prompts in the chat / speech UI in order (or skip steps if time is short).
"""

from __future__ import annotations

import argparse
import os
import re
import sys


def _repo_roots() -> tuple[str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    librarian = here
    cogrob = os.path.dirname(librarian)
    tools = os.path.join(cogrob, "vex-aim-tools")
    return tools, librarian


def _personality_from_source() -> tuple[str, str]:
    """Parse ``CelesteLibrarian.py`` so ``--personality`` works without VEX ``vex``."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CelesteLibrarian.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    ver_m = re.search(r'CELESTE_LIBRARIAN_VERSION\s*=\s*"([^"]+)"', src)
    version = ver_m.group(1) if ver_m else "?"
    marker = '_LIBRARIAN_PERSONALITY = """'
    i0 = src.index(marker) + len(marker)
    i1 = src.index('"""\n\n# Public alias', i0)
    return version, src[i0:i1].strip()


def _print_personality() -> None:
    try:
        from CelesteLibrarian import (
            CELESTE_LIBRARIAN_PERSONALITY,
            CELESTE_LIBRARIAN_VERSION,
        )

        version, body = CELESTE_LIBRARIAN_VERSION, CELESTE_LIBRARIAN_PERSONALITY.strip()
    except ModuleNotFoundError:
        version, body = _personality_from_source()

    print(f"Celeste Librarian GPT personality (v{version})\n")
    print(body)


def _print_walkthrough() -> None:
    tools, librarian = _repo_roots()
    parts: list[str] = []
    for p in (tools, librarian):
        if p not in parts:
            parts.append(p)
    if os.environ.get("PYTHONPATH"):
        for p in os.environ["PYTHONPATH"].split(os.pathsep):
            if p and p not in parts:
                parts.append(p)
    pypath = os.pathsep.join(parts)

    print(
        """
================================================================================
 Celeste Librarian — feature showcase (for a live audience)
================================================================================
""".rstrip()
    )

    print(
        f"""
1) Environment
   export PYTHONPATH="{pypath}"

2) Launch CLI (from anywhere)
   python3 {os.path.join(tools, "simple_cli")}

3) Load the FSM (in the CLI)
   runfsm('CelesteLibrarian')
   start

4) Narrate for the room: Celeste is still the stock VEX AIM robot in the GPT preamble,
   plus a "stacks duty" librarian voice (warm, brief, kid-safe) layered on top.

5) Demo script — type or speak these to GPT (one idea per beat; adjust book ids to your shelf):

   A. Personality / small talk
      "Hi! Who are you and what is your job here?"
      → Should answer as Celeste + mobile librarian, concise.

   B. Stock Celeste body (still available)
      "#forward 30"   (or ask in natural language: "roll forward thirty millimeters")
      "#turn 45"
      "#glow 0 80 120"

   C. Homing corner (marker 16 must be visible in camera when you start)
      "Please go dock in the homing corner."
      → Must emit a lone line: #gohome

   D. Fetch a book (spine ArUco ids per field; slots 1..N map to lowest id upward)
      "Bring me book from slot 2"   or   "Please fetch book 10."
      → Must emit exactly: #getbook <id or slot>
      Watch: pilot → engage → attach → back off → 180° presentation turn → patron point
      → kick release → navigate home → localize scan → face marker 20.

   E. Return / shelve (book on patron staging pad, target empty slot on return row)
      "The book is on the pad—please shelve it in the leftmost return slot."
      → Must emit exactly: #returnbook <slot or spine id 9–12>

   F. Re-localize without full homing
      "Run localization again."
      → #localize

   G. Map + vision (already on in default CelesteLibrarian __init__)
      Point out world map viewer, path viewer, particle viewer, camera window.

6) Optional parallel demos (separate FSMs, not CelesteLibrarian)
   runfsm('BooksIdleDemo')     — idle vision → BookObj on map
   runfsm('SwapBooksDemo')     — scripted physical swap

================================================================================
""".strip()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Celeste Librarian showcase helper")
    parser.add_argument(
        "--personality",
        action="store_true",
        help="Print the GPT librarian personality preamble and exit",
    )
    args = parser.parse_args(argv)

    tools, librarian = _repo_roots()
    if librarian not in sys.path:
        sys.path.insert(0, librarian)
    if tools not in sys.path:
        sys.path.insert(0, tools)

    if args.personality:
        _print_personality()
        return 0

    _print_walkthrough()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

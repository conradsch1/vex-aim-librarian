# VEX AIM Librarian

## Running with vex-aim-tools (stock)

Add both repos to `PYTHONPATH` (parent folder or each repo path). Programs should call `install_librarian_extensions(robot)` once per run after `StateMachineProgram.__init__` and before `start()` (see `navigate_to_marker/NavigateToMarker.py`). Book logic and 3D QML live under `aim_librarian/`; **do not** fork book changes into `vex-aim-tools`.

`export PYTHONPATH="/Users/tsumacpro/CogRob/vex-aim-tools:/Users/tsumacpro/CogRob/vex-aim-librarian${PYTHONPATH:+:$PYTHONPATH}"`

**Quick demo:** from `simple_cli`, run `runfsm('BooksIdleDemo')` to open the map + camera and ingest spine markers (ArUco ids 9–15) as books while idle.

## Features (depth-first)

- Scan book spines
- Move a book to a target place
- Swap two books
- Sort books
- **GPT-4o** — human-facing interface

## Action items

- [ ] Decide on book representation (appearance and ID mechanism)
  - [ ] Verify book movement (magnets)
  - [ ] Verify book identification
  - [ ] Build books
- [ ] Set up world
  - [ ] Test navigation precision with different environment approaches (decide if ArUco is needed)
  - [ ] Decide on “stack” representation
- [ ] Algorithm: move book to place
- [ ] Algorithm: swap two books
- [ ] Algorithm: sort stack of books
- [ ] GPT-4o integration — scope and UX (replace placeholder notes)


Robot AIM-2A81818 has a well-positioned magnet.
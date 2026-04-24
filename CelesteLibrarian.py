"""
Celeste personality + librarian book delivery.

Extends stock :class:`Celeste` with the same GPT preamble and command dispatch, plus
``#getbook N`` to fetch a book by spine ArUco id (or 1-based shelf index), and
``#returnbook S`` to take a book from the patron staging area and shelve it into an
empty slot ``S`` (slot index 1–4 left-to-right, or spine id 9–12). Requires
``vex-aim-tools`` and ``vex-aim-librarian`` on ``PYTHONPATH``.

From ``simple_cli``::

    runfsm('CelesteLibrarian')

Sequence for ``#getbook``: capture current pose → pilot to book → settle → optional
engage forward → attach → small shelf-clearance back-off → pilot back to the captured
pose → turn 180° to present to the patron → wait until spine marker is no longer seen
→ turn back.

Sequence for ``#returnbook``: verify the robot is not already holding a book and the
book-10 shelve cell is vacant → turn 90° right (robot convention −90°) → pilot to the
staged book (spine id 10) → pick it up → turn toward and pilot to the fixed book-10
shelve pose (``RETURN_SHELF_BOOK10_*_MM``) → release there on the map.

Tune poses and distances for your field (class attributes on :class:`CelesteLibrarian`).

Books are not inserted into the world map at startup; :class:`BookObj` instances are created
from the camera when spine ArUco ids are seen (librarian world map).
"""

from __future__ import annotations

from math import pi

from aim_fsm import *
from aim_fsm.aim_kin import AIMKinematics
from aim_fsm.particle import ArucoCombinedSensorModel
from aim_fsm.pilot import PilotToPose

from aim_librarian import (
    BOOK_FIRST_ID,
    BOOK_LAST_ID,
    PilotToBook,
    install_librarian_extensions,
    is_book_aruco_id,
)
from aim_librarian.worldmap_ext import ensure_bookobj_from_vision
from aim_librarian.book_manip import AttachBook, DetachBookAtPose, WaitUntilBookRemoved
from aim_librarian.books import BookObj
from aim_librarian.pilot_ext import TurnTowardPose

from Celeste import CELESTE_VERSION, Celeste, new_preamble

CELESTE_LIBRARIAN_VERSION = "1.1"

_RETURN_SHELF_FIRST = 9
_RETURN_SHELF_LAST = 12
_RETURN_SHELF_NSLOTS = _RETURN_SHELF_LAST - _RETURN_SHELF_FIRST + 1

_LIBRARIAN_PREAMBLE_EXTENSION = f"""
  # LIBRARIAN SECTION.
  You also help patrons in a small library with magnet-held books on a shelf.
  Books are world objects whose names look like ``Book-N.a``, with spine ArUco ids
  in the range {BOOK_FIRST_ID}..{BOOK_LAST_ID}.

  IMPORTANT BOOK-HANDLING RULE (overrides the body-control section):
  Whenever the user refers to a book — by number, by spine id, by shelf slot,
  or by saying things like "get book 9", "go get book 9", "fetch book 9",
  "bring me book 9", "grab book 9", "take book 9", "retrieve book 9",
  "pick up book 9", "go to book 9" — you MUST output exactly one line:

      #getbook N

  where N is either (a) the spine ArUco id from {BOOK_FIRST_ID} through
  {BOOK_LAST_ID}, or (b) a shelf slot index 1..{BOOK_LAST_ID - BOOK_FIRST_ID + 1}
  counting from the lowest slot (slot 1 → id {BOOK_FIRST_ID}).

  NEVER use ``#pilottoobject``, ``#pickup``, ``#turntoward``, or any other
  body-control command for a book; ``#getbook`` is the only correct command for
  any object whose name starts with ``Book-``. Output nothing else on the line.
  After the trip completes you may speak briefly.

  RETURN-BOOK RULE (also overrides body-control for shelving):
  When the user hands back a book at the staging pad, or asks to shelve, return,
  "put it away", "put this in slot", etc., you MUST output exactly one line:

      #returnbook S

  where S is either (a) a shelf slot index 1..{_RETURN_SHELF_NSLOTS} counting **from the left**
  (slot 1 is the leftmost slot, corresponding to spine id {_RETURN_SHELF_FIRST}), or (b) a spine id
  {_RETURN_SHELF_FIRST}..{_RETURN_SHELF_LAST} naming which **empty** slot to fill.

  NEVER use ``#pilottoobject``, ``#pickup``, ``#drop``, or other body-control commands for this;
  ``#returnbook`` is the only correct command for shelving a patron-returned book.
  Output nothing else on the line. After the trip completes you may speak briefly.
"""

class CelesteLibrarian(Celeste):
    """Celeste + librarian field: ``#getbook`` and ``#returnbook`` macros."""

    # Field layout (mm, radians) — match ``world_setup/WorldSetup.fsm`` / ``SwapBooksDemo``.
    _TAG_Y = 138
    _TAG_X = 175
    BOX_WIDTH = 35

    ROBOT_STANDOFF_MM = 115.0
    # Two-stage book approach.
    #   Stage 1 (path planner): stop with the robot's MAGNET face roughly
    #       ``ENGAGE_MM`` short of the book's spine surface, perpendicular to
    #       the shelf. Splitting the approach this way keeps the path planner
    #       from cutting close to neighboring books.
    #   Stage 2: drive straight forward by ``ENGAGE_MM`` so the magnet
    #       face just touches the spine.
    #
    # ``BOOK_APPROACH_OFFSET_MM`` is the shift applied to the *robot center*'s
    # goal pose, measured from the spine *marker centroid* back along the line
    # robot->book. It must include:
    #   - body radius                : robot center -> magnet face
    #   - half spine thickness       : marker centroid -> spine front surface
    #   - the desired pre-engage gap : ``ENGAGE_MM``
    # otherwise the engage step drives the body interior past the spine and
    # the robot rams the book.
    ENGAGE_MM = 50.0
    BOOK_APPROACH_OFFSET_MM = (
        AIMKinematics.body_diameter / 2     # 28.5 mm: center -> magnet face
        + BookObj.SPINE_THICKNESS_MM / 2    # 12.7 mm: marker -> spine surface
        + ENGAGE_MM                         # 50.0 mm: pre-engage standoff gap
    )
    # Short straight-line back-off after attach so PilotToStart isn't planning
    # from inside the shelf/book footprint. The full trip back to the original
    # pose is handled by the pilot-back step.
    SHELF_CLEARANCE_MM = 80.0
    SETTLE_S = 0.4
    PRESENT_TURN_DEG = 180.0
    # Degrees per second for the 180° present / un-present turns. Slow enough to
    # look deliberate when handing the book to a patron, and slow enough that
    # the IMU's gyro-threshold check doesn't false-trigger a "LOST" pose.
    PRESENT_TURN_SPEED_DPS = 45.0

    # Patron return / ``#returnbook`` — layout numbers are **centimeters**, converted to mm.
    _RETURN_CM_TO_MM = 10.0
    RETURN_STAGING_X_MM = 0.0
    RETURN_STAGING_Y_MM = -14.0 * _RETURN_CM_TO_MM
    RETURN_SHELF_MARKER_IDS = (9, 10, 11, 12)
    RETURN_SLOT_Y_MM = {
        9: 6.75 * _RETURN_CM_TO_MM,
        10: 2.25 * _RETURN_CM_TO_MM,
        11: -2.25 * _RETURN_CM_TO_MM,
        12: -6.75 * _RETURN_CM_TO_MM,
    }
    RETURN_STAGING_DETECT_RADIUS_MM = 160.0
    # Patron-return flow: after vacancy checks, turn toward staging then pilot to this spine id.
    RETURN_STAGING_SPINE_ID = 10
    # Staging pickup only: drive a bit farther forward than ``ENGAGE_MM`` so the magnet meets the spine.
    RETURN_STAGING_ENGAGE_EXTRA_MM = 12.0
    RETURN_STAGING_ENGAGE_MM = ENGAGE_MM + RETURN_STAGING_ENGAGE_EXTRA_MM
    # After pickup, shelve at this world pose (cm → mm). Book 10 column: (+2.25 cm, +18 cm) x, y.
    RETURN_SHELF_BOOK10_X_MM = -2.25 * _RETURN_CM_TO_MM
    RETURN_SHELF_BOOK10_Y_MM = -18.0 * _RETURN_CM_TO_MM
    RETURN_SLOT_OCCUPANCY_TOL_MM = 28.0
    RETURN_SHELF_ROW_X_TOL_MM = 55.0
    RETURN_NAV_RETREAT_MM = 220.0
    RETURN_RELEASE_BACK_MM = -38.0
    RETURN_POST_DROP_CLEAR_MM = -60.0
    # ``PilotToPose`` always runs the RRT (see ``aim_fsm.pilot.PilotToPose``). Staging at
    # y≈-14 cm lines up with corner ``ArucoMarker-18`` at ``(-_TAG_Y)``; a shallow +X
    # standoff collides that goal with marker 18's inflated obstacle — add extra +X.
    RETURN_STAGING_EXTRA_X_MM = 100.0

    @classmethod
    def shelf_x_mm(cls) -> float:
        return cls._TAG_X - 45.0

    @classmethod
    def default_home_pose(cls) -> Pose:
        """Standoff in front of the shelf row, facing the shelf (heading 0)."""
        return Pose(cls.shelf_x_mm() - 185.0, 0.0, 0.0, 0.0)

    def __init__(self, **kwargs):
        landmarks = {
            "ArucoMarker-17": Pose(self._TAG_X - self.BOX_WIDTH / 2, self._TAG_Y, 5, pi / 2),
            "ArucoMarker-18": Pose(self._TAG_X - self.BOX_WIDTH / 2, -self._TAG_Y, 5, 3 * pi / 2),
            "ArucoMarker-19": Pose(self._TAG_X, -self._TAG_Y + self.BOX_WIDTH / 2, 5, pi),
            "ArucoMarker-20": Pose(self._TAG_X, self._TAG_Y - self.BOX_WIDTH / 2, 5, pi),
        }
        pf = ParticleFilter(
            robot,
            num_particles=500,
            landmarks=landmarks,
            sensor_model=ArucoCombinedSensorModel,
        )
        opts = dict(
            particle_filter=pf,
            wall_marker_dict=None,
            speech=True,
            launch_particle_viewer=True,
            launch_path_viewer=True,
            launch_worldmap_viewer=True,
            launch_cam_viewer=True,
        )
        opts.update(kwargs)
        super().__init__(**opts)
        install_librarian_extensions(self.robot)

    def seed_librarian_world(self) -> None:
        """Fixed corner ArUco landmarks only; :class:`BookObj` entries come from vision."""
        m17 = ArucoMarkerObj(
            {"name": "ArucoMarker-17", "id": 17, "marker": None},
            x=self._TAG_X - self.BOX_WIDTH / 2,
            y=self._TAG_Y,
            theta=pi / 2,
        )
        m18 = ArucoMarkerObj(
            {"name": "ArucoMarker-18", "id": 18, "marker": None},
            x=self._TAG_X - self.BOX_WIDTH / 2,
            y=-self._TAG_Y,
            theta=3 * pi / 2,
        )
        m17.is_fixed = True
        m18.is_fixed = True
        self.robot.world_map.objects["ArucoMarker-17.a"] = m17
        self.robot.world_map.objects["ArucoMarker-18.a"] = m18
        m19 = ArucoMarkerObj(
            {"name": "ArucoMarker-19", "id": 19, "marker": None},
            x=self._TAG_X,
            y=-self._TAG_Y + self.BOX_WIDTH / 2,
            theta=pi,
        )
        m20 = ArucoMarkerObj(
            {"name": "ArucoMarker-20", "id": 20, "marker": None},
            x=self._TAG_X,
            y=self._TAG_Y - self.BOX_WIDTH / 2,
            theta=pi,
        )
        m19.is_fixed = True
        m20.is_fixed = True
        self.robot.world_map.objects["ArucoMarker-19.a"] = m19
        self.robot.world_map.objects["ArucoMarker-20.a"] = m20

    def start(self):
        self.seed_librarian_world()
        self.robot.openai_client.set_preamble(new_preamble + _LIBRARIAN_PREAMBLE_EXTENSION)
        self.picked_up_handler = self.picked_up_celeste
        self.put_down_handler = self.put_down_celeste
        StateMachineProgram.start(self)

    class CmdGetBook(StateNode):
        """Pilot to book → attach → return to original pose → present → wait for removal → turn back."""

        marker_id: int = BOOK_FIRST_ID

        def start(self, event=None):
            if self.running:
                return
            self._parse_ok = self._parse_getbook_event(event)
            # Capture the robot's pose at command start so we can return to it
            # after attaching the book ("retreat to its original position").
            rp = self.robot.pose
            self.start_pose = Pose(rp.x, rp.y, rp.z, rp.theta)
            print(
                f"CmdGetBook: start pose = ({rp.x:.1f}, {rp.y:.1f}) "
                f"@ {rp.theta * 180 / pi:.1f} deg"
            )
            super().start(event)

        def _parse_getbook_event(self, event) -> bool:
            # The default StateNode.start does not pass the triggering event to
            # start_node, so the parse must happen here where event.data is live.
            raw = getattr(event, "data", "") if event is not None else ""
            if not isinstance(raw, str):
                print(f"CmdGetBook: non-string event data: {raw!r}")
                return False
            parts = raw.strip().split()
            if len(parts) < 2:
                print(f"CmdGetBook: missing book id in {raw!r}")
                return False
            try:
                n = int(parts[1])
            except ValueError:
                print(f"CmdGetBook: book id is not an integer in {raw!r}")
                return False
            if not is_book_aruco_id(n):
                nslots = BOOK_LAST_ID - BOOK_FIRST_ID + 1
                if 1 <= n <= nslots:
                    n = BOOK_FIRST_ID + (n - 1)
                else:
                    print(
                        f"CmdGetBook: id {n} is not a valid spine id "
                        f"({BOOK_FIRST_ID}-{BOOK_LAST_ID}) or slot (1-{nslots})"
                    )
                    return False
            self.marker_id = n
            print(f"CmdGetBook: fetching Book-{n}")
            return True

        class ParseGetBookId(StateNode):
            def start(self, event=None):
                super().start(event)
                if getattr(self.parent, "_parse_ok", False):
                    self.post_completion()
                else:
                    self.post_failure()

        class PilotToParsedBook(PilotToBook):
            def __init__(self, **kw):
                # Heading is fixed by ``PilotToBook._refine_target_pose`` to
                # point from the standoff back toward the spine centroid, so
                # the subsequent ``Forward(ENGAGE_MM)`` engage step drives
                # straight INTO the book (not sideways across neighboring
                # spines, and not away from the book as align_heading=True
                # would have done).
                super().__init__(BOOK_FIRST_ID, **kw)

            def start(self, event=None):
                self.marker_id = self.parent.marker_id
                super().start(event)

        class AttachParsedBook(AttachBook):
            def __init__(self):
                super().__init__(BOOK_FIRST_ID)

            def start(self, event=None):
                self.marker_id = self.parent.marker_id
                super().start(event)

        class PilotToStart(PilotToPose):
            """Pilot back to the pose captured by ``CmdGetBook.start``."""

            def __init__(self, **kw):
                super().__init__(target_pose=None, **kw)

            def start(self, event=None):
                self.target_pose = self.parent.start_pose
                super().start(event)

        def setup(self):
            parse = self.ParseGetBookId() .set_name("gb_parse") .set_parent(self)
            pilot = self.PilotToParsedBook(
                book_approach_offset_mm=CelesteLibrarian.BOOK_APPROACH_OFFSET_MM
            ) .set_name("gb_pilot") .set_parent(self)
            settle = StateNode() .set_name("gb_settle") .set_parent(self)
            engage = Forward(CelesteLibrarian.ENGAGE_MM) .set_name("gb_engage") .set_parent(self)
            attach = self.AttachParsedBook() .set_name("gb_attach") .set_parent(self)
            # Small mechanical clearance from the shelf so path planning from the
            # next state isn't trying to start inside the book/shelf footprint.
            back_off = Forward(-CelesteLibrarian.SHELF_CLEARANCE_MM) .set_name(
                "gb_back_off"
            ) .set_parent(self)
            pilot_back = self.PilotToStart() .set_name("gb_pilot_back") .set_parent(self)
            turn_present = Turn(
                CelesteLibrarian.PRESENT_TURN_DEG,
                turn_speed=CelesteLibrarian.PRESENT_TURN_SPEED_DPS,
            ) .set_name("gb_turn_present") .set_parent(self)
            wait_rm = WaitUntilBookRemoved() .set_name("gb_wait_rm") .set_parent(self)
            turn_back = Turn(
                -CelesteLibrarian.PRESENT_TURN_DEG,
                turn_speed=CelesteLibrarian.PRESENT_TURN_SPEED_DPS,
            ) .set_name("gb_turn_back") .set_parent(self)
            done = ParentCompletes() .set_name("gb_done") .set_parent(self)
            fail = Say("Sorry, I couldn't bring that book. Let's try again.") .set_name(
                "gb_fail"
            ) .set_parent(self)
            fail_done = ParentCompletes() .set_name("gb_fail_done") .set_parent(self)

            CompletionTrans() .add_sources(parse) .add_destinations(pilot)
            FailureTrans() .add_sources(parse) .add_destinations(fail)

            CompletionTrans() .add_sources(pilot) .add_destinations(settle)
            FailureTrans() .add_sources(pilot) .add_destinations(fail)

            TimerTrans(CelesteLibrarian.SETTLE_S) .add_sources(settle) .add_destinations(engage)

            CompletionTrans() .add_sources(engage) .add_destinations(attach)
            FailureTrans() .add_sources(engage) .add_destinations(fail)

            CompletionTrans() .add_sources(attach) .add_destinations(back_off)
            FailureTrans() .add_sources(attach) .add_destinations(fail)

            CompletionTrans() .add_sources(back_off) .add_destinations(pilot_back)
            FailureTrans() .add_sources(back_off) .add_destinations(fail)

            CompletionTrans() .add_sources(pilot_back) .add_destinations(turn_present)
            PilotTrans(GoalUnreachable) .add_sources(pilot_back) .add_destinations(fail)
            FailureTrans() .add_sources(pilot_back) .add_destinations(fail)

            CompletionTrans() .add_sources(turn_present) .add_destinations(wait_rm)
            FailureTrans() .add_sources(turn_present) .add_destinations(fail)

            CompletionTrans() .add_sources(wait_rm) .add_destinations(turn_back)
            FailureTrans() .add_sources(wait_rm) .add_destinations(fail)

            CompletionTrans() .add_sources(turn_back) .add_destinations(done)
            FailureTrans() .add_sources(turn_back) .add_destinations(fail)

            CompletionTrans() .add_sources(fail) .add_destinations(fail_done)

            return self

    class CmdReturnBook(StateNode):
        """Pilot to patron staging, confirm a spine 9-12 there, pick up, shelve in an empty slot."""

        target_slot_id: int = _RETURN_SHELF_FIRST
        detected_marker_id: int = _RETURN_SHELF_FIRST

        def start(self, event=None):
            if self.running:
                return
            self._parse_ok = self._parse_returnbook_event(event)
            if self._parse_ok:
                self.detected_marker_id = CelesteLibrarian.RETURN_STAGING_SPINE_ID
            super().start(event)

        def _parse_returnbook_event(self, event) -> bool:
            raw = getattr(event, "data", "") if event is not None else ""
            if not isinstance(raw, str):
                print(f"CmdReturnBook: non-string event data: {raw!r}")
                return False
            parts = raw.strip().split()
            if len(parts) < 2:
                print(f"CmdReturnBook: missing slot in {raw!r}")
                return False
            try:
                n = int(parts[1])
            except ValueError:
                print(f"CmdReturnBook: slot is not an integer in {raw!r}")
                return False
            ids = CelesteLibrarian.RETURN_SHELF_MARKER_IDS
            nslots = len(ids)
            if 1 <= n <= nslots:
                n = ids[0] + (n - 1)
            if n not in ids:
                print(
                    f"CmdReturnBook: {n} is not spine id {_RETURN_SHELF_FIRST}-{_RETURN_SHELF_LAST} "
                    f"or slot index 1-{nslots}"
                )
                return False
            self.target_slot_id = n
            print(f"CmdReturnBook: shelve into column / spine id {n}")
            return True

        class ParseReturnSlot(StateNode):
            def start(self, event=None):
                super().start(event)
                if getattr(self.parent, "_parse_ok", False):
                    self.post_completion()
                else:
                    self.post_failure()

        class CheckNotHolding(StateNode):
            def start(self, event=None):
                super().start(event)
                if self.robot.holding is not None:
                    print("CmdReturnBook: robot is already holding an object")
                    self.post_failure()
                    return
                self.post_completion()

        class CheckTargetSlotVacant(StateNode):
            def start(self, event=None):
                super().start(event)
                cl = CelesteLibrarian
                sx = cl.RETURN_SHELF_BOOK10_X_MM
                sy = cl.RETURN_SHELF_BOOK10_Y_MM
                tol = cl.RETURN_SLOT_OCCUPANCY_TOL_MM
                xtol = cl.RETURN_SHELF_ROW_X_TOL_MM
                for obj in self.robot.world_map.objects.values():
                    if not isinstance(obj, BookObj):
                        continue
                    if obj.held_by is not None:
                        continue
                    if abs(obj.pose.x - sx) > xtol:
                        continue
                    if abs(obj.pose.y - sy) <= tol:
                        print(
                            "CmdReturnBook: book-10 shelve slot looks occupied "
                            f"(BookObj marker {obj.marker_id} near ({sx:.1f}, {sy:.1f}) mm)"
                        )
                        self.post_failure()
                        return
                self.post_completion()

        class TurnTowardStagedBook(Turn):
            """Face the staged book: 90° right in robot convention (-90°)."""

            def __init__(self):
                super().__init__(
                    -90.0,
                    turn_speed=CelesteLibrarian.PRESENT_TURN_SPEED_DPS,
                )

        class PilotToStagedBook(PilotToBook):
            def __init__(self, **kw):
                super().__init__(BOOK_FIRST_ID, **kw)

            def start(self, event=None):
                self.marker_id = CelesteLibrarian.RETURN_STAGING_SPINE_ID
                super().start(event)

        class AttachDetectedBook(AttachBook):
            def __init__(self):
                super().__init__(BOOK_FIRST_ID)

            def start(self, event=None):
                self.marker_id = self.parent.detected_marker_id
                # Map often has no BookObj until pending vision frames complete; attach only
                # consults ``world_map.objects``. Seed from the live snapshot if needed.
                ensure_bookobj_from_vision(self.robot, self.marker_id)
                super().start(event)

        class TurnTowardTargetSlot(TurnTowardPose):
            def __init__(self):
                super().__init__(target_pose=Pose(0.0, 0.0, 0.0, 0.0))

            def start(self, event=None):
                cl = CelesteLibrarian
                sx = cl.RETURN_SHELF_BOOK10_X_MM
                sy = cl.RETURN_SHELF_BOOK10_Y_MM
                self.target_pose = Pose(180.0, -25.0, 0.0, 0.0)
                super().start(event)

        class PilotToSlotStandoff(PilotToPose):
            def __init__(self, **kw):
                super().__init__(target_pose=None, **kw)

            def start(self, event=None):
                cl = CelesteLibrarian
                sy = cl.RETURN_SHELF_BOOK10_Y_MM
                sx = cl.RETURN_SHELF_BOOK10_X_MM
                # self.target_pose = Pose(sx, sy, 0.0, 0.0)
                self.target_pose = Pose(180.0, -25.0, 0.0, 0.0)
                super().start(event)

        class DetachAtTargetSlot(DetachBookAtPose):
            def __init__(self):
                super().__init__(Pose(0.0, 0.0, BookObj.HEIGHT_MM / 2, pi))

            def start(self, event=None):
                cl = CelesteLibrarian
                self.pose = Pose(
                    cl.RETURN_SHELF_BOOK10_X_MM,
                    cl.RETURN_SHELF_BOOK10_Y_MM,
                    BookObj.HEIGHT_MM / 2,
                    pi,
                )
                super().start(event)

        def setup(self):
            parse = self.ParseReturnSlot() .set_name("rb_parse") .set_parent(self)
            not_hold = self.CheckNotHolding() .set_name("rb_not_holding") .set_parent(self)
            vacant = self.CheckTargetSlotVacant() .set_name("rb_slot_vacant") .set_parent(self)
            turn_staged = self.TurnTowardStagedBook() .set_name("rb_turn_staged") .set_parent(self)
            pilot_book = self.PilotToStagedBook(
                book_approach_offset_mm=CelesteLibrarian.BOOK_APPROACH_OFFSET_MM
            ) .set_name("rb_pilot_book") .set_parent(self)
            settle_b = StateNode() .set_name("rb_settle_b") .set_parent(self)
            engage = Forward(CelesteLibrarian.RETURN_STAGING_ENGAGE_MM) .set_name("rb_engage") .set_parent(self)
            attach = self.AttachDetectedBook() .set_name("rb_attach") .set_parent(self)
            retreat_stg = Forward(-CelesteLibrarian.RETURN_NAV_RETREAT_MM) .set_name(
                "rb_retreat_staging"
            ) .set_parent(self)
            turn_slot = self.TurnTowardTargetSlot() .set_name("rb_turn_slot") .set_parent(self)
            pilot_slot = self.PilotToSlotStandoff() .set_name("rb_pilot_slot") .set_parent(self)
            release_fwd = Forward(CelesteLibrarian.RETURN_RELEASE_BACK_MM) .set_name(
                "rb_release_fwd"
            ) .set_parent(self)
            detach = self.DetachAtTargetSlot() .set_name("rb_detach") .set_parent(self)
            kick = Kick() .set_name("rb_kick") .set_parent(self)
            post_clear = Forward(CelesteLibrarian.RETURN_POST_DROP_CLEAR_MM) .set_name(
                "rb_post_clear"
            ) .set_parent(self)
            done = ParentCompletes() .set_name("rb_done") .set_parent(self)
            fail = Say(
                "Sorry, I couldn't put that book away. Let's check the staging area and the shelf."
            ) .set_name("rb_fail") .set_parent(self)
            fail_done = ParentCompletes() .set_name("rb_fail_done") .set_parent(self)

            CompletionTrans() .add_sources(parse) .add_destinations(not_hold)
            FailureTrans() .add_sources(parse) .add_destinations(fail)

            CompletionTrans() .add_sources(not_hold) .add_destinations(vacant)
            FailureTrans() .add_sources(not_hold) .add_destinations(fail)

            CompletionTrans() .add_sources(vacant) .add_destinations(turn_staged)
            FailureTrans() .add_sources(vacant) .add_destinations(fail)

            CompletionTrans() .add_sources(turn_staged) .add_destinations(pilot_book)
            FailureTrans() .add_sources(turn_staged) .add_destinations(fail)

            CompletionTrans() .add_sources(pilot_book) .add_destinations(settle_b)
            FailureTrans() .add_sources(pilot_book) .add_destinations(fail)
            PilotTrans(GoalUnreachable) .add_sources(pilot_book) .add_destinations(fail)

            TimerTrans(CelesteLibrarian.SETTLE_S) .add_sources(settle_b) .add_destinations(engage)

            CompletionTrans() .add_sources(engage) .add_destinations(attach)
            FailureTrans() .add_sources(engage) .add_destinations(fail)

            CompletionTrans() .add_sources(attach) .add_destinations(retreat_stg)
            FailureTrans() .add_sources(attach) .add_destinations(fail)

            CompletionTrans() .add_sources(retreat_stg) .add_destinations(turn_slot)
            FailureTrans() .add_sources(retreat_stg) .add_destinations(fail)

            CompletionTrans() .add_sources(turn_slot) .add_destinations(pilot_slot)
            FailureTrans() .add_sources(turn_slot) .add_destinations(fail)

            CompletionTrans() .add_sources(pilot_slot) .add_destinations(release_fwd)
            FailureTrans() .add_sources(pilot_slot) .add_destinations(fail)
            PilotTrans(GoalUnreachable) .add_sources(pilot_slot) .add_destinations(fail)

            CompletionTrans() .add_sources(release_fwd) .add_destinations(detach)
            FailureTrans() .add_sources(release_fwd) .add_destinations(fail)

            CompletionTrans() .add_sources(detach) .add_destinations(kick)
            FailureTrans() .add_sources(detach) .add_destinations(fail)

            CompletionTrans() .add_sources(kick) .add_destinations(post_clear)
            FailureTrans() .add_sources(kick) .add_destinations(fail)

            CompletionTrans() .add_sources(post_clear) .add_destinations(done)
            FailureTrans() .add_sources(post_clear) .add_destinations(fail)

            CompletionTrans() .add_sources(fail) .add_destinations(fail_done)

            return self

    def setup(self):
        super().setup()
        dispatch = self.children["dispatch"]
        cmdgetbook = self.CmdGetBook() .set_name("cmdgetbook") .set_parent(self)
        DataTrans(re.compile(r"#getbook\s+")) .add_sources(dispatch) .add_destinations(cmdgetbook)
        CNextTrans() .add_sources(cmdgetbook) .add_destinations(dispatch)
        cmdreturnbook = self.CmdReturnBook() .set_name("cmdreturnbook") .set_parent(self)
        DataTrans(re.compile(r"#returnbook\s+")) .add_sources(dispatch) .add_destinations(cmdreturnbook)
        CNextTrans() .add_sources(cmdreturnbook) .add_destinations(dispatch)


__all__ = ["CelesteLibrarian", "CELESTE_VERSION", "CELESTE_LIBRARIAN_VERSION"]

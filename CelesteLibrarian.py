"""
Celeste personality + librarian book delivery.

Extends stock :class:`Celeste` with the same GPT preamble and command dispatch, plus
``#getbook N`` to fetch a book by spine ArUco id (or 1-based shelf index). Requires
``vex-aim-tools`` and ``vex-aim-librarian`` on ``PYTHONPATH``.

From ``simple_cli``::

    runfsm('CelesteLibrarian')

Sequence for ``#getbook``: capture current pose → pilot to book → settle → optional
engage forward → attach → small shelf-clearance back-off → pilot back to the captured
pose → turn 180° to present to the patron → wait until spine marker is no longer seen
→ turn back.

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
from aim_librarian.book_manip import AttachBook, WaitUntilBookRemoved
from aim_librarian.books import BookObj

from Celeste import CELESTE_VERSION, Celeste, new_preamble

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
"""

class CelesteLibrarian(Celeste):
    """Celeste + localized librarian field (corner ArUco only) and ``#getbook`` macro."""

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

    def setup(self):
        super().setup()
        dispatch = self.children["dispatch"]
        cmdgetbook = self.CmdGetBook() .set_name("cmdgetbook") .set_parent(self)
        DataTrans(re.compile(r"#getbook\s+")) .add_sources(dispatch) .add_destinations(cmdgetbook)
        CNextTrans() .add_sources(cmdgetbook) .add_destinations(dispatch)


__all__ = ["CelesteLibrarian", "CELESTE_VERSION"]

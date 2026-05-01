"""
Celeste personality + librarian book delivery.

Extends stock :class:`Celeste` with the same GPT preamble and command dispatch, plus
``#getbook N`` to fetch a book by spine ArUco id (or 1-based shelf index), and
``#returnbook S`` to take a book from the patron staging area and shelve it into an
empty slot ``S`` (slot index 1–4 left-to-right, or spine id 9–12). Requires
``vex-aim-tools`` and ``vex-aim-librarian`` on ``PYTHONPATH``.

From ``simple_cli``::

    runfsm('CelesteLibrarian')

Sequence for ``#getbook``: pilot to book → settle → engage forward → attach → 8 cm back-off
→ turn −180° → pilot to patron hand-off (layout +2, −28.5 cm) → hard kicker ping → same homing
sequence as ``#gohome``: pilot toward marker 16 (vision or map), corner shove, ``#localize`` scan,
then turn to face fixed map pose of ArUco marker **20**.

Sequence for ``#returnbook``: verify the robot is not already holding a book and the
target shelve cell is vacant → turn 90° right (robot convention −90°) → resolve which
return-row spine (9–12) is on the staging pad from map + vision → pilot to that book →
pick it up → turn toward and pilot to the shelve standoff → release at the target slot pose
(slots 2–3 use ``RETURN_SHELF_BOOK10_*``; slot 1 (**spine 9**) at ArUco **19** minus **17 mm + 13.5 cm** in **X**, same **Y** as **19**, heading **π + π/2**; slot 4 is exactly marker **19** at **π**).

Tune poses and distances for your field (class attributes on :class:`CelesteLibrarian`).

Book **9** is seeded on the map at **return slot 1** (see :meth:`CelesteLibrarian.seed_librarian_world`);
other :class:`BookObj` instances are normally created from the camera when spine ArUco ids are seen.
"""

from __future__ import annotations

from math import pi, radians

import vex

from aim_fsm import *
from aim_fsm.aim_kin import AIMKinematics
from aim_fsm.particle import ArucoCombinedSensorModel
from aim_fsm.pilot import PilotToPose
from aim_fsm.utils import PoseEstimate

from aim_librarian import (
    BOOK_FIRST_ID,
    BOOK_LAST_ID,
    PilotToBook,
    install_librarian_extensions,
    is_book_aruco_id,
)
from aim_librarian.worldmap_ext import (
    ensure_bookobj_from_vision,
    ensure_patron_return_staging_bookobj,
    pick_return_staging_book_marker_id,
)
from aim_librarian.book_manip import AttachBook, ClearBookHolding, DetachBookAtPose
from aim_librarian.books import BookObj
from aim_librarian.pilot_ext import TurnTowardPose

from Celeste import CELESTE_VERSION, Celeste, new_preamble

CELESTE_LIBRARIAN_VERSION = "1.2"

# Voice layer for GPT: appended before operational librarian rules (still obeys Celeste safety).
_LIBRARIAN_PERSONALITY = """
  # LIBRARIAN PERSONALITY (this session).
  You are Celeste on "stacks duty": a pocket-sized mobile librarian for a magnet-shelf
  mini-library. You are warm, a little bookish, and quietly delighted when spines line
  up straight and patrons bring things back. Default to short, friendly sentences—never
  condescending. You may drop one light metaphor (catalog, chapter, dog-eared page,
  overdue joke) when it fits; do not info-dump about classification systems unless asked.

  When you are not emitting a body command, stay chatty-but-brief. After a successful
  ``#getbook`` run completes, offer one cheerful line that names the spine or slot you
  fetched (e.g., "Book 12, coming right up—enjoy."). After ``#returnbook``, confirm the
  slot in plain words ("Shelved in the leftmost column—thanks for closing the loop.").
  If something fails, stay calm and invite a simple retry.

  You still follow all base Celeste safety and child-interaction rules: vocabulary
  appropriate for students about 9–14, same refusals for unsafe topics, and honesty that
  you are software—not real emotions, just a stubborn love of tidy shelves.

  Optional flair: ``#emoji`` or ``#flash`` only when it will not delay motion. Never put
  prose on the same line as ``#getbook``, ``#returnbook``, or ``#gohome``; those commands
  stay alone on their lines exactly as specified below.
"""

# Public alias for demos / UI that want to display the personality blurb.
CELESTE_LIBRARIAN_PERSONALITY = _LIBRARIAN_PERSONALITY

_RETURN_SHELF_FIRST = 9
_RETURN_SHELF_LAST = 12
_RETURN_SHELF_NSLOTS = _RETURN_SHELF_LAST - _RETURN_SHELF_FIRST + 1

_LIBRARIAN_BOOK_CATALOG = f"""
  NAMED TITLES ON THIS FIELD (map the patron's words to spine ArUco id ``N`` for ``#getbook N``):
  - Spine id 9 — Dune: science fiction about space.
  - Spine id 10 — I, Robot: science fiction about robots.
  - Spine id 11 — It: horror.
  - Spine id 12 — Jaws: action.

  When the user asks by title, nickname, or genre (e.g. "get me Dune", "the horror book",
  "sci-fi about robots", "something with sharks", "space sci-fi"), infer the matching id from
  this list and output ``#getbook N`` with that id. If two or more titles could match and they
  do not narrow it (e.g. both are sci-fi), reply with one short clarifying question in plain text and
  do not output ``#getbook`` until they disambiguate. Disambiguation hints: wording about robots
  or Asimov → 10; deserts, sandworms, spice, or Arrakis → 9; horror or small-town kids vs monster → 11;
  sharks, Amity Island, or ocean thriller action → 12.

  Spine ids {BOOK_FIRST_ID}..{BOOK_LAST_ID} may exist on the shelf; only 9–12 are catalogued above.
  For other ids the patron must give a number or shelf slot, as in the rule below.
"""

_LIBRARIAN_PREAMBLE_EXTENSION = _LIBRARIAN_PERSONALITY + f"""
  # LIBRARIAN SECTION.
  You also help patrons in a small library with magnet-held books on a shelf.
  Books are world objects whose names look like ``Book-N.a``, with spine ArUco ids
  in the range {BOOK_FIRST_ID}..{BOOK_LAST_ID}.
{_LIBRARIAN_BOOK_CATALOG}
  IMPORTANT BOOK-HANDLING RULE (overrides the body-control section):
  Whenever the user refers to a book — by spine id, by shelf slot, by title or genre as in
  NAMED TITLES above, or by phrases like "get book 9", "go get book 9", "fetch book 9",
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
  (slot 1 → spine {_RETURN_SHELF_FIRST}: ArUco **19** pose − **17 mm − 13.5 cm** in **X**, same **Y**;
  detach θ = **π + π/2** (90° inward vs other slots); slot {_RETURN_SHELF_NSLOTS} → spine {_RETURN_SHELF_LAST}: exactly ArUco **19**, θ = **π**), or (b) a spine id
  {_RETURN_SHELF_FIRST}..{_RETURN_SHELF_LAST} naming which **empty** slot to fill.

  NEVER use ``#pilottoobject``, ``#pickup``, ``#drop``, or other body-control commands for this;
  ``#returnbook`` is the only correct command for shelving a patron-returned book.
  Output nothing else on the line. After the trip completes you may speak briefly.

  HOMING-CORNER RULE:
  When the user asks you to dock in the homing corner, go home, park against the stable
  corner marker (ArUco id 16), or similarly return to your physical calibration corner,
  you MUST output exactly one line ``#gohome`` — not ``#forward`` / ``#turn`` improvisations —
  whenever that behavior is implemented for your session (marker 16 must be visible in the
  camera before starting). Output nothing else on the line until the maneuver completes,
  then you may speak briefly if appropriate.
"""

class CelesteLibrarian(Celeste):
    """Celeste + librarian field: ``#getbook`` and ``#returnbook`` macros."""

    # Field layout (mm, radians) — match ``world_setup/WorldSetup.fsm`` / ``SwapBooksDemo``.
    _TAG_Y = 138
    _TAG_X = 175
    BOX_WIDTH = 35

    # Field layout shift (−3 cm X, −3 cm Y) vs prior librarian coordinates.
    SETUP_SHIFT_X_MM = -30.0
    SETUP_SHIFT_Y_MM = -30.0
    HOMING_MARKER_ID_SEED = 16

    # Corner homing offset vs prior reference (mm): robot start moved −1 cm in X, +14.5 cm in Y.
    HOMING_DELTA_X_MM = -10.0
    HOMING_DELTA_Y_MM = 145.0

    ROBOT_STANDOFF_MM = 115.0
    # Two-stage book approach.
    #   Stage 1 (path planner): stop with the robot's MAGNET face roughly
    #       ``ENGAGE_MM`` short of the book's spine surface, perpendicular to
    #       the shelf. Splitting the approach this way keeps the path planner
    #       from cutting close to neighboring books.
    #   Stage 2: drive straight forward by ``ENGAGE_MM + GETBOOK_ENGAGE_EXTRA_MM`` so the
    #       magnet meets the spine (extra mimics ``RETURN_STAGING_ENGAGE_EXTRA_MM`` when RRT
    #       stops slightly short or the field magnet needs more compression).
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
    # Shelf ``#getbook`` only: added to the post-pilot ``Forward`` (not to ``BOOK_APPROACH_OFFSET_MM``)
    # so the path planner standoff stays the same but the final creep reaches the spine.
    GETBOOK_ENGAGE_EXTRA_MM = 5.0
    GETBOOK_SHELF_ENGAGE_MM = ENGAGE_MM + GETBOOK_ENGAGE_EXTRA_MM
    BOOK_APPROACH_OFFSET_MM = (
        AIMKinematics.body_diameter / 2     # 28.5 mm: center -> magnet face
        + BookObj.SPINE_THICKNESS_MM / 2    # 12.7 mm: marker -> spine surface
        + ENGAGE_MM                         # 50.0 mm: pre-engage standoff gap
    )
    # Short straight-line back-off after attach (8 cm away from shelf) before the in-place
    # presentation turn (−180°) and patron hand-off pilot; keeps the path planner away from shelf clutter.
    SHELF_CLEARANCE_MM = 80.0
    SETTLE_S = 0.4
    # ``#returnbook``: pause after ``TurnTowardStagedBook`` (−90°) before pilot to staging.
    RETURN_BOOK_POST_TURN_HOLD_S = 3.0
    PRESENT_TURN_DEG = 180.0
    # Degrees per second for the −180° presentation turn. Slow enough to
    # look deliberate when handing the book to a patron, and slow enough that
    # the IMU's gyro-threshold check doesn't false-trigger a "LOST" pose.
    PRESENT_TURN_SPEED_DPS = 45.0
    # Magnet knock after patron takes the book; stronger than shelving ``Kick()`` default.
    GETBOOK_POST_RELEASE_KICK = vex.KickType.HARD
    # Homing-after-getbook matches ``Celeste.CmdGoHome`` standoff and corner bump (see ``gb_*_home`` states).

    # Patron return / ``#returnbook`` — layout numbers are **centimeters**, converted to mm.
    _RETURN_CM_TO_MM = 10.0
    # Homing ArUco 16: **X** = −3 cm in world frame; **Y** and **theta** match ArucoMarker-20.
    HOMING_MARKER16_X_MM = -3.0 * _RETURN_CM_TO_MM

    # Patron presentation after ``#getbook``: layout (+2, −28.5) cm with the same cm→mm
    # bookkeeping as ``RETURN_*`` (`HOMING_DELTA_*`, ``SETUP_SHIFT_*``)—field X with homing
    # marker 16 line, field Y with ArucoMarker-19 corridor line.
    GETBOOK_PATRON_PRESENT_X_CM = 2.0
    GETBOOK_PATRON_PRESENT_Y_CM = -28.5
    GETBOOK_PATRON_PRESENT_X_MM = GETBOOK_PATRON_PRESENT_X_CM * _RETURN_CM_TO_MM + HOMING_DELTA_X_MM + SETUP_SHIFT_X_MM
    GETBOOK_PATRON_PRESENT_Y_MM = GETBOOK_PATRON_PRESENT_Y_CM * _RETURN_CM_TO_MM + HOMING_DELTA_Y_MM + SETUP_SHIFT_Y_MM

    @classmethod
    def marker20_y_mm(cls) -> float:
        return cls._TAG_Y - cls.BOX_WIDTH / 2 + cls.SETUP_SHIFT_Y_MM

    @classmethod
    def marker20_x_mm(cls) -> float:
        """Matches ``ArucoMarker-20`` X in librarian landmarks / ``seed_librarian_world``."""
        return cls._TAG_X + cls.SETUP_SHIFT_X_MM

    @classmethod
    def marker19_xy_mm(cls) -> tuple[float, float]:
        """World *(x, y)* in mm for ArUco marker **19** (matches ``seed_librarian_world``).

        **X** matches marker **20** (both ``_TAG_X + SETUP_SHIFT_X_MM``).
        """
        return (
            cls._TAG_X + cls.SETUP_SHIFT_X_MM,
            -cls._TAG_Y + cls.BOX_WIDTH / 2 + cls.SETUP_SHIFT_Y_MM,
        )

    RETURN_STAGING_X_MM = HOMING_DELTA_X_MM + SETUP_SHIFT_X_MM
    RETURN_STAGING_Y_MM = -14.0 * _RETURN_CM_TO_MM + HOMING_DELTA_Y_MM + SETUP_SHIFT_Y_MM
    RETURN_SHELF_MARKER_IDS = (9, 10, 11, 12)
    RETURN_SLOT_Y_MM = {
        # Slot 1 / spine 9: same Y as ArUco 19; X is M19 X − ``RETURN_SLOT1_MINUS_M19_X_MM``.
        9: -_TAG_Y + BOX_WIDTH / 2 + SETUP_SHIFT_Y_MM,
        10: 2.25 * _RETURN_CM_TO_MM + HOMING_DELTA_Y_MM + SETUP_SHIFT_Y_MM,
        11: -2.25 * _RETURN_CM_TO_MM + HOMING_DELTA_Y_MM + SETUP_SHIFT_Y_MM,
        # Slot 4 / spine 12: exact seeded ArUco 19 *(x, y)*.
        12: -_TAG_Y + BOX_WIDTH / 2 + SETUP_SHIFT_Y_MM,
    }
    # Slot 1: M19 X − 17 mm − 13.5 cm; Y = M19; θ = π + π/2 for detach/seed (see ``RETURN_SLOT1_DETACH_THETA_RAD``).
    RETURN_SLOT1_MINUS_M19_X_MM = 17.0 + 13.5 * _RETURN_CM_TO_MM
    RETURN_SLOT1_DETACH_THETA_RAD = 3 * pi / 2
    # Staging pickup only: drive a bit farther forward than ``ENGAGE_MM`` so the magnet meets the spine.
    RETURN_STAGING_ENGAGE_EXTRA_MM = 20.0
    RETURN_STAGING_ENGAGE_MM = ENGAGE_MM + RETURN_STAGING_ENGAGE_EXTRA_MM
    # After pickup, shelve at this world pose (cm → mm); includes ``HOMING_DELTA_*``.
    RETURN_SHELF_BOOK10_X_MM = -2.25 * _RETURN_CM_TO_MM + HOMING_DELTA_X_MM + SETUP_SHIFT_X_MM
    RETURN_SHELF_BOOK10_Y_MM = -18.0 * _RETURN_CM_TO_MM + HOMING_DELTA_Y_MM + SETUP_SHIFT_Y_MM
    # Layout-only book-10 column anchor (mm); matches cm terms in ``RETURN_SHELF_BOOK10_*`` before shifts.
    _RETURN_SHELF_LAYOUT_BOOK10_X_MM = -2.25 * _RETURN_CM_TO_MM
    _RETURN_SHELF_LAYOUT_BOOK10_Y_MM = -18.0 * _RETURN_CM_TO_MM
    RETURN_SLOT_OCCUPANCY_TOL_MM = 28.0
    RETURN_SHELF_ROW_X_TOL_MM = 55.0
    RETURN_RELEASE_BACK_MM = -38.0
    RETURN_POST_DROP_CLEAR_MM = -60.0
    # World pose fed to ``PilotToSlotStandoff`` / ``TurnTowardTargetSlot``: values are *before*
    # adding ``HOMING_DELTA_*`` and ``SETUP_SHIFT_*`` on each axis.
    #
    # Skinny RRT places ~20×40 mm pads on seeded corner ArUcos. This standoff must clear them with
    # the robot hull at θ≈0: marker **17** ~(127.5, 108); marker **20** ~(145.0, 90.5) along the shelf
    # front—independent of camera (``GoalCollides`` is pure geometry vs ``world_map``).
    RETURN_BOOK_SHELF_PILOT_OFFSET_X_MM = 250.0
    RETURN_BOOK_SHELF_PILOT_OFFSET_Y_MM = -25.0
    # Vector from detach pose to RRT pilot standoff for ``#returnbook`` (slots 1, 4 off-book10-column).
    _RETURN_BOOK_PILOT_MINUS_DETACH_DX_MM = (
        RETURN_BOOK_SHELF_PILOT_OFFSET_X_MM - _RETURN_SHELF_LAYOUT_BOOK10_X_MM
    )
    _RETURN_BOOK_PILOT_MINUS_DETACH_DY_MM = (
        RETURN_BOOK_SHELF_PILOT_OFFSET_Y_MM - _RETURN_SHELF_LAYOUT_BOOK10_Y_MM
    )
    # ``PilotToPose`` always runs the RRT (see ``aim_fsm.pilot.PilotToPose``). Staging sits near corner
    # ``ArucoMarker-18``; a shallow +X standoff can collide that goal with marker 18's inflated obstacle
    # — add extra +X (`RETURN_STAGING_EXTRA_X_MM`).
    RETURN_STAGING_EXTRA_X_MM = 100.0

    @classmethod
    def shelf_x_mm(cls) -> float:
        return cls._TAG_X - 45.0 + cls.SETUP_SHIFT_X_MM

    @classmethod
    def default_home_pose(cls) -> Pose:
        """Standoff in front of the shelf row, facing the shelf (heading 0)."""
        return Pose(
            cls.shelf_x_mm() - 185.0 + cls.HOMING_DELTA_X_MM,
            cls.HOMING_DELTA_Y_MM + cls.SETUP_SHIFT_Y_MM,
            0.0,
            0.0,
        )

    @classmethod
    def return_book_detach_xy_mm(cls, spine_id: int) -> tuple[float, float]:
        """Map pose (mm) where ``DetachBookAtPose`` drops the book for ``#returnbook``."""
        if spine_id == _RETURN_SHELF_LAST:
            # Slot 4: same *(x, y)* as seeded ArUco 19 (= same X as ArUco 20).
            return cls.marker19_xy_mm()
        if spine_id == _RETURN_SHELF_FIRST:
            mx, my = cls.marker19_xy_mm()
            return mx - cls.RETURN_SLOT1_MINUS_M19_X_MM, my
        return cls.RETURN_SHELF_BOOK10_X_MM, cls.RETURN_SHELF_BOOK10_Y_MM

    @classmethod
    def return_book_detach_theta_rad(cls, spine_id: int) -> float:
        """Heading (rad) for ``DetachBookAtPose`` / seeded Book-9; slot 1 is +90° inward vs ``π``."""
        if spine_id == _RETURN_SHELF_FIRST:
            return cls.RETURN_SLOT1_DETACH_THETA_RAD
        return pi

    @classmethod
    def return_book_pilot_standoff_xy_mm(cls, spine_id: int) -> tuple[float, float]:
        """RRT goal (mm) before the final shelve creep; keeps fixed offset from detach pose."""
        dx, dy = cls.return_book_detach_xy_mm(spine_id)
        return (
            dx + cls._RETURN_BOOK_PILOT_MINUS_DETACH_DX_MM,
            dy + cls._RETURN_BOOK_PILOT_MINUS_DETACH_DY_MM,
        )

    def __init__(self, **kwargs):
        sx = self.SETUP_SHIFT_X_MM
        sy = self.SETUP_SHIFT_Y_MM
        m18_x = self._TAG_X - self.BOX_WIDTH / 2 + sx
        m18_y = -self._TAG_Y + sy
        m20_y = self._TAG_Y - self.BOX_WIDTH / 2 + sy
        landmarks = {
            "ArucoMarker-17": Pose(
                self._TAG_X - self.BOX_WIDTH / 2 + sx, self._TAG_Y + sy, 5, radians(90)
            ),
            "ArucoMarker-18": Pose(m18_x, m18_y, 5, radians(270)),
            "ArucoMarker-19": Pose(
                self._TAG_X + sx, -self._TAG_Y + self.BOX_WIDTH / 2 + sy, 5, radians(180)
            ),
            "ArucoMarker-20": Pose(self._TAG_X + sx, m20_y, 5, radians(180)),
            "ArucoMarker-16": Pose(
                self.HOMING_MARKER16_X_MM,
                m20_y,
                5,
                radians(180),
            ),
        }
        pf = ParticleFilter(
            robot,
            num_particles=5000,
            landmarks=landmarks,
            sensor_model=ArucoCombinedSensorModel,
            # Fewer premature resamples: wait until weights are clearly peaked.
            resample_if_neff_below_frac=0.2,
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
        """Fixed corner ArUco landmarks + homing marker 16; seed Book-9 at return slot 1."""
        sx = self.SETUP_SHIFT_X_MM
        sy = self.SETUP_SHIFT_Y_MM
        m18_x = self._TAG_X - self.BOX_WIDTH / 2 + sx
        m18_y = -self._TAG_Y + sy
        m20_y = self._TAG_Y - self.BOX_WIDTH / 2 + sy
        m17 = ArucoMarkerObj(
            {"name": "ArucoMarker-17", "id": 17, "marker": None},
            x=self._TAG_X - self.BOX_WIDTH / 2 + sx,
            y=self._TAG_Y + sy,
            theta=radians(90),
        )
        m18 = ArucoMarkerObj(
            {"name": "ArucoMarker-18", "id": 18, "marker": None},
            x=m18_x,
            y=m18_y,
            theta=radians(270),
        )
        m17.is_fixed = True
        m18.is_fixed = True
        self.robot.world_map.objects["ArucoMarker-17.a"] = m17
        self.robot.world_map.objects["ArucoMarker-18.a"] = m18
        m16 = ArucoMarkerObj(
            {"name": f"ArucoMarker-{self.HOMING_MARKER_ID_SEED}", "id": self.HOMING_MARKER_ID_SEED, "marker": None},
            x=self.HOMING_MARKER16_X_MM,
            y=m20_y,
            z=5,
            theta=radians(180),
        )
        m16.is_fixed = True
        self.robot.world_map.objects[f"ArucoMarker-{self.HOMING_MARKER_ID_SEED}.a"] = m16
        m19 = ArucoMarkerObj(
            {"name": "ArucoMarker-19", "id": 19, "marker": None},
            x=self._TAG_X + sx,
            y=-self._TAG_Y + self.BOX_WIDTH / 2 + sy,
            theta=radians(180),
        )
        m20 = ArucoMarkerObj(
            {"name": "ArucoMarker-20", "id": 20, "marker": None},
            x=self._TAG_X + sx,
            y=m20_y,
            theta=radians(180),
        )
        m19.is_fixed = True
        m20.is_fixed = True
        self.robot.world_map.objects["ArucoMarker-19.a"] = m19
        self.robot.world_map.objects["ArucoMarker-20.a"] = m20
        self._seed_return_slot_book_9()

    def _seed_return_slot_book_9(self) -> None:
        """Insert Book-9 at return slot 1 if not already present (vision may add it later)."""
        wm = self.robot.world_map
        mid = _RETURN_SHELF_FIRST
        with wm._lock:
            if any(
                isinstance(o, BookObj) and o.marker_id == mid for o in wm.objects.values()
            ):
                return
        rx, ry = self.return_book_detach_xy_mm(mid)
        th = self.return_book_detach_theta_rad(mid)
        name = f"Book-{mid}"
        spec = {"name": name, "id": mid, "marker": None}
        b = BookObj(spec)
        b.pose = PoseEstimate(rx, ry, BookObj.HEIGHT_MM / 2, th)
        b.is_visible = True
        b.is_missing = False
        b.pose_confidence = +1
        # Catalog pose for RRT/navigation; must not block ``#returnbook`` vacancy when slot is empty.
        b._skip_return_slot_vacancy_check = True
        with wm._lock:
            if any(
                isinstance(o, BookObj) and o.marker_id == mid for o in wm.objects.values()
            ):
                return
            obj_id = wm.next_in_sequence(name)
            wm.objects[obj_id] = b
        print(
            f"seed_librarian_world: seeded {obj_id} at return slot 1 pose ({rx:.1f}, {ry:.1f}) mm"
        )

    def start(self):
        self.seed_librarian_world()
        self.robot.openai_client.set_preamble(new_preamble + _LIBRARIAN_PREAMBLE_EXTENSION)
        self.picked_up_handler = self.picked_up_celeste
        self.put_down_handler = self.put_down_celeste
        StateMachineProgram.start(self)

    class CmdGetBook(StateNode):
        """Book → … → homing tail through localize, then rotate to face ArUco 20 map pose."""

        marker_id: int = BOOK_FIRST_ID

        def start(self, event=None):
            if self.running:
                return
            self._parse_ok = self._parse_getbook_event(event)
            # Log pose at command start (for debug / future use).
            rp = self.robot.pose
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
                # the subsequent ``Forward(GETBOOK_SHELF_ENGAGE_MM)`` engage step drives
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

        class PilotToPatronHandoff(PilotToPose):
            """Pilot to patron presentation XY; preserves heading set by the preceding 180° turn."""

            def __init__(self, **kw):
                super().__init__(target_pose=None, **kw)

            def start(self, event=None):
                cl = CelesteLibrarian
                self.target_pose = Pose(
                    cl.GETBOOK_PATRON_PRESENT_X_MM,
                    cl.GETBOOK_PATRON_PRESENT_Y_MM,
                    0.0,
                    self.robot.pose.theta,
                )
                super().start(event)

        class TurnTowardMarker20(TurnTowardPose):
            """Rotate in place toward fixed ``ArucoMarker-20`` map pose (no vision required)."""

            def __init__(self):
                super().__init__(Pose(0.0, 0.0, 0.0, 0.0), turn_speed=CelesteLibrarian.PRESENT_TURN_SPEED_DPS)

            def start(self, event=None):
                cl = CelesteLibrarian
                self.target_pose = Pose(
                    cl.marker20_x_mm(),
                    cl.marker20_y_mm(),
                    5.0,
                    0.0,
                )
                super().start(event)

        def setup(self):
            parse = self.ParseGetBookId() .set_name("gb_parse") .set_parent(self)
            pilot = self.PilotToParsedBook(
                book_approach_offset_mm=CelesteLibrarian.BOOK_APPROACH_OFFSET_MM
            ) .set_name("gb_pilot") .set_parent(self)
            settle = StateNode() .set_name("gb_settle") .set_parent(self)
            engage = Forward(CelesteLibrarian.GETBOOK_SHELF_ENGAGE_MM) .set_name("gb_engage") .set_parent(self)
            attach = self.AttachParsedBook() .set_name("gb_attach") .set_parent(self)
            # Small mechanical clearance from the shelf so path planning from the
            # next state isn't trying to start inside the book/shelf footprint.
            back_off = Forward(-CelesteLibrarian.SHELF_CLEARANCE_MM) .set_name(
                "gb_back_off"
            ) .set_parent(self)
            turn_present = Turn(
                -CelesteLibrarian.PRESENT_TURN_DEG,
                turn_speed=CelesteLibrarian.PRESENT_TURN_SPEED_DPS,
            ) .set_name("gb_turn_present") .set_parent(self)
            pilot_patron = self.PilotToPatronHandoff() .set_name("gb_pilot_patron") .set_parent(self)
            kick_release = Kick(CelesteLibrarian.GETBOOK_POST_RELEASE_KICK) .set_name(
                "gb_kick_release"
            ) .set_parent(self)
            clear_kick_hold = ClearBookHolding() .set_name("gb_clear_kick_hold") .set_parent(self)
            pilot_home = Celeste.PilotToHomingCorner(
                CelesteLibrarian.HOMING_MARKER_ID_SEED,
                approach_standoff_mm=Celeste.CmdGoHome.APPROACH_STANDOFF_MM,
                align_heading=True,
            ) .set_name("gb_pilot_home") .set_parent(self)
            home_bump = Forward(Celeste.CmdGoHome.CORNER_SHOVE_MM) .set_name("gb_home_bump") .set_parent(
                self
            )
            home_loc = Celeste.CmdLocalize() .set_name("gb_home_localize") .set_parent(self)
            face_m20 = self.TurnTowardMarker20() .set_name("gb_face_marker20") .set_parent(self)
            done = ParentCompletes() .set_name("gb_done") .set_parent(self)
            fail = Say(
                "Drat—that fetch didn't quite work. Same book once more when you're ready?"
            ) .set_name(
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

            CompletionTrans() .add_sources(back_off) .add_destinations(turn_present)
            FailureTrans() .add_sources(back_off) .add_destinations(fail)

            CompletionTrans() .add_sources(turn_present) .add_destinations(pilot_patron)
            FailureTrans() .add_sources(turn_present) .add_destinations(fail)

            CompletionTrans() .add_sources(pilot_patron) .add_destinations(kick_release)
            PilotTrans(GoalUnreachable) .add_sources(pilot_patron) .add_destinations(fail)
            FailureTrans() .add_sources(pilot_patron) .add_destinations(fail)

            CompletionTrans() .add_sources(kick_release) .add_destinations(clear_kick_hold)
            FailureTrans() .add_sources(kick_release) .add_destinations(fail)

            CompletionTrans() .add_sources(clear_kick_hold) .add_destinations(pilot_home)

            CompletionTrans() .add_sources(pilot_home) .add_destinations(home_bump)
            PilotTrans(GoalUnreachable) .add_sources(pilot_home) .add_destinations(fail)
            FailureTrans() .add_sources(pilot_home) .add_destinations(fail)

            CompletionTrans() .add_sources(home_bump) .add_destinations(home_loc)
            FailureTrans() .add_sources(home_bump) .add_destinations(fail)

            CompletionTrans() .add_sources(home_loc) .add_destinations(face_m20)
            FailureTrans() .add_sources(home_loc) .add_destinations(fail)

            CompletionTrans() .add_sources(face_m20) .add_destinations(done)
            FailureTrans() .add_sources(face_m20) .add_destinations(fail)

            CompletionTrans() .add_sources(fail) .add_destinations(fail_done)

            return self

    class CmdReturnBook(StateNode):
        """Pilot to patron staging, resolve spine 9–12 on the pad, pick up, shelve in an empty slot."""

        target_slot_id: int = _RETURN_SHELF_FIRST
        detected_marker_id: int = _RETURN_SHELF_FIRST

        def start(self, event=None):
            if self.running:
                return
            self._parse_ok = self._parse_returnbook_event(event)
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
                tid = self.parent.target_slot_id
                sx, sy = cl.return_book_detach_xy_mm(tid)
                tol = cl.RETURN_SLOT_OCCUPANCY_TOL_MM
                xtol = cl.RETURN_SHELF_ROW_X_TOL_MM
                for obj in self.robot.world_map.objects.values():
                    if not isinstance(obj, BookObj):
                        continue
                    if getattr(obj, "_skip_return_slot_vacancy_check", False):
                        continue
                    if obj.held_by is not None:
                        continue
                    if abs(obj.pose.x - sx) > xtol:
                        continue
                    if abs(obj.pose.y - sy) <= tol:
                        print(
                            "CmdReturnBook: target shelve pose looks occupied "
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
                cl = CelesteLibrarian
                mid = self.parent.detected_marker_id
                self.marker_id = mid
                ensure_patron_return_staging_bookobj(
                    self.robot,
                    cl.RETURN_STAGING_X_MM,
                    cl.RETURN_STAGING_Y_MM,
                    mid,
                )
                super().start(event)

        class ResolveStagingBookMarker(StateNode):
            """Set ``detected_marker_id`` from map + vision (nearest to layout staging pose)."""

            def start(self, event=None):
                super().start(event)
                parent = self.parent
                cl = CelesteLibrarian
                mid = pick_return_staging_book_marker_id(
                    self.robot,
                    cl.RETURN_STAGING_X_MM,
                    cl.RETURN_STAGING_Y_MM,
                    cl.RETURN_SHELF_MARKER_IDS,
                )
                parent.detected_marker_id = mid
                print(f"CmdReturnBook: staging pickup spine id {mid}")
                self.post_completion()

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
                px, py = cl.return_book_pilot_standoff_xy_mm(self.parent.target_slot_id)
                self.target_pose = Pose(px, py, 0.0, 0.0)
                super().start(event)

        class PilotToSlotStandoff(PilotToPose):
            def __init__(self, **kw):
                super().__init__(target_pose=None, **kw)

            def start(self, event=None):
                cl = CelesteLibrarian
                px, py = cl.return_book_pilot_standoff_xy_mm(self.parent.target_slot_id)
                self.target_pose = Pose(px, py, 0.0, 0.0)
                super().start(event)

        class DetachAtTargetSlot(DetachBookAtPose):
            def __init__(self):
                super().__init__(Pose(0.0, 0.0, BookObj.HEIGHT_MM / 2, pi))

            def start(self, event=None):
                cl = CelesteLibrarian
                tid = self.parent.target_slot_id
                dx, dy = cl.return_book_detach_xy_mm(tid)
                th = cl.return_book_detach_theta_rad(tid)
                self.pose = Pose(dx, dy, BookObj.HEIGHT_MM / 2, th)
                super().start(event)

        def setup(self):
            parse = self.ParseReturnSlot() .set_name("rb_parse") .set_parent(self)
            not_hold = self.CheckNotHolding() .set_name("rb_not_holding") .set_parent(self)
            vacant = self.CheckTargetSlotVacant() .set_name("rb_slot_vacant") .set_parent(self)
            turn_staged = self.TurnTowardStagedBook() .set_name("rb_turn_staged") .set_parent(self)
            wait_after_turn = StateNode() .set_name("rb_wait_after_turn_staged") .set_parent(self)
            resolve_staging = self.ResolveStagingBookMarker() .set_name("rb_resolve_staging") .set_parent(self)
            pilot_book = self.PilotToStagedBook(
                book_approach_offset_mm=CelesteLibrarian.BOOK_APPROACH_OFFSET_MM
            ) .set_name("rb_pilot_book") .set_parent(self)
            settle_b = StateNode() .set_name("rb_settle_b") .set_parent(self)
            engage = Forward(CelesteLibrarian.RETURN_STAGING_ENGAGE_MM) .set_name("rb_engage") .set_parent(self)
            attach = self.AttachDetectedBook() .set_name("rb_attach") .set_parent(self)
            # Same as ``CmdGetBook`` ``gb_back_off``: short straight clearance after attach so RRT &
            # the next maneuver do not start inside the book footprint (mirror shelf vs staging).
            back_off = Forward(-CelesteLibrarian.SHELF_CLEARANCE_MM) .set_name(
                "rb_back_off"
            ) .set_parent(self)
            turn_slot = self.TurnTowardTargetSlot() .set_name("rb_turn_slot") .set_parent(self)
            pilot_slot = self.PilotToSlotStandoff() .set_name("rb_pilot_slot") .set_parent(self)
            release_fwd = Forward(CelesteLibrarian.RETURN_RELEASE_BACK_MM) .set_name(
                "rb_release_fwd"
            ) .set_parent(self)
            detach = self.DetachAtTargetSlot() .set_name("rb_detach") .set_parent(self)
            kick = Kick() .set_name("rb_kick") .set_parent(self)
            clear_shelf_kick_hold = ClearBookHolding() .set_name("rb_clear_kick_hold") .set_parent(self)
            post_clear = Forward(CelesteLibrarian.RETURN_POST_DROP_CLEAR_MM) .set_name(
                "rb_post_clear"
            ) .set_parent(self)
            done = ParentCompletes() .set_name("rb_done") .set_parent(self)
            fail = Say(
                "Hmm—I couldn't finish shelving that one. Peek at the staging pad and the "
                "empty slot, then we can try again."
            ) .set_name("rb_fail") .set_parent(self)
            fail_done = ParentCompletes() .set_name("rb_fail_done") .set_parent(self)

            CompletionTrans() .add_sources(parse) .add_destinations(not_hold)
            FailureTrans() .add_sources(parse) .add_destinations(fail)

            CompletionTrans() .add_sources(not_hold) .add_destinations(vacant)
            FailureTrans() .add_sources(not_hold) .add_destinations(fail)

            CompletionTrans() .add_sources(vacant) .add_destinations(turn_staged)
            FailureTrans() .add_sources(vacant) .add_destinations(fail)

            CompletionTrans() .add_sources(turn_staged) .add_destinations(wait_after_turn)
            FailureTrans() .add_sources(turn_staged) .add_destinations(fail)

            TimerTrans(CelesteLibrarian.RETURN_BOOK_POST_TURN_HOLD_S) .add_sources(
                wait_after_turn
            ) .add_destinations(resolve_staging)

            CompletionTrans() .add_sources(resolve_staging) .add_destinations(pilot_book)

            CompletionTrans() .add_sources(pilot_book) .add_destinations(settle_b)
            FailureTrans() .add_sources(pilot_book) .add_destinations(fail)
            PilotTrans(GoalUnreachable) .add_sources(pilot_book) .add_destinations(fail)

            TimerTrans(CelesteLibrarian.SETTLE_S) .add_sources(settle_b) .add_destinations(engage)

            CompletionTrans() .add_sources(engage) .add_destinations(attach)
            FailureTrans() .add_sources(engage) .add_destinations(fail)

            CompletionTrans() .add_sources(attach) .add_destinations(back_off)
            FailureTrans() .add_sources(attach) .add_destinations(fail)

            CompletionTrans() .add_sources(back_off) .add_destinations(turn_slot)
            FailureTrans() .add_sources(back_off) .add_destinations(fail)

            CompletionTrans() .add_sources(turn_slot) .add_destinations(pilot_slot)
            FailureTrans() .add_sources(turn_slot) .add_destinations(fail)

            CompletionTrans() .add_sources(pilot_slot) .add_destinations(release_fwd)
            FailureTrans() .add_sources(pilot_slot) .add_destinations(fail)
            PilotTrans(GoalUnreachable) .add_sources(pilot_slot) .add_destinations(fail)

            CompletionTrans() .add_sources(release_fwd) .add_destinations(detach)
            FailureTrans() .add_sources(release_fwd) .add_destinations(fail)

            CompletionTrans() .add_sources(detach) .add_destinations(kick)
            FailureTrans() .add_sources(detach) .add_destinations(fail)

            CompletionTrans() .add_sources(kick) .add_destinations(clear_shelf_kick_hold)
            FailureTrans() .add_sources(kick) .add_destinations(fail)

            CompletionTrans() .add_sources(clear_shelf_kick_hold) .add_destinations(post_clear)

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


__all__ = [
    "CelesteLibrarian",
    "CELESTE_VERSION",
    "CELESTE_LIBRARIAN_VERSION",
    "CELESTE_LIBRARIAN_PERSONALITY",
]

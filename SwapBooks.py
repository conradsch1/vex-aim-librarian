"""
Restore the canonical shelf order by physically swapping two adjacent books.

Corner ArUco landmark **positions** (mm) and particle-filter **poses** match
``SwapBooksDemo.fsm``; seeded ``ArucoMarkerObj`` **orientations** match that demo's
``seed_world`` (radians). Field constants ``_TAG_X`` / ``_TAG_Y`` / ``BOX_WIDTH``
are the same as there. Shelf geometry uses ``aim_librarian.shelf_slots`` for
slot centers; a *staging* parking spot lives on the robot's start x-axis 140 mm
south of the origin.

Canonical slot order (when full)::

    slot 0 (leftmost,  +y) -> book 9
    slot 1                 -> book 10
    slot 2                 -> book 11
    slot 3 (rightmost, -y) -> book 12

Only the **two books being swapped** are inserted into the world map at startup
(their ids and swapped poses). Other spine markers / books are **not** invented:
they appear when the camera sees them (librarian world map). Empty slots stay
free of phantom ``BookObj`` entries so RRT does not collide with books that are
not on the shelf.

Procedure
---------
1. ``_LocalizeSweep`` rotates -45 deg, +45 deg, then back to 0 (relative to start)
   so the rear-corner ArUco landmarks come into view; the particle filter latches
   before any navigation runs.
2. Three reusable ``_PickAndPlace`` legs perform the swap:
     a) book *a* (in slot_b) -> staging
     b) book *b* (in slot_a) -> slot_b
     c) book *a* (at staging) -> slot_a
3. Any failure inside any leg jumps to a single fail-fast sink that prints what
   failed and exits via ``ParentCompletes``. No recovery is attempted.

From ``simple_cli``::

    runfsm('swapbooks')

Then type ``start`` (or ``tm start``).

Regenerate Python with::

    python3 path/to/vex-aim-tools/genfsm swapbooks.fsm swapbooks.py
"""

from __future__ import annotations

from math import nan, pi

from aim_fsm import *
from aim_fsm.particle import ArucoCombinedSensorModel
from aim_fsm.pilot import PilotToPose

from aim_librarian import BookObj, BOOK_FIRST_ID, install_librarian_extensions
from aim_librarian.book_manip import AttachBook, DetachBookAtPose
from aim_librarian.pilot_ext import PilotToBook, TurnTowardPose
from aim_librarian.shelf_slots import NUM_SHELF_SLOTS, book_center_pose

# Same field constants as ``SwapBooksDemo.fsm`` / ``world_setup/WorldSetup.fsm``.
# The four fixed ArUco tags (17--20) sit on the corner boxes that frame the field,
# and ``_TAG_X`` / ``_TAG_Y`` are the distances from the field origin to those boxes.
_TAG_Y = 138
_TAG_X = 175
BOX_WIDTH = 35

# Off-shelf staging: ``x = 0`` on the robot start line, ``y = -140`` mm (= 14 cm south).
# Use ``theta = pi`` like :func:`aim_librarian.shelf_slots.book_center_pose` so the
# book footprint / viewer matches shelf books (``theta = 0`` placed the spine wrong).
_STAGING_X_MM = 0.0
_STAGING_Y_MM = -140.0


def _aruco_landmarks() -> dict:
    """Ground-truth poses for the particle filter — identical to ``SwapBooksDemo``."""
    return {
        "ArucoMarker-17": Pose(_TAG_X - BOX_WIDTH / 2, _TAG_Y, 5, 90),
        "ArucoMarker-18": Pose(_TAG_X - BOX_WIDTH / 2, -_TAG_Y, 5, 270),
        "ArucoMarker-19": Pose(_TAG_X, -_TAG_Y + BOX_WIDTH / 2, 5, 180),
        "ArucoMarker-20": Pose(_TAG_X, _TAG_Y - BOX_WIDTH / 2, 5, 180),
    }


class SwapBooks(StateMachineProgram):
    """Localize, then physically swap two adjacent shelf books.

    Default behavior swaps book 9 (canonically slot 0) with book 10 (canonically
    slot 1).
    """

    # --- Tunables (mm / s) ----------------------------------------------------
    # Drive-center to spine clearance when stopped in front of a slot. Must
    # leave room for the magnet to reach the spine.
    ROBOT_STANDOFF_MM = 115.0
    # Pick-up recipe matches ``CelesteLibrarian.CmdGetBook``: PilotToBook (optional spine
    # offset) -> settle -> Forward(ENGAGE_MM) -> AttachBook -> retreat. Defaults match that
    # class: approach offset 0 (PilotToBook uses raw spine / map goal), ENGAGE_MM 0 (magnet
    # mates when the pilot completes). Increase ENGAGE_MM if you need a final creep.
    BOOK_APPROACH_OFFSET_MM = 0.0
    ENGAGE_MM = 0.0
    SETTLE_S = 0.4
    # How far to back up after attach/detach so we clear the shelf before turning.
    SHELF_RETREAT_MM = 300.0
    # After leg 2 the robot is still near ``x ~= _TAG_X`` with only ``POST_DROP_CLEAR_MM``.
    # Turning straight toward staging ``(0, -140)`` swings the body toward the +y / shelf
    # corner (ArUco 17/20). Back off the row first, then ``TurnTowardPose``.
    POST_PLACE_ROW_CLEAR_MM = 300.0
    # Small reverse nudge after a drop so the robot doesn't catch the book on its
    # way out (negative = backwards).
    POST_DROP_CLEAR_MM = -60.0
    # Back away this far from the shelf *before* releasing on a slot, so the
    # detached book lines up with the slot center (negative = backwards).
    RELEASE_BACK_MM = -38.0

    # Localization sweep amplitude / dwell.
    LOCALIZE_HALF_ANGLE_DEG = 45.0
    LOCALIZE_SETTLE_S = 0.6
    # Extra pause after the sweep returns to heading 0 so the particle filter can
    # converge while the robot is still (before any ``PilotToBook`` / shelf motion).
    POST_LOCALIZE_WAIT_S = 0.45

    def __init__(
        self,
        book_id_a: int = BOOK_FIRST_ID,         # default book 9
        book_id_b: int = BOOK_FIRST_ID + 1,     # default book 10
        slot_a: int = 0,                         # canonical slot of book a (leftmost)
        slot_b: int = 1,                         # canonical slot of book b
        seed_initial_layout: bool = True,
        **kwargs,
    ):
        if not 0 <= slot_a < NUM_SHELF_SLOTS:
            raise ValueError(f"slot_a must be 0..{NUM_SHELF_SLOTS - 1}, got {slot_a!r}")
        if not 0 <= slot_b < NUM_SHELF_SLOTS:
            raise ValueError(f"slot_b must be 0..{NUM_SHELF_SLOTS - 1}, got {slot_b!r}")
        if slot_a == slot_b:
            raise ValueError("slot_a and slot_b must differ")
        if book_id_a == book_id_b:
            raise ValueError("book_id_a and book_id_b must differ")

        self.book_id_a = int(book_id_a)
        self.book_id_b = int(book_id_b)
        self.slot_a = int(slot_a)
        self.slot_b = int(slot_b)
        self.seed_initial_layout = bool(seed_initial_layout)

        # Canonical book-center poses for slots a and b.
        self._pose_slot_a = book_center_pose(self.slot_a, tag_x_mm=_TAG_X)
        self._pose_slot_b = book_center_pose(self.slot_b, tag_x_mm=_TAG_X)
        # Where book a temporarily lives between legs 1 and 3.
        self._pose_staging_drop = Pose(
            _STAGING_X_MM, _STAGING_Y_MM, BookObj.HEIGHT_MM / 2.0, pi
        )
        # Robot pose for the staging drop / re-pick: stand ROBOT_STANDOFF_MM
        # north of the drop spot facing south so the front-mounted magnet
        # hovers over (0, -140).
        self._pose_staging_robot = Pose(
            _STAGING_X_MM,
            _STAGING_Y_MM + self.ROBOT_STANDOFF_MM,
            0.0,
            -pi / 2,
        )
        # World-map keys (".a" suffix = "apparent"); used by seed/detach.
        self._key_a = f"Book-{self.book_id_a}.a"
        self._key_b = f"Book-{self.book_id_b}.a"

        pf = ParticleFilter(
            robot,
            num_particles=500,
            landmarks=_aruco_landmarks(),
            sensor_model=ArucoCombinedSensorModel,
        )
        opts = dict(
            particle_filter=pf,
            wall_marker_dict=None,
            speech=False,
            launch_particle_viewer=True,
            launch_path_viewer=True,
            launch_worldmap_viewer=True,
            launch_cam_viewer=True,
        )
        opts.update(kwargs)
        super().__init__(**opts)
        install_librarian_extensions(self.robot)

    # ------------------------------------------------------------------
    # World map seeding
    # ------------------------------------------------------------------
    def _seed_book_at(self, book_id: int, key: str, pose: Pose) -> None:
        spec = {"name": f"Book-{book_id}", "id": book_id, "marker": None}
        b = BookObj(spec, x=pose.x, y=pose.y, z=pose.z, theta=pose.theta)
        b.is_visible = True
        self.robot.world_map.objects[key] = b

    def _seed_landmarks(self) -> None:
        for mid, pose in (
            (17, Pose(_TAG_X - BOX_WIDTH / 2, _TAG_Y, 5, pi / 2)),
            (18, Pose(_TAG_X - BOX_WIDTH / 2, -_TAG_Y, 5, 3 * pi / 2)),
            (19, Pose(_TAG_X, -_TAG_Y + BOX_WIDTH / 2, 5, pi)),
            (20, Pose(_TAG_X, _TAG_Y - BOX_WIDTH / 2, 5, pi)),
        ):
            obj = ArucoMarkerObj(
                {"name": f"ArucoMarker-{mid}", "id": mid, "marker": None},
                x=pose.x, y=pose.y, theta=pose.theta,
            )
            obj.is_fixed = True
            self.robot.world_map.objects[f"ArucoMarker-{mid}.a"] = obj

    def seed_world(self) -> None:
        """Seed corner landmarks plus only books *a* and *b* in the swapped layout.

        Initial layout: book *a* sits in slot_b's pose and book *b* in slot_a's
        pose. No other ``BookObj`` rows are created—only physical books present
        on your field should exist in the map at startup for this demo.
        """
        self._seed_landmarks()
        if not self.seed_initial_layout:
            return
        self._seed_book_at(self.book_id_a, self._key_a, self._pose_slot_b)
        self._seed_book_at(self.book_id_b, self._key_b, self._pose_slot_a)

    def start(self):
        self.seed_world()
        super().start()

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------
    def _robot_goal_for_slot(self, slot_pose: Pose) -> Pose:
        """Robot pose in front of the shelf slot at ``slot_pose``."""
        return Pose(slot_pose.x - self.ROBOT_STANDOFF_MM, slot_pose.y, 0.0, nan)

    # ------------------------------------------------------------------
    # Reusable state-machine fragments
    # ------------------------------------------------------------------
    class _LocalizeSweep(StateNode):
        """Rotate -A deg, +A deg, then back to 0 so rear ArUco corners come into view.

        After each turn we dwell ``settle_s`` seconds so the camera can post
        marker detections and the particle filter can update. Any failed turn
        bubbles up via ``ParentFails`` so the caller's ``=F=>`` fires.
        """

        def __init__(self, half_angle_deg=45.0, settle_s=0.6, turn_speed=None):
            self.half_angle_deg = float(half_angle_deg)
            self.settle_s = float(settle_s)
            self.turn_speed = turn_speed
            super().__init__()

        def setup(self):
            #             announce: Print("swapbooks: localize sweep -- turning left, then right.")
            #             announce =N=> turn_left
            # 
            #             # Turn left A deg to expose the +y (north) rear-corner markers.
            #             turn_left: Turn(self.half_angle_deg, self.turn_speed)
            #             turn_left =C=> settle_left
            #             turn_left =F=> ParentFails()
            # 
            #             settle_left: StateNode() =T(self.settle_s)=> turn_right
            # 
            #             # Turn right 2A deg (so we're at -A relative to start) to expose
            #             # the -y (south) rear-corner markers.
            #             turn_right: Turn(-2.0 * self.half_angle_deg, self.turn_speed)
            #             turn_right =C=> settle_right
            #             turn_right =F=> ParentFails()
            # 
            #             settle_right: StateNode() =T(self.settle_s)=> turn_center
            # 
            #             # Return to the starting heading.
            #             turn_center: Turn(self.half_angle_deg, self.turn_speed)
            #             turn_center =C=> done
            #             turn_center =F=> ParentFails()
            # 
            #             done: ParentCompletes()
            
            # Code generated by genfsm on Thu Apr 23 21:22:20 2026:
            
            announce = Print("swapbooks: localize sweep -- turning left, then right.") .set_name("announce") .set_parent(self)
            turn_left = Turn(self.half_angle_deg, self.turn_speed) .set_name("turn_left") .set_parent(self)
            parentfails1 = ParentFails() .set_name("parentfails1") .set_parent(self)
            settle_left = StateNode() .set_name("settle_left") .set_parent(self)
            turn_right = Turn(-2.0 * self.half_angle_deg, self.turn_speed) .set_name("turn_right") .set_parent(self)
            parentfails2 = ParentFails() .set_name("parentfails2") .set_parent(self)
            settle_right = StateNode() .set_name("settle_right") .set_parent(self)
            turn_center = Turn(self.half_angle_deg, self.turn_speed) .set_name("turn_center") .set_parent(self)
            parentfails3 = ParentFails() .set_name("parentfails3") .set_parent(self)
            done = ParentCompletes() .set_name("done") .set_parent(self)
            
            nulltrans1 = NullTrans() .set_name("nulltrans1")
            nulltrans1 .add_sources(announce) .add_destinations(turn_left)
            
            completiontrans1 = CompletionTrans() .set_name("completiontrans1")
            completiontrans1 .add_sources(turn_left) .add_destinations(settle_left)
            
            failuretrans1 = FailureTrans() .set_name("failuretrans1")
            failuretrans1 .add_sources(turn_left) .add_destinations(parentfails1)
            
            timertrans1 = TimerTrans(self.settle_s) .set_name("timertrans1")
            timertrans1 .add_sources(settle_left) .add_destinations(turn_right)
            
            completiontrans2 = CompletionTrans() .set_name("completiontrans2")
            completiontrans2 .add_sources(turn_right) .add_destinations(settle_right)
            
            failuretrans2 = FailureTrans() .set_name("failuretrans2")
            failuretrans2 .add_sources(turn_right) .add_destinations(parentfails2)
            
            timertrans2 = TimerTrans(self.settle_s) .set_name("timertrans2")
            timertrans2 .add_sources(settle_right) .add_destinations(turn_center)
            
            completiontrans3 = CompletionTrans() .set_name("completiontrans3")
            completiontrans3 .add_sources(turn_center) .add_destinations(done)
            
            failuretrans3 = FailureTrans() .set_name("failuretrans3")
            failuretrans3 .add_sources(turn_center) .add_destinations(parentfails3)
            
            return self

    class _PickAndPlace(StateNode):
        """Drive to a book, magnet-attach, retreat, navigate, release, detach.

        Constructor::

            book_id           Spine ArUco id of the book to pick.
            robot_drop_pose   Robot pose at the dropoff (slot front or staging stand).
            world_drop_pose   World pose where the book ends up (slot center / staging).
            face_pose         Pose to ``TurnTowardPose`` toward before piloting to the
                              dropoff. Defaults to ``robot_drop_pose``.

        Class constants ``RETREAT_MM`` / ``RELEASE_BACK_MM`` / ``POST_DROP_CLEAR_MM``
        / ``BOOK_APPROACH_OFFSET_MM`` / ``ENGAGE_MM`` / ``SETTLE_S`` mirror the
        outer-class tunables; override on a subclass if you need per-leg variations.
        """

        RETREAT_MM = 300.0
        RELEASE_BACK_MM = -38.0
        POST_DROP_CLEAR_MM = -60.0
        BOOK_APPROACH_OFFSET_MM = 0.0
        ENGAGE_MM = 0.0
        SETTLE_S = 0.4

        def __init__(
            self,
            book_id,
            robot_drop_pose,
            world_drop_pose,
            face_pose=None,
            book_approach_offset_mm=None,
            engage_mm=None,
            settle_s=None,
        ):
            self.book_id = int(book_id)
            self.robot_drop_pose = robot_drop_pose
            self.world_drop_pose = world_drop_pose
            self.face_pose = face_pose if face_pose is not None else robot_drop_pose
            if book_approach_offset_mm is not None:
                self.book_approach_offset_mm = float(book_approach_offset_mm)
            else:
                self.book_approach_offset_mm = float(self.BOOK_APPROACH_OFFSET_MM)
            if engage_mm is not None:
                self.engage_mm = float(engage_mm)
            else:
                self.engage_mm = float(self.ENGAGE_MM)
            if settle_s is not None:
                self.settle_s = float(settle_s)
            else:
                self.settle_s = float(self.SETTLE_S)
            super().__init__()

        def setup(self):
            #             # Same order as ``CelesteLibrarian.CmdGetBook``: pilot -> settle -> engage -> attach.
            #             pick: PilotToBook(self.book_id, book_approach_offset_mm=self.book_approach_offset_mm)
            #             pick =C=> settle
            #             pick =F=> ParentFails()
            # 
            #             settle: StateNode() =T(self.settle_s)=> engage
            # 
            #             engage: Forward(self.engage_mm)
            #             engage =C=> attach
            #             engage =F=> ParentFails()
            # 
            #             # Mark the book as held in the world map.
            #             attach: AttachBook(self.book_id)
            #             attach =C=> retreat
            #             attach =F=> ParentFails()
            # 
            #             # Back straight away from the shelf so we have room to pivot.
            #             retreat: Forward(-self.RETREAT_MM)
            #             retreat =C=> face
            #             retreat =F=> ParentFails()
            # 
            #             # Face the dropoff before path-planning to it (so the planner has a
            #             # sensible initial heading and the camera can spot any markers near
            #             # the dropoff for SLAM).
            #             face: TurnTowardPose(self.face_pose)
            #             face =C=> pilot_drop
            #             face =F=> ParentFails()
            # 
            #             # Drive to the dropoff standoff. PilotToPose enforces the final
            #             # heading encoded in robot_drop_pose.
            #             pilot_drop: PilotToPose(self.robot_drop_pose)
            #             pilot_drop =C=> back_for_release
            #             pilot_drop =F=> ParentFails()
            # 
            #             # Optional small back-up before releasing so the detach pose lines
            #             # up with the slot center.
            #             back_for_release: Forward(self.RELEASE_BACK_MM)
            #             back_for_release =C=> release_magnet
            #             back_for_release =F=> ParentFails()
            # 
            #             # Disengage the magnet (kick releases the held book).
            #             release_magnet: Kick()
            #             release_magnet =C=> detach
            #             release_magnet =F=> ParentFails()
            # 
            #             # Update the world map: book now lives at world_drop_pose.
            #             detach: DetachBookAtPose(self.world_drop_pose)
            #             detach =C=> clear
            #             detach =F=> ParentFails()
            # 
            #             # Final reverse nudge so the robot doesn't catch the placed book.
            #             clear: Forward(self.POST_DROP_CLEAR_MM)
            #             clear =C=> done
            #             clear =F=> ParentFails()
            # 
            #             done: ParentCompletes()
            
            # Code generated by genfsm on Thu Apr 23 21:22:20 2026:
            
            pick = PilotToBook(self.book_id, book_approach_offset_mm=self.book_approach_offset_mm) .set_name("pick") .set_parent(self)
            parentfails4 = ParentFails() .set_name("parentfails4") .set_parent(self)
            settle = StateNode() .set_name("settle") .set_parent(self)
            engage = Forward(self.engage_mm) .set_name("engage") .set_parent(self)
            parentfails5 = ParentFails() .set_name("parentfails5") .set_parent(self)
            attach = AttachBook(self.book_id) .set_name("attach") .set_parent(self)
            parentfails6 = ParentFails() .set_name("parentfails6") .set_parent(self)
            retreat = Forward(-self.RETREAT_MM) .set_name("retreat") .set_parent(self)
            parentfails7 = ParentFails() .set_name("parentfails7") .set_parent(self)
            face = TurnTowardPose(self.face_pose) .set_name("face") .set_parent(self)
            parentfails8 = ParentFails() .set_name("parentfails8") .set_parent(self)
            pilot_drop = PilotToPose(self.robot_drop_pose) .set_name("pilot_drop") .set_parent(self)
            parentfails9 = ParentFails() .set_name("parentfails9") .set_parent(self)
            back_for_release = Forward(self.RELEASE_BACK_MM) .set_name("back_for_release") .set_parent(self)
            parentfails10 = ParentFails() .set_name("parentfails10") .set_parent(self)
            release_magnet = Kick() .set_name("release_magnet") .set_parent(self)
            parentfails11 = ParentFails() .set_name("parentfails11") .set_parent(self)
            detach = DetachBookAtPose(self.world_drop_pose) .set_name("detach") .set_parent(self)
            parentfails12 = ParentFails() .set_name("parentfails12") .set_parent(self)
            clear = Forward(self.POST_DROP_CLEAR_MM) .set_name("clear") .set_parent(self)
            parentfails13 = ParentFails() .set_name("parentfails13") .set_parent(self)
            done = ParentCompletes() .set_name("done") .set_parent(self)
            
            completiontrans4 = CompletionTrans() .set_name("completiontrans4")
            completiontrans4 .add_sources(pick) .add_destinations(settle)
            
            failuretrans4 = FailureTrans() .set_name("failuretrans4")
            failuretrans4 .add_sources(pick) .add_destinations(parentfails4)
            
            timertrans3 = TimerTrans(self.settle_s) .set_name("timertrans3")
            timertrans3 .add_sources(settle) .add_destinations(engage)
            
            completiontrans5 = CompletionTrans() .set_name("completiontrans5")
            completiontrans5 .add_sources(engage) .add_destinations(attach)
            
            failuretrans5 = FailureTrans() .set_name("failuretrans5")
            failuretrans5 .add_sources(engage) .add_destinations(parentfails5)
            
            completiontrans6 = CompletionTrans() .set_name("completiontrans6")
            completiontrans6 .add_sources(attach) .add_destinations(retreat)
            
            failuretrans6 = FailureTrans() .set_name("failuretrans6")
            failuretrans6 .add_sources(attach) .add_destinations(parentfails6)
            
            completiontrans7 = CompletionTrans() .set_name("completiontrans7")
            completiontrans7 .add_sources(retreat) .add_destinations(face)
            
            failuretrans7 = FailureTrans() .set_name("failuretrans7")
            failuretrans7 .add_sources(retreat) .add_destinations(parentfails7)
            
            completiontrans8 = CompletionTrans() .set_name("completiontrans8")
            completiontrans8 .add_sources(face) .add_destinations(pilot_drop)
            
            failuretrans8 = FailureTrans() .set_name("failuretrans8")
            failuretrans8 .add_sources(face) .add_destinations(parentfails8)
            
            completiontrans9 = CompletionTrans() .set_name("completiontrans9")
            completiontrans9 .add_sources(pilot_drop) .add_destinations(back_for_release)
            
            failuretrans9 = FailureTrans() .set_name("failuretrans9")
            failuretrans9 .add_sources(pilot_drop) .add_destinations(parentfails9)
            
            completiontrans10 = CompletionTrans() .set_name("completiontrans10")
            completiontrans10 .add_sources(back_for_release) .add_destinations(release_magnet)
            
            failuretrans10 = FailureTrans() .set_name("failuretrans10")
            failuretrans10 .add_sources(back_for_release) .add_destinations(parentfails10)
            
            completiontrans11 = CompletionTrans() .set_name("completiontrans11")
            completiontrans11 .add_sources(release_magnet) .add_destinations(detach)
            
            failuretrans11 = FailureTrans() .set_name("failuretrans11")
            failuretrans11 .add_sources(release_magnet) .add_destinations(parentfails11)
            
            completiontrans12 = CompletionTrans() .set_name("completiontrans12")
            completiontrans12 .add_sources(detach) .add_destinations(clear)
            
            failuretrans12 = FailureTrans() .set_name("failuretrans12")
            failuretrans12 .add_sources(detach) .add_destinations(parentfails12)
            
            completiontrans13 = CompletionTrans() .set_name("completiontrans13")
            completiontrans13 .add_sources(clear) .add_destinations(done)
            
            failuretrans13 = FailureTrans() .set_name("failuretrans13")
            failuretrans13 .add_sources(clear) .add_destinations(parentfails13)
            
            return self

    # ------------------------------------------------------------------
    # Leg factories. Methods (not attributes) so they're evaluated at
    # ``setup()`` time after ``__init__`` has populated the pose attrs.
    # ------------------------------------------------------------------
    def _leg_a_to_staging(self):
        """Pick book *a* from its current (incorrect) slot and park at staging."""
        return self._PickAndPlace(
            self.book_id_a,
            robot_drop_pose=self._pose_staging_robot,
            world_drop_pose=self._pose_staging_drop,
            face_pose=self._pose_staging_robot,
            book_approach_offset_mm=self.BOOK_APPROACH_OFFSET_MM,
            engage_mm=self.ENGAGE_MM,
            settle_s=self.SETTLE_S,
        )

    def _leg_b_to_slot_b(self):
        """Pick book *b* from slot_a (its incorrect spot) and place into slot_b."""
        return self._PickAndPlace(
            self.book_id_b,
            robot_drop_pose=self._robot_goal_for_slot(self._pose_slot_b),
            world_drop_pose=self._pose_slot_b,
            face_pose=self._pose_slot_b,
            book_approach_offset_mm=self.BOOK_APPROACH_OFFSET_MM,
            engage_mm=self.ENGAGE_MM,
            settle_s=self.SETTLE_S,
        )

    def _leg_a_to_slot_a(self):
        """Pick book *a* from staging and place into its canonical slot_a."""
        return self._PickAndPlace(
            self.book_id_a,
            robot_drop_pose=self._robot_goal_for_slot(self._pose_slot_a),
            world_drop_pose=self._pose_slot_a,
            face_pose=self._pose_slot_a,
            book_approach_offset_mm=self.BOOK_APPROACH_OFFSET_MM,
            engage_mm=self.ENGAGE_MM,
            settle_s=self.SETTLE_S,
        )

    # ------------------------------------------------------------------
    # Top-level state graph: localize -> 3 pick/place legs -> success.
    # Any =F=> from any leg or pose-facing turn jumps to a single fail-fast
    # sink that prints diagnostic context and exits.
    # ------------------------------------------------------------------
    def setup(self):
        #         prompt: Print("swapbooks: ready. Type 'start' to localize and run swap.")
        #         prompt =N=> wait
        # 
        #         wait: StateNode()
        #         wait =TM('^\\s*start\\s*$')=> localize
        # 
        #         # ---- Localization preamble ----
        #         localize: self._LocalizeSweep(
        #             half_angle_deg=self.LOCALIZE_HALF_ANGLE_DEG,
        #             settle_s=self.LOCALIZE_SETTLE_S,
        #         )
        #         localize =C=> localize_post_wait
        #         localize =F=> fail
        # 
        #         localize_post_wait: StateNode() =T(self.POST_LOCALIZE_WAIT_S)=> announce_swap
        # 
        #         announce_swap: Print("swapbooks: localized. Beginning 3-leg swap.")
        #         announce_swap =N=> face_b_first
        # 
        #         # ---- Leg 1: book a (currently in slot_b) -> staging ----
        #         # Face slot_b (book a's current location) so its spine marker is in
        #         # the camera before PilotToBook runs.
        #         face_b_first: TurnTowardPose(self._pose_slot_b)
        #         face_b_first =C=> leg1_pick_a
        #         face_b_first =F=> fail
        # 
        #         leg1_pick_a: self._leg_a_to_staging()
        #         leg1_pick_a =C=> face_a_for_b
        #         leg1_pick_a =F=> fail
        # 
        #         # ---- Leg 2: book b (currently in slot_a) -> slot_b ----
        #         face_a_for_b: TurnTowardPose(self._pose_slot_a)
        #         face_a_for_b =C=> leg2_pick_b
        #         face_a_for_b =F=> fail
        # 
        #         leg2_pick_b: self._leg_b_to_slot_b()
        #         leg2_pick_b =C=> back_after_leg2
        #         leg2_pick_b =F=> fail
        # 
        #         back_after_leg2: Forward(-self.POST_PLACE_ROW_CLEAR_MM)
        #         back_after_leg2 =C=> face_staging_for_a
        #         back_after_leg2 =F=> fail
        # 
        #         # ---- Leg 3: book a (at staging) -> slot_a ----
        #         face_staging_for_a: TurnTowardPose(self._pose_staging_drop)
        #         face_staging_for_a =C=> leg3_pick_a
        #         face_staging_for_a =F=> fail
        # 
        #         leg3_pick_a: self._leg_a_to_slot_a()
        #         leg3_pick_a =C=> success
        #         leg3_pick_a =F=> fail
        # 
        #         success: Print("swapbooks: swap complete -- books restored to canonical order.")
        #         success =N=> done
        # 
        #         # ---- Fail-fast sink ----
        #         # Any =F=> above lands here, prints a clear diagnostic, and exits via
        #         # ParentCompletes. No retry / no recovery.
        #         fail: Print("swapbooks: FAILED -- fail-fast exit. Check localization, marker visibility, and magnet engagement.")
        #         fail =N=> done
        # 
        #         done: ParentCompletes()
        
        # Code generated by genfsm on Thu Apr 23 21:22:20 2026:
        
        prompt = Print("swapbooks: ready. Type 'start' to localize and run swap.") .set_name("prompt") .set_parent(self)
        wait = StateNode() .set_name("wait") .set_parent(self)
        localize = self._LocalizeSweep(
            half_angle_deg=self.LOCALIZE_HALF_ANGLE_DEG,
            settle_s=self.LOCALIZE_SETTLE_S,
        ) .set_name("localize") .set_parent(self)
        localize_post_wait = StateNode() .set_name("localize_post_wait") .set_parent(self)
        announce_swap = Print("swapbooks: localized. Beginning 3-leg swap.") .set_name("announce_swap") .set_parent(self)
        face_b_first = TurnTowardPose(self._pose_slot_b) .set_name("face_b_first") .set_parent(self)
        leg1_pick_a = self._leg_a_to_staging() .set_name("leg1_pick_a") .set_parent(self)
        face_a_for_b = TurnTowardPose(self._pose_slot_a) .set_name("face_a_for_b") .set_parent(self)
        leg2_pick_b = self._leg_b_to_slot_b() .set_name("leg2_pick_b") .set_parent(self)
        back_after_leg2 = Forward(-self.POST_PLACE_ROW_CLEAR_MM) .set_name("back_after_leg2") .set_parent(self)
        face_staging_for_a = TurnTowardPose(self._pose_staging_drop) .set_name("face_staging_for_a") .set_parent(self)
        leg3_pick_a = self._leg_a_to_slot_a() .set_name("leg3_pick_a") .set_parent(self)
        success = Print("swapbooks: swap complete -- books restored to canonical order.") .set_name("success") .set_parent(self)
        fail = Print("swapbooks: FAILED -- fail-fast exit. Check localization, marker visibility, and magnet engagement.") .set_name("fail") .set_parent(self)
        done = ParentCompletes() .set_name("done") .set_parent(self)
        
        nulltrans2 = NullTrans() .set_name("nulltrans2")
        nulltrans2 .add_sources(prompt) .add_destinations(wait)
        
        textmsgtrans1 = TextMsgTrans('^\\s*start\\s*$') .set_name("textmsgtrans1")
        textmsgtrans1 .add_sources(wait) .add_destinations(localize)
        
        completiontrans14 = CompletionTrans() .set_name("completiontrans14")
        completiontrans14 .add_sources(localize) .add_destinations(localize_post_wait)
        
        failuretrans14 = FailureTrans() .set_name("failuretrans14")
        failuretrans14 .add_sources(localize) .add_destinations(fail)
        
        timertrans4 = TimerTrans(self.POST_LOCALIZE_WAIT_S) .set_name("timertrans4")
        timertrans4 .add_sources(localize_post_wait) .add_destinations(announce_swap)
        
        nulltrans3 = NullTrans() .set_name("nulltrans3")
        nulltrans3 .add_sources(announce_swap) .add_destinations(face_b_first)
        
        completiontrans15 = CompletionTrans() .set_name("completiontrans15")
        completiontrans15 .add_sources(face_b_first) .add_destinations(leg1_pick_a)
        
        failuretrans15 = FailureTrans() .set_name("failuretrans15")
        failuretrans15 .add_sources(face_b_first) .add_destinations(fail)
        
        completiontrans16 = CompletionTrans() .set_name("completiontrans16")
        completiontrans16 .add_sources(leg1_pick_a) .add_destinations(face_a_for_b)
        
        failuretrans16 = FailureTrans() .set_name("failuretrans16")
        failuretrans16 .add_sources(leg1_pick_a) .add_destinations(fail)
        
        completiontrans17 = CompletionTrans() .set_name("completiontrans17")
        completiontrans17 .add_sources(face_a_for_b) .add_destinations(leg2_pick_b)
        
        failuretrans17 = FailureTrans() .set_name("failuretrans17")
        failuretrans17 .add_sources(face_a_for_b) .add_destinations(fail)
        
        completiontrans18 = CompletionTrans() .set_name("completiontrans18")
        completiontrans18 .add_sources(leg2_pick_b) .add_destinations(back_after_leg2)
        
        failuretrans18 = FailureTrans() .set_name("failuretrans18")
        failuretrans18 .add_sources(leg2_pick_b) .add_destinations(fail)
        
        completiontrans19 = CompletionTrans() .set_name("completiontrans19")
        completiontrans19 .add_sources(back_after_leg2) .add_destinations(face_staging_for_a)
        
        failuretrans19 = FailureTrans() .set_name("failuretrans19")
        failuretrans19 .add_sources(back_after_leg2) .add_destinations(fail)
        
        completiontrans20 = CompletionTrans() .set_name("completiontrans20")
        completiontrans20 .add_sources(face_staging_for_a) .add_destinations(leg3_pick_a)
        
        failuretrans20 = FailureTrans() .set_name("failuretrans20")
        failuretrans20 .add_sources(face_staging_for_a) .add_destinations(fail)
        
        completiontrans21 = CompletionTrans() .set_name("completiontrans21")
        completiontrans21 .add_sources(leg3_pick_a) .add_destinations(success)
        
        failuretrans21 = FailureTrans() .set_name("failuretrans21")
        failuretrans21 .add_sources(leg3_pick_a) .add_destinations(fail)
        
        nulltrans4 = NullTrans() .set_name("nulltrans4")
        nulltrans4 .add_sources(success) .add_destinations(done)
        
        nulltrans5 = NullTrans() .set_name("nulltrans5")
        nulltrans5 .add_sources(fail) .add_destinations(done)
        
        return self

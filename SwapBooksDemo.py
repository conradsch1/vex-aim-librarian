"""
Physically swap two books (magnet + navigation).

Assumes spine ArUco ids ``BOOK_FIRST_ID`` and ``BOOK_FIRST_ID+1``, **localized** SLAM, and
markers visible when each ``PilotToBook`` runs. The robot: drives to each book,
runs ``Forward`` into the shelf to engage the magnet, marks the book as held, backs off,
turns toward the deposit standoff, navigates there, backs off, and releases the book on
the map at the target slot. **Tune** ``ROBOT_STANDOFF_MM``, ``BOOK_APPROACH_OFFSET_MM``,
``SHELF_RETREAT_MM``, ``ENGAGE_MM``, ``RELEASE_BACK_MM``, ``POST_DROP_CLEAR_MM``, and survey
pose for your field and magnet.

Field layout matches ``world_setup/WorldSetup.fsm``. Requires ``install_librarian_extensions``.

From ``simple_cli``::

    runfsm('SwapBooksDemo')

Then type ``start`` or ``tm start``.

Regenerate Python with::

    python3 path/to/vex-aim-tools/genfsm SwapBooksDemo.fsm SwapBooksDemo.py
"""

from __future__ import annotations

from math import nan, pi

from aim_fsm import *
from aim_fsm.particle import ArucoCombinedSensorModel

from aim_librarian import BookObj, BOOK_FIRST_ID, install_librarian_extensions
from aim_librarian.book_manip import AttachBook, DetachBookAtPose
from aim_librarian.pilot_ext import PilotToBook, TurnTowardPose

# Same field constants as ``world_setup/WorldSetup.fsm``.
# The four fixed ArUco tags (17-20) sit on the corner boxes that frame the field,
# and ``_TAG_X``/``_TAG_Y`` are the distances from the field origin to those boxes.
_TAG_Y = 138
_TAG_X = 175
BOX_WIDTH = 35


class SwapBooksDemo(StateMachineProgram):
    """World setup + physical swap: pick A to staging, B to A's slot, A to B's slot.

    The swap takes three legs because the magnet can only hold one book at a time:
    A must be parked somewhere (staging) to free its slot before B can be moved into it,
    and only then can A be moved into B's now-empty slot.
    """

    # How far *north* (along +Y) of book A's original slot the staging pose sits.
    STAGING_OFFSET_Y_MM = 130.0
    # Distance the robot keeps between its drive center and the book spine when
    # stopped in front of a shelf slot.  Must leave room for the magnet to reach.
    ROBOT_STANDOFF_MM = 115.0
    # Extra approach offset fed to PilotToBook so we stop just shy of the spine.
    BOOK_APPROACH_OFFSET_MM = 0 # BookObj.SPINE_THICKNESS_MM / 2 + 5.0
    # How far to back up after attach/detach so we clear the shelf before turning.
    SHELF_RETREAT_MM = 300.0
    # Small reverse nudge after a drop so the robot doesn't catch the book when
    # it pivots away (negative = backwards).
    POST_DROP_CLEAR_MM = -60.0
    # Final forward creep to mate the magnet with the spine once we've piloted in.
    # Zero means PilotToBook already stops us in contact; raise if the magnet misses.
    ENGAGE_MM = 0.0
    # How far to back away from the shelf *before* releasing the book, so the
    # detach pose lines up with the target slot (negative = backwards).
    RELEASE_BACK_MM = -38.0

    def __init__(self, book_id_a: int = BOOK_FIRST_ID, book_id_b: int = BOOK_FIRST_ID + 1, **kwargs):
        self.book_id_a = book_id_a
        self.book_id_b = book_id_b
        # Books sit on the shelf with their center 45 mm inboard of the tag row.
        # ``theta = pi`` means the spine faces the robot's approach direction (-X).
        z_book = BookObj.HEIGHT_MM / 2
        shelf_x = _TAG_X - 45.0
        self._pose0_a = Pose(shelf_x, 15.0, z_book, pi)   # book A's home slot
        self._pose0_b = Pose(shelf_x, -15.0, z_book, pi)  # book B's home slot
        # Temporary parking pose for A while B is being moved; offset in +Y from A's slot.
        self._pose_staging_a = Pose(
            self._pose0_a.x,
            self._pose0_a.y + self.STAGING_OFFSET_Y_MM,
            self._pose0_a.z,
            self._pose0_a.theta,
        )
        # World-map keys (``.a`` suffix = "apparent") used by seed_world / detach.
        self._key_a = f"Book-{self.book_id_a}.a"
        self._key_b = f"Book-{self.book_id_b}.a"

        # Ground-truth poses of the four corner ArUco tags, fed to the particle
        # filter so SLAM can localize the robot the instant it sees any tag.
        landmarks = {
            "ArucoMarker-17": Pose(_TAG_X - BOX_WIDTH / 2, _TAG_Y, 5, 90),
            "ArucoMarker-18": Pose(_TAG_X - BOX_WIDTH / 2, -_TAG_Y, 5, 270),
            "ArucoMarker-19": Pose(_TAG_X, -_TAG_Y + BOX_WIDTH / 2, 5, 180),
            "ArucoMarker-20": Pose(_TAG_X, _TAG_Y - BOX_WIDTH / 2, 5, 180),
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
            speech=False,
            launch_particle_viewer=True,
            launch_path_viewer=True,
            launch_worldmap_viewer=True,
            launch_cam_viewer=True,
        )
        opts.update(kwargs)
        super().__init__(**opts)
        install_librarian_extensions(self.robot)

    def seed_world(self) -> None:
        """Fixed ArUco landmarks and two books at distinct shelf poses."""
        spec_a = {"name": f"Book-{self.book_id_a}", "id": self.book_id_a, "marker": None}
        spec_b = {"name": f"Book-{self.book_id_b}", "id": self.book_id_b, "marker": None}
        ba = BookObj(
            spec_a,
            x=self._pose0_a.x,
            y=self._pose0_a.y,
            z=self._pose0_a.z,
            theta=self._pose0_a.theta,
        )
        bb = BookObj(
            spec_b,
            x=self._pose0_b.x,
            y=self._pose0_b.y,
            z=self._pose0_b.z,
            theta=self._pose0_b.theta,
        )
        ba.is_visible = True
        bb.is_visible = True

        self.robot.world_map.objects[self._key_a] = ba
        self.robot.world_map.objects[self._key_b] = bb

        m17 = ArucoMarkerObj(
            {"name": "ArucoMarker-17", "id": 17, "marker": None},
            x=_TAG_X - BOX_WIDTH / 2,
            y=_TAG_Y,
            theta=pi / 2,
        )
        m18 = ArucoMarkerObj(
            {"name": "ArucoMarker-18", "id": 18, "marker": None},
            x=_TAG_X - BOX_WIDTH / 2,
            y=-_TAG_Y,
            theta=3 * pi / 2,
        )
        m17.is_fixed = True
        m18.is_fixed = True
        self.robot.world_map.objects["ArucoMarker-17.a"] = m17
        self.robot.world_map.objects["ArucoMarker-18.a"] = m18
        m19 = ArucoMarkerObj(
            {"name": "ArucoMarker-19", "id": 19, "marker": None},
            x=_TAG_X,
            y=-_TAG_Y + BOX_WIDTH / 2,
            theta=pi,
        )
        m20 = ArucoMarkerObj(
            {"name": "ArucoMarker-20", "id": 20, "marker": None},
            x=_TAG_X,
            y=_TAG_Y - BOX_WIDTH / 2,
            theta=pi,
        )
        m19.is_fixed = True
        m20.is_fixed = True
        self.robot.world_map.objects["ArucoMarker-19.a"] = m19
        self.robot.world_map.objects["ArucoMarker-20.a"] = m20

    def start(self):
        self.seed_world()
        super().start()

    def _robot_goal_for_book(self, book_pose: Pose) -> Pose:
        """Robot pose in front of the shelf slot ``book_pose``."""
        return Pose(book_pose.x - self.ROBOT_STANDOFF_MM, book_pose.y, 0.0, nan)

    def _pose_survey(self) -> Pose:
        """Pose between shelf slots and back from the row so both spines can be in view."""
        shelf_x = _TAG_X - 45.0
        return Pose(shelf_x - 185.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # State graph
    # ------------------------------------------------------------------
    # Every pick/place leg follows the same 11-step recipe:
    #   PilotToBook  -> Forward(ENGAGE_MM)   -> settle -> AttachBook
    #   Forward(-SHELF_RETREAT_MM) -> TurnTowardPose(target) -> PilotToPose(target)
    #   Forward(RELEASE_BACK_MM) -> DetachBookAtPose -> Kick -> Forward(POST_DROP_CLEAR_MM)
    # Any =F=> (failure) transition from any node jumps to the shared ``fail`` node.
    # ------------------------------------------------------------------
    def setup(self):
        #         # --- Boot: announce readiness and wait for the operator ---
        #         prompt: Print("SwapBooksDemo: LOCALIZED, cameras see spines. Type start — magnet swap runs.")
        #         prompt =N=> wait
        # 
        #         wait: StateNode()
        #         wait =TM('^\\s*start\\s*$')=> p_a
        # 
        #         # ================================================================
        #         # LEG 1: Pick book A from its slot, park it at the staging pose.
        #         # ================================================================
        #         # Drive up to book A and stop a spine-thickness short of contact.
        #         p_a: PilotToBook(self.book_id_a, book_approach_offset_mm=self.BOOK_APPROACH_OFFSET_MM)
        #         p_a =C=> settle_a
        #         p_a =F=> fail
        # 
        #         # Pause so the magnet grabs and the pose stabilizes before we commit.
        #         settle_a: StateNode() =T(0.4)=> att_a
        # 
        #         # Mark book A as "held by robot" in the world map.
        #         att_a: AttachBook(self.book_id_a)
        #         att_a =C=> retreat_stg
        #         att_a =F=> fail
        # 
        #         # Back straight away from the shelf so we have room to pivot.
        #         retreat_stg: Forward(-self.SHELF_RETREAT_MM)
        #         retreat_stg =C=> kick_a
        #         retreat_stg =F=> fail
        # 
        #         # Kick the magnet to release the book.
        #         kick_a: Kick()
        #         kick_a =C=> turn_to_staged
        #         kick_a =F=> fail
        # 
        #         # Face the staged book: 90° right in robot convention (-90°).
        #         turn_to_staged: Turn(-90)
        #         turn_to_staged =C=> p_staged
        #         turn_to_staged =F=> fail
        # 
        #         # ================================================================
        #         # LEG 2: Navigate to the book parked at staging (spine id 10).
        #         # ================================================================
        #         p_staged: PilotToBook(10, book_approach_offset_mm=self.BOOK_APPROACH_OFFSET_MM)
        #         p_staged =C=> back_up2
        #         p_staged =F=> fail
        # 
        #         p_b: PilotToBook(self.book_id_b, book_approach_offset_mm=self.BOOK_APPROACH_OFFSET_MM)
        #         p_b =C=> back_up2
        #         p_b =F=> fail
        # 
        #         back_up2: Forward(-100.0)
        #         back_up2 =C=> turn_right2
        #         back_up2 =F=> fail
        # 
        #         turn_right2: Turn(-90)
        #         turn_right2 =C=> move_right2
        #         turn_right2 =F=> fail
        # 
        #         move_right2: Forward(50.0)
        #         move_right2 =C=> turn_left2
        #         move_right2 =F=> fail
        # 
        #         turn_left2: Turn(90)
        #         turn_left2 =C=> move_forward2
        #         turn_left2 =F=> fail
        # 
        #         move_forward2: Forward(100.0)
        #         move_forward2 =C=> p_b
        #         move_forward2 =F=> fail
        # 
        #         # ================================================================
        #         # LEG 3: Pick book A from staging and drop it into book B's old slot.
        #         # This completes the swap: A is where B was, B is where A was.
        #         # ================================================================
        # 
        #         # --- Shared failure sink: any =F=> above lands here ---
        #         fail: Print("SwapBooksDemo: failed — localize, clear path, ensure markers in view. Tune ENGAGE_MM / standoff.") =N=> done
        # 
        #         done: ParentCompletes()
        
        # Code generated by genfsm on Thu Apr 23 23:14:28 2026:
        
        prompt = Print("SwapBooksDemo: LOCALIZED, cameras see spines. Type start — magnet swap runs.") .set_name("prompt") .set_parent(self)
        wait = StateNode() .set_name("wait") .set_parent(self)
        p_a = PilotToBook(self.book_id_a, book_approach_offset_mm=self.BOOK_APPROACH_OFFSET_MM) .set_name("p_a") .set_parent(self)
        settle_a = StateNode() .set_name("settle_a") .set_parent(self)
        att_a = AttachBook(self.book_id_a) .set_name("att_a") .set_parent(self)
        retreat_stg = Forward(-self.SHELF_RETREAT_MM) .set_name("retreat_stg") .set_parent(self)
        kick_a = Kick() .set_name("kick_a") .set_parent(self)
        turn_to_staged = Turn(-90) .set_name("turn_to_staged") .set_parent(self)
        p_staged = PilotToBook(10, book_approach_offset_mm=self.BOOK_APPROACH_OFFSET_MM) .set_name("p_staged") .set_parent(self)
        p_b = PilotToBook(self.book_id_b, book_approach_offset_mm=self.BOOK_APPROACH_OFFSET_MM) .set_name("p_b") .set_parent(self)
        back_up2 = Forward(-100.0) .set_name("back_up2") .set_parent(self)
        turn_right2 = Turn(-90) .set_name("turn_right2") .set_parent(self)
        move_right2 = Forward(50.0) .set_name("move_right2") .set_parent(self)
        turn_left2 = Turn(90) .set_name("turn_left2") .set_parent(self)
        move_forward2 = Forward(100.0) .set_name("move_forward2") .set_parent(self)
        fail = Print("SwapBooksDemo: failed — localize, clear path, ensure markers in view. Tune ENGAGE_MM / standoff.") .set_name("fail") .set_parent(self)
        done = ParentCompletes() .set_name("done") .set_parent(self)
        
        nulltrans1 = NullTrans() .set_name("nulltrans1")
        nulltrans1 .add_sources(prompt) .add_destinations(wait)
        
        textmsgtrans1 = TextMsgTrans('^\\s*start\\s*$') .set_name("textmsgtrans1")
        textmsgtrans1 .add_sources(wait) .add_destinations(p_a)
        
        completiontrans1 = CompletionTrans() .set_name("completiontrans1")
        completiontrans1 .add_sources(p_a) .add_destinations(settle_a)
        
        failuretrans1 = FailureTrans() .set_name("failuretrans1")
        failuretrans1 .add_sources(p_a) .add_destinations(fail)
        
        timertrans1 = TimerTrans(0.4) .set_name("timertrans1")
        timertrans1 .add_sources(settle_a) .add_destinations(att_a)
        
        completiontrans2 = CompletionTrans() .set_name("completiontrans2")
        completiontrans2 .add_sources(att_a) .add_destinations(retreat_stg)
        
        failuretrans2 = FailureTrans() .set_name("failuretrans2")
        failuretrans2 .add_sources(att_a) .add_destinations(fail)
        
        completiontrans3 = CompletionTrans() .set_name("completiontrans3")
        completiontrans3 .add_sources(retreat_stg) .add_destinations(kick_a)
        
        failuretrans3 = FailureTrans() .set_name("failuretrans3")
        failuretrans3 .add_sources(retreat_stg) .add_destinations(fail)
        
        completiontrans4 = CompletionTrans() .set_name("completiontrans4")
        completiontrans4 .add_sources(kick_a) .add_destinations(turn_to_staged)
        
        failuretrans4 = FailureTrans() .set_name("failuretrans4")
        failuretrans4 .add_sources(kick_a) .add_destinations(fail)
        
        completiontrans5 = CompletionTrans() .set_name("completiontrans5")
        completiontrans5 .add_sources(turn_to_staged) .add_destinations(p_staged)
        
        failuretrans5 = FailureTrans() .set_name("failuretrans5")
        failuretrans5 .add_sources(turn_to_staged) .add_destinations(fail)
        
        completiontrans6 = CompletionTrans() .set_name("completiontrans6")
        completiontrans6 .add_sources(p_staged) .add_destinations(back_up2)
        
        failuretrans6 = FailureTrans() .set_name("failuretrans6")
        failuretrans6 .add_sources(p_staged) .add_destinations(fail)
        
        completiontrans7 = CompletionTrans() .set_name("completiontrans7")
        completiontrans7 .add_sources(p_b) .add_destinations(back_up2)
        
        failuretrans7 = FailureTrans() .set_name("failuretrans7")
        failuretrans7 .add_sources(p_b) .add_destinations(fail)
        
        completiontrans8 = CompletionTrans() .set_name("completiontrans8")
        completiontrans8 .add_sources(back_up2) .add_destinations(turn_right2)
        
        failuretrans8 = FailureTrans() .set_name("failuretrans8")
        failuretrans8 .add_sources(back_up2) .add_destinations(fail)
        
        completiontrans9 = CompletionTrans() .set_name("completiontrans9")
        completiontrans9 .add_sources(turn_right2) .add_destinations(move_right2)
        
        failuretrans9 = FailureTrans() .set_name("failuretrans9")
        failuretrans9 .add_sources(turn_right2) .add_destinations(fail)
        
        completiontrans10 = CompletionTrans() .set_name("completiontrans10")
        completiontrans10 .add_sources(move_right2) .add_destinations(turn_left2)
        
        failuretrans10 = FailureTrans() .set_name("failuretrans10")
        failuretrans10 .add_sources(move_right2) .add_destinations(fail)
        
        completiontrans11 = CompletionTrans() .set_name("completiontrans11")
        completiontrans11 .add_sources(turn_left2) .add_destinations(move_forward2)
        
        failuretrans11 = FailureTrans() .set_name("failuretrans11")
        failuretrans11 .add_sources(turn_left2) .add_destinations(fail)
        
        completiontrans12 = CompletionTrans() .set_name("completiontrans12")
        completiontrans12 .add_sources(move_forward2) .add_destinations(p_b)
        
        failuretrans12 = FailureTrans() .set_name("failuretrans12")
        failuretrans12 .add_sources(move_forward2) .add_destinations(fail)
        
        nulltrans2 = NullTrans() .set_name("nulltrans2")
        nulltrans2 .add_sources(fail) .add_destinations(done)
        
        return self

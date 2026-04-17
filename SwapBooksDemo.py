"""
Physically swap two books (magnet + navigation).

Assumes spine ArUco ids ``BOOK_FIRST_ID`` and ``BOOK_FIRST_ID+1``, **localized** SLAM, and
markers visible when each ``PilotToBook`` runs. The robot: drives to each book,
runs ``Forward`` into the shelf to engage the magnet, marks the book as held, navigates to a
drop pose, backs off, and releases the book on the map at the target slot. **Tune**
``ROBOT_STANDOFF_MM``, ``ENGAGE_MM``, and ``RELEASE_BACK_MM`` for your field and magnet.

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
from aim_librarian.pilot_ext import PilotToBook

# Same field constants as ``world_setup/WorldSetup.fsm``
_TAG_Y = 138
_TAG_X = 175
BOX_WIDTH = 35


class SwapBooksDemo(StateMachineProgram):
    """World setup + physical swap: pick A to staging, B to A's slot, A to B's slot."""

    STAGING_OFFSET_Y_MM = 130.0
    ROBOT_STANDOFF_MM = 115.0
    ENGAGE_MM = 0.0
    RELEASE_BACK_MM = -38.0

    def __init__(self, book_id_a: int = BOOK_FIRST_ID, book_id_b: int = BOOK_FIRST_ID + 1, **kwargs):
        self.book_id_a = book_id_a
        self.book_id_b = book_id_b
        z_book = BookObj.HEIGHT_MM / 2
        shelf_x = _TAG_X - 45.0
        self._pose0_a = Pose(shelf_x, 70.0, z_book, pi)
        self._pose0_b = Pose(shelf_x, -70.0, z_book, pi)
        self._pose_staging_a = Pose(
            self._pose0_a.x,
            self._pose0_a.y + self.STAGING_OFFSET_Y_MM,
            self._pose0_a.z,
            self._pose0_a.theta,
        )
        self._key_a = f"Book-{self.book_id_a}.a"
        self._key_b = f"Book-{self.book_id_b}.a"

        landmarks = {
            "ArucoMarker-17": Pose(_TAG_X - BOX_WIDTH / 2, _TAG_Y, 5, pi / 2),
            "ArucoMarker-18": Pose(_TAG_X - BOX_WIDTH / 2, -_TAG_Y, 5, 3 * pi / 2),
            "ArucoMarker-19": Pose(_TAG_X, -_TAG_Y + BOX_WIDTH / 2, 5, pi),
            "ArucoMarker-20": Pose(_TAG_X, _TAG_Y - BOX_WIDTH / 2, 5, pi),
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

    def setup(self):
        #         prompt: Print("SwapBooksDemo: LOCALIZED, cameras see spines. Type start — magnet swap runs.")
        #         prompt =N=> wait
        # 
        #         wait: StateNode()
        #         wait =TM('^\\s*start\\s*$')=> p_a
        # 
        #         p_a: PilotToBook(self.book_id_a)
        #         p_a =C=> f_a
        #         p_a =F=> fail
        # 
        #         f_a: Forward(self.ENGAGE_MM)
        #         f_a =C=> settle_a
        #         f_a =F=> fail
        # 
        #         settle_a: StateNode() =T(0.4)=> att_a
        # 
        #         att_a: AttachBook(self.book_id_a)
        #         att_a =C=> go_stg
        #         att_a =F=> fail
        # 
        #         go_stg: PilotToPose(self._robot_goal_for_book(self._pose_staging_a))
        #         go_stg =C=> bk_a
        #         go_stg =F=> fail
        # 
        #         bk_a: Forward(self.RELEASE_BACK_MM)
        #         bk_a =C=> drop_stg
        #         bk_a =F=> fail
        # 
        #         drop_stg: DetachBookAtPose(self._pose_staging_a)
        #         drop_stg =C=> p_b
        #         drop_stg =F=> fail
        # 
        #         p_b: PilotToBook(self.book_id_b)
        #         p_b =C=> f_b
        #         p_b =F=> fail
        # 
        #         f_b: Forward(self.ENGAGE_MM)
        #         f_b =C=> settle_b
        #         f_b =F=> fail
        # 
        #         settle_b: StateNode() =T(0.4)=> att_b
        # 
        #         att_b: AttachBook(self.book_id_b)
        #         att_b =C=> go_a
        #         att_b =F=> fail
        # 
        #         go_a: PilotToPose(self._robot_goal_for_book(self._pose0_a))
        #         go_a =C=> bk_b
        #         go_a =F=> fail
        # 
        #         bk_b: Forward(self.RELEASE_BACK_MM)
        #         bk_b =C=> drop_b
        #         bk_b =F=> fail
        # 
        #         drop_b: DetachBookAtPose(self._pose0_a)
        #         drop_b =C=> p_a2
        #         drop_b =F=> fail
        # 
        #         p_a2: PilotToBook(self.book_id_a)
        #         p_a2 =C=> f_a2
        #         p_a2 =F=> fail
        # 
        #         f_a2: Forward(self.ENGAGE_MM)
        #         f_a2 =C=> settle_a2
        #         f_a2 =F=> fail
        # 
        #         settle_a2: StateNode() =T(0.4)=> att_a2
        # 
        #         att_a2: AttachBook(self.book_id_a)
        #         att_a2 =C=> go_b
        #         att_a2 =F=> fail
        # 
        #         go_b: PilotToPose(self._robot_goal_for_book(self._pose0_b))
        #         go_b =C=> bk_a2
        #         go_b =F=> fail
        # 
        #         bk_a2: Forward(self.RELEASE_BACK_MM)
        #         bk_a2 =C=> drop_fin
        #         bk_a2 =F=> fail
        # 
        #         drop_fin: DetachBookAtPose(self._pose0_b)
        #         drop_fin =C=> done
        #         drop_fin =F=> fail
        # 
        #         fail: Print("SwapBooksDemo: failed — localize, clear path, ensure markers in view. Tune ENGAGE_MM / standoff.") =N=> done
        # 
        #         done: ParentCompletes()
        
        # Code generated by genfsm on Fri Apr 17 16:14:59 2026:
        
        prompt = Print("SwapBooksDemo: LOCALIZED, cameras see spines. Type start — magnet swap runs.") .set_name("prompt") .set_parent(self)
        wait = StateNode() .set_name("wait") .set_parent(self)
        p_a = PilotToBook(self.book_id_a) .set_name("p_a") .set_parent(self)
        f_a = Forward(self.ENGAGE_MM) .set_name("f_a") .set_parent(self)
        settle_a = StateNode() .set_name("settle_a") .set_parent(self)
        att_a = AttachBook(self.book_id_a) .set_name("att_a") .set_parent(self)
        go_stg = PilotToPose(self._robot_goal_for_book(self._pose_staging_a)) .set_name("go_stg") .set_parent(self)
        bk_a = Forward(self.RELEASE_BACK_MM) .set_name("bk_a") .set_parent(self)
        drop_stg = DetachBookAtPose(self._pose_staging_a) .set_name("drop_stg") .set_parent(self)
        p_b = PilotToBook(self.book_id_b) .set_name("p_b") .set_parent(self)
        f_b = Forward(self.ENGAGE_MM) .set_name("f_b") .set_parent(self)
        settle_b = StateNode() .set_name("settle_b") .set_parent(self)
        att_b = AttachBook(self.book_id_b) .set_name("att_b") .set_parent(self)
        go_a = PilotToPose(self._robot_goal_for_book(self._pose0_a)) .set_name("go_a") .set_parent(self)
        bk_b = Forward(self.RELEASE_BACK_MM) .set_name("bk_b") .set_parent(self)
        drop_b = DetachBookAtPose(self._pose0_a) .set_name("drop_b") .set_parent(self)
        p_a2 = PilotToBook(self.book_id_a) .set_name("p_a2") .set_parent(self)
        f_a2 = Forward(self.ENGAGE_MM) .set_name("f_a2") .set_parent(self)
        settle_a2 = StateNode() .set_name("settle_a2") .set_parent(self)
        att_a2 = AttachBook(self.book_id_a) .set_name("att_a2") .set_parent(self)
        go_b = PilotToPose(self._robot_goal_for_book(self._pose0_b)) .set_name("go_b") .set_parent(self)
        bk_a2 = Forward(self.RELEASE_BACK_MM) .set_name("bk_a2") .set_parent(self)
        drop_fin = DetachBookAtPose(self._pose0_b) .set_name("drop_fin") .set_parent(self)
        fail = Print("SwapBooksDemo: failed — localize, clear path, ensure markers in view. Tune ENGAGE_MM / standoff.") .set_name("fail") .set_parent(self)
        done = ParentCompletes() .set_name("done") .set_parent(self)
        
        nulltrans1 = NullTrans() .set_name("nulltrans1")
        nulltrans1 .add_sources(prompt) .add_destinations(wait)
        
        textmsgtrans1 = TextMsgTrans('^\\s*start\\s*$') .set_name("textmsgtrans1")
        textmsgtrans1 .add_sources(wait) .add_destinations(p_a)
        
        completiontrans1 = CompletionTrans() .set_name("completiontrans1")
        completiontrans1 .add_sources(p_a) .add_destinations(f_a)
        
        failuretrans1 = FailureTrans() .set_name("failuretrans1")
        failuretrans1 .add_sources(p_a) .add_destinations(fail)
        
        completiontrans2 = CompletionTrans() .set_name("completiontrans2")
        completiontrans2 .add_sources(f_a) .add_destinations(settle_a)
        
        failuretrans2 = FailureTrans() .set_name("failuretrans2")
        failuretrans2 .add_sources(f_a) .add_destinations(fail)
        
        timertrans1 = TimerTrans(0.4) .set_name("timertrans1")
        timertrans1 .add_sources(settle_a) .add_destinations(att_a)
        
        completiontrans3 = CompletionTrans() .set_name("completiontrans3")
        completiontrans3 .add_sources(att_a) .add_destinations(go_stg)
        
        failuretrans3 = FailureTrans() .set_name("failuretrans3")
        failuretrans3 .add_sources(att_a) .add_destinations(fail)
        
        completiontrans4 = CompletionTrans() .set_name("completiontrans4")
        completiontrans4 .add_sources(go_stg) .add_destinations(bk_a)
        
        failuretrans4 = FailureTrans() .set_name("failuretrans4")
        failuretrans4 .add_sources(go_stg) .add_destinations(fail)
        
        completiontrans5 = CompletionTrans() .set_name("completiontrans5")
        completiontrans5 .add_sources(bk_a) .add_destinations(drop_stg)
        
        failuretrans5 = FailureTrans() .set_name("failuretrans5")
        failuretrans5 .add_sources(bk_a) .add_destinations(fail)
        
        completiontrans6 = CompletionTrans() .set_name("completiontrans6")
        completiontrans6 .add_sources(drop_stg) .add_destinations(p_b)
        
        failuretrans6 = FailureTrans() .set_name("failuretrans6")
        failuretrans6 .add_sources(drop_stg) .add_destinations(fail)
        
        completiontrans7 = CompletionTrans() .set_name("completiontrans7")
        completiontrans7 .add_sources(p_b) .add_destinations(f_b)
        
        failuretrans7 = FailureTrans() .set_name("failuretrans7")
        failuretrans7 .add_sources(p_b) .add_destinations(fail)
        
        completiontrans8 = CompletionTrans() .set_name("completiontrans8")
        completiontrans8 .add_sources(f_b) .add_destinations(settle_b)
        
        failuretrans8 = FailureTrans() .set_name("failuretrans8")
        failuretrans8 .add_sources(f_b) .add_destinations(fail)
        
        timertrans2 = TimerTrans(0.4) .set_name("timertrans2")
        timertrans2 .add_sources(settle_b) .add_destinations(att_b)
        
        completiontrans9 = CompletionTrans() .set_name("completiontrans9")
        completiontrans9 .add_sources(att_b) .add_destinations(go_a)
        
        failuretrans9 = FailureTrans() .set_name("failuretrans9")
        failuretrans9 .add_sources(att_b) .add_destinations(fail)
        
        completiontrans10 = CompletionTrans() .set_name("completiontrans10")
        completiontrans10 .add_sources(go_a) .add_destinations(bk_b)
        
        failuretrans10 = FailureTrans() .set_name("failuretrans10")
        failuretrans10 .add_sources(go_a) .add_destinations(fail)
        
        completiontrans11 = CompletionTrans() .set_name("completiontrans11")
        completiontrans11 .add_sources(bk_b) .add_destinations(drop_b)
        
        failuretrans11 = FailureTrans() .set_name("failuretrans11")
        failuretrans11 .add_sources(bk_b) .add_destinations(fail)
        
        completiontrans12 = CompletionTrans() .set_name("completiontrans12")
        completiontrans12 .add_sources(drop_b) .add_destinations(p_a2)
        
        failuretrans12 = FailureTrans() .set_name("failuretrans12")
        failuretrans12 .add_sources(drop_b) .add_destinations(fail)
        
        completiontrans13 = CompletionTrans() .set_name("completiontrans13")
        completiontrans13 .add_sources(p_a2) .add_destinations(f_a2)
        
        failuretrans13 = FailureTrans() .set_name("failuretrans13")
        failuretrans13 .add_sources(p_a2) .add_destinations(fail)
        
        completiontrans14 = CompletionTrans() .set_name("completiontrans14")
        completiontrans14 .add_sources(f_a2) .add_destinations(settle_a2)
        
        failuretrans14 = FailureTrans() .set_name("failuretrans14")
        failuretrans14 .add_sources(f_a2) .add_destinations(fail)
        
        timertrans3 = TimerTrans(0.4) .set_name("timertrans3")
        timertrans3 .add_sources(settle_a2) .add_destinations(att_a2)
        
        completiontrans15 = CompletionTrans() .set_name("completiontrans15")
        completiontrans15 .add_sources(att_a2) .add_destinations(go_b)
        
        failuretrans15 = FailureTrans() .set_name("failuretrans15")
        failuretrans15 .add_sources(att_a2) .add_destinations(fail)
        
        completiontrans16 = CompletionTrans() .set_name("completiontrans16")
        completiontrans16 .add_sources(go_b) .add_destinations(bk_a2)
        
        failuretrans16 = FailureTrans() .set_name("failuretrans16")
        failuretrans16 .add_sources(go_b) .add_destinations(fail)
        
        completiontrans17 = CompletionTrans() .set_name("completiontrans17")
        completiontrans17 .add_sources(bk_a2) .add_destinations(drop_fin)
        
        failuretrans17 = FailureTrans() .set_name("failuretrans17")
        failuretrans17 .add_sources(bk_a2) .add_destinations(fail)
        
        completiontrans18 = CompletionTrans() .set_name("completiontrans18")
        completiontrans18 .add_sources(drop_fin) .add_destinations(done)
        
        failuretrans18 = FailureTrans() .set_name("failuretrans18")
        failuretrans18 .add_sources(drop_fin) .add_destinations(fail)
        
        nulltrans2 = NullTrans() .set_name("nulltrans2")
        nulltrans2 .add_sources(fail) .add_destinations(done)
        
        return self

"""
Idle demo for librarian books on the world map.

- Calls ``install_librarian_extensions`` (``BookObj``, patched RRT, librarian QML).
- Opens the camera and world-map viewers; map updates while the robot is stopped.

From ``simple_cli`` (with ``vex-aim-tools`` and ``vex-aim-librarian`` on ``PYTHONPATH``), run::

    runfsm('BooksIdleDemo')

Stop the program from the CLI or close the FSM as usual.
"""

from aim_fsm import *

from aim_librarian import install_librarian_extensions


class BooksIdleDemo(StateMachineProgram):
    """Single idle node so vision + world_map.update() keep running."""

    def __init__(self, **kwargs):
        opts = dict(
            speech=False,
            launch_path_viewer=False,
            launch_worldmap_viewer=True,
            launch_cam_viewer=True,
        )
        opts.update(kwargs)
        super().__init__(**opts)
        install_librarian_extensions(self.robot)

    def setup(self):
        idle: StateNode()
        return self

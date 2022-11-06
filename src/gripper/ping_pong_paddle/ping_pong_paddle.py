"""
Ping pong paddle to hit a ball
"""
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class PingPongPaddleGripper(GripperModel):
    """
    A Ping Pong Paddle Gripper with no actuation and enabled with sensors to detect contact forces

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        #super().__init__(xml_path_completion("ping_pong_paddle_gripper.xml"), idn=idn)
        super().__init__("gripper/ping_pong_paddle/ping_pong_paddle_gripper.xml", idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return None

    @property
    def _important_geoms(self):
        return {
            "left_finger": [],
            "right_finger": [],
            "left_fingerpad": [],
            "right_fingerpad": [],
        }

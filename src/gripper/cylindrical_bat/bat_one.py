"""
Gripper without fingers to wipe a surface
"""
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class BatOneGripper(GripperModel):
    """
    A BatOne Gripper with no actuation and enabled with sensors to detect contact forces

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        #super().__init__(xml_path_completion("bat_one_gripper.xml"), idn=idn)
        super().__init__("gripper/bat_one_gripper.xml", idn=idn)

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

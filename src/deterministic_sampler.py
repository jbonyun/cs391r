import numpy as np
from robosuite.utils.placement_samplers import ObjectPositionSampler

class DeterministicSampler(ObjectPositionSampler):
    """
    All objects placed exactly where you expect them. No noise.

    Args:
        name (str): Name of this sampler.

        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        ensure_valid_placement (bool): If True, will check for correct (valid) object placements

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
    ):
        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=0,
        )

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).

        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement on top of the reference object. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form

        Raises:
            RandomizationError: [Cannot place all objects]
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(base_offset)

        # Sample pos and quat for all objects assigned to this sampler
        for obj in self.mujoco_objects:
            # First make sure the currently sampled object hasn't already been sampled
            assert obj.name not in placed_objects, "Object '{}' has already been sampled!".format(obj.name)

            bottom_offset = obj.bottom_offset
            success = False
            object_x = base_offset[0]
            object_y = base_offset[1]
            object_z = self.z_offset + base_offset[2]
            if on_top:
                object_z -= bottom_offset[-1]

            # objects cannot overlap
            location_valid = True
            if self.ensure_valid_placement:
                for (x, y, z), _, other_obj in placed_objects.values():
                    if (
                        np.linalg.norm((object_x - x, object_y - y))
                        <= other_obj.horizontal_radius + horizontal_radius
                    ) and (object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]):
                        location_valid = False
                        break

            if location_valid:
                # multiply this quat by the object's initial rotation if it has the attribute specified
                if hasattr(obj, "init_quat"):
                    quat = obj.init_quat
                else:
                    quat = np.array([0, 0, 0, 1])

                # location is valid, put the object down
                pos = (object_x, object_y, object_z)
                placed_objects[obj.name] = (pos, quat, obj)
                success = True
                break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects

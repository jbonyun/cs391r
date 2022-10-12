from robosuite.models.arenas import Arena

class SpaceArena(Arena):
    """A truly empty arena of empty space and no gravity."""

    def __init__(self):
        super().__init__('empty_space.xml')
        # Disable gravity by setting it to zero acceleration in x/y/z
        self.option.attrib['gravity'] = '0 0 0'

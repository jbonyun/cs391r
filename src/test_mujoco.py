import mujoco as mj


def main():

    cam = mj.MjvCamera()
    opt = mj.MjvOption()

    mj.glfw.glfw.init()
    window = mj.glfw.glfw.create_window(1200, 900, "Demo", None, None)
    mj.glfw.glfw.make_context_current(window)
    mj.glfw.glfw.swap_interval(1)

    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)

    xml_path = "example.xml"
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    data.qvel[0] = 1.0

    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    while not mj.glfw.glfw.window_should_close(window):
        print('Frame', 'velocity', data.qvel)
        simstart = data.time

        while (data.time - simstart < 1.0/60.0):
            mj.mj_step(model, data)

        viewport = mj.MjrRect(0, 0, 1200, 900)

        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        mj.glfw.glfw.swap_buffers(window)
        mj.glfw.glfw.poll_events()

    mj.glfw.glfw.terminate()


if __name__ == "__main__":
    main()

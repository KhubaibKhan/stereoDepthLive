import pyrealsense2 as rs
import numpy as np

class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

        depth_sensor = device.query_sensors()[0]
        emitter = depth_sensor.get_option(rs.option.emitter_enabled)
        print("emitter = ", emitter)
        set_emitter = 0
        depth_sensor.set_option(rs.option.emitter_enabled, set_emitter)
        # Start streaming
        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        l_ir = frames.get_infrared_frame(1)
        r_ir = frames.get_infrared_frame(2)

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        lir_image = np.asanyarray(l_ir.get_data())
        rir_image = np.asanyarray(r_ir.get_data())
        
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image, lir_image, rir_image

    def release(self):
        self.pipeline.stop()
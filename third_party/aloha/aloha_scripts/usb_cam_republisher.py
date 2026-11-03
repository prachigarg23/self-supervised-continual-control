#!/usr/bin/env python3
"""
Thin bridge: subscribes to usb_cam_node sensor_msgs/Image topics,
wraps each frame in aloha/RGBGrayscaleImage, publishes on /cam_* topics.

WHY THIS IS NEEDED
------------------
  usb_cam_node publishes:  sensor_msgs/Image  on /usb_cam_*/image_raw
  OpenPI client expects:   RGBGrayscaleImage  on /cam_*

  Different topic names AND different message types — hence the bridge.

COLOR CONVERSION (matches realsense_publisher.py line-for-line)
---------------------------------------------------------------
  usb_cam_node delivers BGR bytes (YUYV→BGR conversion inside the driver).
  realsense_publisher.py does:
      color_frame = np.array(frameset.get_color_frame().get_data())   # BGR
      color_frame = color_frame[..., ::-1]                            # → RGB
      cv_bridge.cv2_to_imgmsg(color_frame, encoding="bgr8")          # RGB bytes, labelled "bgr8"

  This republisher applies the SAME flip so the raw bytes on /cam_* are
  identical to what realsense_publisher would produce.  The ONLY difference
  is that the auto-exposure is now handled by the camera's on-board UVC
  algorithm (same as during 4arms_teleop data collection) instead of the
  Intel RealSense SDK — which is the fix for the overexposed images.

BYTE PATH
---------
  Camera YUYV → usb_cam_node BGR → this node flips → RGB bytes (labelled bgr8)
  → ImageRecorder passthrough → env.py resize → model gets RGB uint8
"""

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from aloha.msg import RGBGrayscaleImage

# usb_cam topic  →  openpi topic
_TOPIC_MAP = {
    "/usb_cam_high/image_raw": "cam_high",
    # "/usb_cam_low/image_raw": "cam_low",
    "/usb_cam_left_wrist/image_raw": "cam_left_wrist",
    "/usb_cam_right_wrist/image_raw": "cam_right_wrist",
}

_bridge = CvBridge()
_publishers = {}


def _on_image(msg, cam_name):
    # 1. Decode — usb_cam already gives BGR
    bgr = _bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # 2. Flip BGR→RGB  (same as realsense_publisher.py line 154)
    rgb = bgr[..., ::-1]

    # 3. Wrap in RGBGrayscaleImage (same encoding label as realsense_publisher.py line 162)
    out = RGBGrayscaleImage()
    out.header.stamp = rospy.Time.now()
    out.images.append(_bridge.cv2_to_imgmsg(rgb, encoding="bgr8"))
    _publishers[cam_name].publish(out)


def main():
    rospy.init_node("usb_cam_republisher")

    for src_topic, cam_name in _TOPIC_MAP.items():
        _publishers[cam_name] = rospy.Publisher(
            cam_name, RGBGrayscaleImage, queue_size=1
        )
        rospy.Subscriber(src_topic, Image, _on_image, callback_args=cam_name)
        rospy.loginfo("Bridging %s -> /%s", src_topic, cam_name)

    rospy.loginfo("usb_cam_republisher running")
    rospy.spin()


if __name__ == "__main__":
    main()

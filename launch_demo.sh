# Autobuntu
# python3 demo.py configs/tusimple.py --test_model weights/tusimple_18.pth --data_root /home/travis/lane_detection/Ultra-Fast-Lane-Detection-Autobuntu --dataset 'ROS' --save_loc '/home/travis/lane_detection/Ultra-Fast-Lane-Detection-Autobuntu' --ros_topic /usb_cam/image_color_rgb --num_lanes 1
python3 detect_and_range.py configs/tusimple.py --test_model weights/tusimple_18.pth --data_root /home/travis/lane_detection/Ultra-Fast-Lane-Detection-Autobuntu --dataset 'ROS' --save_loc '/home/travis/lane_detection/Ultra-Fast-Lane-Detection-Autobuntu' --ros_topic /usb_cam/image_color_rgb --num_lanes 4

# Travis
# python3 demo.py configs/culane.py --test_model weights/culane_18.pth --data_root /home/autobuntu/Documents/Ultra-Fast-Lane-Detection --dataset 'Live' 
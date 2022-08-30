from audioop import avg
import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.dataset import LaneTestDatasetRos
import time
from data.constant import culane_row_anchor, tusimple_row_anchor

from lidar2cam_projection.msg import pixel_ranges
from lidar2cam_projection.msg import pixel_range

import rospy
from sensor_msgs.msg import Image


class pixel_range_sub(object):
    def __init__(self):
        self.pixel_sub  = rospy.Subscriber("/pixel_range_cloud", pixel_ranges, self.get_pixel_range)

    def get_pixel_range(self, data):
        self.pixel_vector =  data

def distance(point1,point2):
    return ((point1[0] - point2[0])**2 + (point1[1]-point2[1])**2)**(1/2)

if __name__ == "__main__":
    

    rospy.init_node('pixel_listener', anonymous=True)
    pixel_range_dat = pixel_range_sub()
    pixel_threshold = 10
    
    

    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()

    rospy.wait_for_message(cfg.ros_topic, Image, timeout=60)

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    elif cfg.dataset == 'Live':
        cls_num_per_lane = 56
    elif cfg.dataset == 'ROS':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if cfg.dataset == 'CULane':
        splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root,split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor

    elif cfg.dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor

    elif cfg.dataset == 'Live':
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root,split),img_transform = img_transforms) for split in splits]
        print((datasets.__getitem__(0)[0][1]))
        loaded_im = cv2.imread(datasets.__getitem__(0)[0][1])
        img_w = loaded_im.shape[1]
        img_h = loaded_im.shape[0]
        row_anchor = tusimple_row_anchor
    
    elif cfg.dataset == 'ROS':
        splits = ['topic.txt']
        datasets = [LaneTestDatasetRos(cfg.ros_topic,img_transform = img_transforms) for split in splits]
        r = rospy.Rate(1)
        r.sleep()

        loaded_im = datasets.__getitem__(0)[0][0]
        img_w = datasets[0].cv_image.shape[1]
        img_h = datasets[0].cv_image.shape[0]
        row_anchor = tusimple_row_anchor
        r = rospy.Rate(1000)

    else:
        raise NotImplementedError


    while not rospy.is_shutdown():

        pixel_info = pixel_range_dat.pixel_vector
        extracted_image = datasets[0].cv_image

        for split, dataset in zip(splits, datasets):
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
            for i, data in enumerate(loader):
                imgs, names = data
                imgs = imgs.cuda()
                with torch.no_grad():
                    out = net(imgs)

                col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
                col_sample_w = col_sample[1] - col_sample[0]

                out_j = out[0].data.cpu().numpy()
                out_j = out_j[:, ::-1, :]
                prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
                idx = np.arange(cfg.griding_num) + 1
                idx = idx.reshape(-1, 1, 1)
                loc = np.sum(prob * idx, axis=0)
                out_j = np.argmax(out_j, axis=0)
                loc[out_j == cfg.griding_num] = 0
                out_j = loc

                line_list = []
                for i in range(out_j.shape[1]):
                    line = []
                    if np.sum(out_j[:, i] != 0) > 2:
                        for k in range(out_j.shape[0]):
                            if out_j[k, i] > 0:
                                ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                                # store_ppp.append(ppp)
                                cv2.circle(extracted_image,ppp,5,(0,255,255),-1)
                                line.append(ppp)
                        line_list.append(line)

                lidar_point_master = []
                for point in pixel_info.points:
                    pixel_u = point.u
                    pixel_v = point.v

                    pixel_r = point.range
                    pixel_x = point.x
                    pixel_y = point.y
                    pixel_z = point.z
                    color = (255,0,0)

                    for line in range(len(line_list)):
                        for point_px in range(len(line_list[line])):
                            line_pix = line_list[line][point_px]
                            distance_px = distance([pixel_u, pixel_v], line_pix)
                            if distance_px < pixel_threshold:
                                lidar_point_master.append([pixel_x, pixel_y, pixel_z, pixel_u, pixel_v, line])
                    cv2.circle(extracted_image,(int(pixel_u), int(pixel_v)), 1, color,-1)

                for line_number in range(len(line_list)):
                    this_line_x  = []
                    this_line_y  = []
                    this_line_z  = []
                    this_line_u  = []
                    this_line_v  = []

                    for matched_point in range(len(lidar_point_master)):
                        extracted_point = lidar_point_master[matched_point]
                        pixel_x =  extracted_point[0]
                        pixel_y =  extracted_point[1]
                        pixel_z =  extracted_point[2]
                        pixel_u =  extracted_point[3]
                        pixel_v =  extracted_point[4]
                        line    =  extracted_point[5]

                        if line == line_number:
                            color = (255,0,255)
                            this_line_x.append(pixel_x)
                            this_line_y.append(pixel_y)
                            this_line_z.append(pixel_z)
                            this_line_u.append(pixel_u)
                            this_line_v.append(pixel_v)

                        cv2.circle(extracted_image,(int(pixel_u), int(pixel_v)), 2, color,-1)

                    line_x_avg = np.average(this_line_x)
                    line_y_avg = np.average(this_line_y)
                    line_z_avg = np.average(this_line_z)
                    line_u_avg = np.average(this_line_u)
                    line_v_avg = np.average(this_line_v)

                    cv2.putText(extracted_image, "LINE "+str(line_number)+" DEV: "+str(round(line_x_avg,2)), (int(line_u_avg+30), int(line_v_avg)), cv2.FONT_HERSHEY_SIMPLEX, 
                                                        1, (255,255,255), 2, cv2.LINE_AA)
   
                vis = cv2.resize(extracted_image, (640,480))
                cv2.imshow('vis',vis)

                if cfg.dataset == 'ROS':
                    r.sleep()
                cv2.waitKey(1)
                
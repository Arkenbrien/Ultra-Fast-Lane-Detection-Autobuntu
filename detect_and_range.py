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
from shapely import geometry

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
    pixel_threshold = 20
    time.sleep(1)

    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()

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

    if cfg.dataset == 'ROS':
        # import rospy 
        # rospy.init_node('ros_ultrafast_lane', anonymous=True)
        splits = ['topic.txt']
        datasets = [LaneTestDatasetRos(cfg.ros_topic,img_transform = img_transforms) for split in splits]
        r = rospy.Rate(1)
        r.sleep()

    while not rospy.is_shutdown():
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
            loaded_im = datasets.__getitem__(0)[0][0]
            img_w = datasets[0].cv_image.shape[1]
            img_h = datasets[0].cv_image.shape[0]
            row_anchor = tusimple_row_anchor

        else:
            raise NotImplementedError

        pixel_info = pixel_range_dat.pixel_vector

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

                # import pdb; pdb.set_trace()
                # vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
                store_ppp = []
                line_list = []
                for i in range(out_j.shape[1]):
                    line = []
                    if np.sum(out_j[:, i] != 0) > 2:
                        for k in range(out_j.shape[0]):
                            if out_j[k, i] > 0:
                                ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                                store_ppp.append(ppp)
                                # cv2.circle(datasets[0].cv_image,ppp,10,(0,0,255),-1)
                                line.append(ppp)
                        line_list.append(line)

                min_list=min(line_list, key=len)
                for point_px in range(len(min_list)):
                    line_1_start = line_list[0][point_px]
                    line_2_start = line_list[1][point_px]

                    center_u = (line_1_start[0]+line_2_start[0])/2
                    center_v = (line_1_start[1]+line_2_start[1])/2

                    cv2.circle(datasets[0].cv_image,(int(line_1_start[0]),int(line_1_start[1])),10,(0,255,255),-1)
                    cv2.circle(datasets[0].cv_image,(int(line_2_start[0]),int(line_2_start[1])),10,(0,255,255),-1)

                    cv2.circle(datasets[0].cv_image,(int(center_u),int(center_v)),5,(255,255,255),-1)

                store_x_lane_master = []
                store_y_lane_master = []
                for point in pixel_info.points:
                    pixel_u = point.u
                    pixel_v = point.v

                    pixel_r = point.range
                    pixel_x = point.x
                    pixel_y = point.y
                    pixel_z = point.z
                    color = (255,0,0)

                    for line in range(len(line_list)):
                        # line_x = []
                        # line_y = []
                        # line_z = []
                        for point_px in range(len(min_list)):
                            line_pix = line_list[line][point_px]
                            distance_px = distance([pixel_u, pixel_v], line_pix)

                            if distance_px < pixel_threshold:
                                # line_x.append(pixel_x)
                                # line_y.append(pixel_y)
                                # line_z.append(pixel_z)
                                # store_u_lane.append(pixel_u)
                                # store_v_lane.append(pixel_v)
                                if line == 0:
                                    color = (255,0,255)
                                if line == 1:
                                    color = (255,255,0)
                            
                        # avg_line_x = np.average(line_x)
                        # avg_line_y = np.average(line_y)
                        # avg_line_z = np.average(line_z)

                        # print("AVERAGE LINE DISTANCE:", str(line), "IS", avg_line_x, avg_line_y)


                    cv2.circle(datasets[0].cv_image ,(int(pixel_u), int(pixel_v)), 1, color,-1)

                # store_x_lane = []
                # store_y_lane = []
                # store_u_lane = []
                # store_v_lane = []
                # for point in pixel_info.points:
                #     pixel_u = point.u
                #     pixel_v = point.v

                #     pixel_r = point.range
                #     pixel_x = point.x
                #     pixel_y = point.y
                #     pixel_z = point.z
                #     color = (255,0,0)

                #     for line_pix in store_ppp:

                #         distance_px = distance([pixel_u, pixel_v], line_pix)
                #         if distance_px < pixel_threshold:
                #             store_x_lane.append(pixel_x)
                #             store_y_lane.append(pixel_y)
                #             store_u_lane.append(pixel_u)
                #             store_v_lane.append(pixel_v)
                #             color = (0,255,0)

                #     cv2.circle(datasets[0].cv_image ,(int(pixel_u), int(pixel_v)), 1, color,-1)

                # avgx = np.average(store_x_lane)
                # avgy = np.average(store_y_lane)

                # avgu = np.average(store_u_lane)
                # avgv = np.average(store_v_lane)
                # print(avgx,avgy)

                # cv2.putText(datasets[0].cv_image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA)
                vis = cv2.resize(datasets[0].cv_image, (640,480))
                cv2.imshow('vis',vis)
                # save_path = cfg.save_loc + '/some_name.jpg'
                # print(save_path)
                # cv2.imwrite(str(save_path),vis)
                r = rospy.Rate(1000)
                r.sleep()
                cv2.waitKey(1)
                
                # vout.write(vis)
            
            # vout.release()
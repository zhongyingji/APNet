import torch
import os
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from PIL import Image
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

class PRWDataset(torch.utils.data.Dataset):



    CLASSES = (
        "__background__ ",
        "person", 
    )

    def __init__(self, ds_dir, mode='train', transforms=None):
        path = os.path.join(ds_dir, 'frame_' + mode + '.mat')
        frame = loadmat(path)
        frame = frame['img_index_' + mode]

        self.transforms = transforms
        self.frame_dir = os.path.join(ds_dir, "frames")
        self.ann_dir = os.path.join(ds_dir, "annotations")

        self.name_shape_map = defaultdict(list)

        self.mode = mode
        self.n_frame = frame.shape[0]
        self.frame = []

        for i in range(self.n_frame):
            self.frame.append(frame[i][0][0] + '.jpg')

        shapefile = os.path.join(ds_dir, mode+'_shape.txt')
        f = open(shapefile)
        for line in f:
            sp = line.split('\t')
            sp[-1] = sp[-1][:-1]
            self.name_shape_map[sp[0]].extend([int(sp[1]), int(sp[2])])

        if self.mode=='train':
            print('Processing id mapping...')
            self._process_idmap()
            print('Processing done.')

        print('Processing keypoint mapping...')
        self.kp_score_map, self.kp_pos_map = self._get_kp_map(ds_dir)
        print('Keypoing mapping done.')

        self.INVIS_THRESH = 0.1 # if the score lower than the threshold, it is invisible
        self.N_KP = 16


    def _get_kp_map(self, ds_dir):
        if self.mode == 'train':
            scores_path = 'scores_new.txt'
            pred_path = 'pred_new.txt'
        elif self.mode == "test":
            scores_path = 'scores_test_new.txt'
            pred_path = 'pred_test_new.txt'

        kp_score_path = os.path.join(ds_dir, scores_path)
        kp_pred_path = os.path.join(ds_dir, pred_path)
        f_score = open(kp_score_path)
        f_pred = open(kp_pred_path)

        score_map = defaultdict(list)
        pred_map = defaultdict(list)

        for line in f_score:
            a = line.split('\t')
            a[-1] = a[-1][:-1]
            tmp = []
            for k in a[1:]:
                tmp.append(float(k))
            score_map[a[0]].append(tmp)

        for line in f_pred:
            a = line.split('\t')
            a[-1] = a[-1][:-1]
            tmp = []
            for k in a[1:]:
                tmp.append(int(k))
            pred_map[a[0]].append(tmp)    
    
        f_score.close()
        f_pred.close()

        for k, v in score_map.items():
            l = len(v)
            if l > 1:
                #print(k)
                
                idx = 0
                conf = 0
                for j in range(l):
                    curr_conf = np.sum(v[j])
                    if curr_conf > conf:
                        conf = curr_conf
                        idx = j
                score_map[k] = [v[idx]]
                pred_map[k] = [pred_map[k][idx]]
        return score_map, pred_map

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.frame_dir, self.frame[idx])

        image = Image.open(img_path)
        
        boxlist = self.get_groundtruth(idx)
        boxlist = boxlist.clip_to_image(remove_empty=True)
        
        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        #boxlist.add_field("ids", idd)
        return image, boxlist, idx

    def __len__(self):
        return self.n_frame

    def get_groundtruth(self, idx):
        bbox_path = os.path.join(self.ann_dir, self.frame[idx])
        bbox_arr = loadmat(bbox_path)
        bbox_arr = bbox_arr[list(bbox_arr.keys())[-1]]
        bbox_raw = bbox_arr[:, 1:].astype(np.float32)
        #bbox_raw = bbox_raw.tolist()
        bbox = []
        kp_loc = []
        kp_score = []

        for j in range(bbox_raw.shape[0]):

            bbox_raw[j, 2] = max(bbox_raw[j, 0], 0)+bbox_raw[j, 2]
            bbox_raw[j, 3] = max(bbox_raw[j, 1], 0)+bbox_raw[j, 3]
            bbox.append(bbox_raw[j, :].tolist())
            # list


            bbox_name = str(int(bbox_arr[j][0]))+'_'+'p'+str(j+1)+'_'+self.frame[idx]
            xmin = max(bbox_raw[j, 0], 0)
            ymin = max(bbox_raw[j, 1], 0)
            if bbox_name not in self.kp_pos_map:
                kp_score.append([-1]*self.N_KP)
                kp_loc.append([-1]*self.N_KP*2)
                # sum score < 0
                # do not regress

            else:
                kp_score.append(self.kp_score_map[bbox_name][0])
                tmp_loc = self.kp_pos_map[bbox_name][0]
                tmp_loc = np.array(tmp_loc)
                tmp_loc[::2] += int(xmin)
                tmp_loc[1::2] += int(ymin)
                kp_loc.append(tmp_loc.tolist())

        kp_loc_ = np.array(kp_loc)
        kp_loc_ = np.reshape(kp_loc_, (bbox_raw.shape[0], self.N_KP, 2))
        kp_score_ = np.array(kp_score)
        nks, kks = np.where(kp_score_<self.INVIS_THRESH)
        kp_score_[kp_score_<self.INVIS_THRESH] = 0
        for nk, kk in zip(nks, kks):
            kp_loc_[nk, kk, :] = 0
        keypoint_bbox = np.concatenate((kp_loc_, kp_score_[:, :, None]), axis=2).tolist()
        
        idd = bbox_arr[:, 0].astype(np.int32)

        label = torch.tensor([1]*idd.shape[0])

        difficult = torch.tensor([False]*idd.shape[0])
        # check if it is 0 or 1
        cam = int(self.frame[idx][1])
        cam = torch.tensor([cam]*idd.shape[0])


        if self.mode == 'train':
            for k in range(idd.shape[0]):
                if idd[k] < 0: 
                    #id[k] = 5555
                    continue
                idd[k] = self.idmap[idd[k]]
        idd = torch.tensor(idd)

        info = self.get_img_info(idx)

        keypoint_bbox = PersonKeypoints(keypoint_bbox, (info["width"], info["height"]))


        boxlist = BoxList(bbox, (info["width"], info["height"]), mode="xyxy")
        boxlist.add_field("labels", label)
        boxlist.add_field("ids", idd)
        boxlist.add_field("cams", cam)
        boxlist.add_field("difficult", difficult)
        boxlist.add_field("keypoints", keypoint_bbox)

        return boxlist

    def get_img_info(self, idx):
        shape = self.name_shape_map[self.frame[idx]]
        return {"height": shape[0], "width": shape[1]}

    def _process_idmap(self):

        self.idmap = defaultdict(int)
        self.idlist = []
        for i in range(self.n_frame):
            bbox_path = os.path.join(self.ann_dir, self.frame[i])
            bbox_arr = loadmat(bbox_path)
            bbox_arr = bbox_arr[list(bbox_arr.keys())[-1]][:, 0].astype(np.int32)

            self.idlist.extend(list(bbox_arr))
        uniq = np.unique(np.array(self.idlist))
        uniq = uniq[uniq >= 0]
        sort_uniq = np.sort(uniq)
        for k in range(sort_uniq.shape[0]):
            self.idmap[sort_uniq[k]] = k

        return

    def map_class_id_to_class_name(self, class_id):
        return PRWDataset.CLASSES[class_id]




class PRWQuery(torch.utils.data.Dataset):
    def __init__(self, ds_dir, transforms=None):
        path = os.path.join(ds_dir, "query_info.txt")
        self.frame_dir = os.path.join(ds_dir, "frames")
      
        self.mapping = defaultdict(list)
        self.frame = []


        f = open(path)
        for line in f:
            z = line.split(' ')
            z[-1] = z[-1][:-1]+'.jpg'
            tmp = []
            tmp.append(int(z[0]))
            for k in z[1:-1]:
                tmp.append(float(k))
            self.mapping[z[-1]].append(tmp)

        f.close()
        for k, v in self.mapping.items():
            self.frame.append(k)

        
        self.transforms = transforms


        self.name_shape_map = defaultdict(list)
        shapefile = os.path.join(ds_dir, 'test_shape.txt')
        f = open(shapefile)
        for line in f:
            sp = line.split('\t')
            sp[-1] = sp[-1][:-1]
            self.name_shape_map[sp[0]].extend([int(sp[1]), int(sp[2])])

        self.g_ann_dir = os.path.join(ds_dir, "annotations")

        print('Processing keypoint mapping...')
        self.kp_score_map, self.kp_pos_map = self._get_kp_map(ds_dir)
        print('Keypoint mapping done.')

        self.N_KP = 16
        self.INVIS_THRESH = 0.1


    def __getitem__(self, idx):
        
        img_path = os.path.join(self.frame_dir, self.frame[idx])

        image = Image.open(img_path)
        
        boxlist = self.get_groundtruth(idx)
        boxlist = boxlist.clip_to_image(remove_empty=True)
        
        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        #boxlist.add_field("ids", idd)
        return image, boxlist, idx

    def __len__(self):
        return len(self.frame)

    def get_groundtruth(self, idx):
        
        _bbox_path = os.path.join(self.g_ann_dir, self.frame[idx])
        _bbox_arr = loadmat(_bbox_path)
        _bbox_arr = _bbox_arr[list(_bbox_arr.keys())[-1]]
        _idlist = _bbox_arr[:, 0].astype(np.int32)        
        
        bbox_arr = self.mapping[self.frame[idx]]
        bbox_arr = np.array(bbox_arr)
        bbox_raw = bbox_arr[:, 1:].astype(np.float32)
        q_idlist = bbox_arr[:, 0].astype(np.int32)

        #bbox_raw = bbox_raw.tolist()
        

        bbox = []
        kp_score = []
        kp_loc = []
        for j in range(bbox_raw.shape[0]):
            bbox_raw[j, 2] = bbox_raw[j, 0]+bbox_raw[j, 2]
            bbox_raw[j, 3] = bbox_raw[j, 1]+bbox_raw[j, 3]
            bbox.append(bbox_raw[j, :].tolist())

        
            _idx = np.where(_idlist==q_idlist[j])[0][0]
            kp_box_name = str(int(bbox_arr[j][0]))+'_'+'p'+str(_idx+1)+'_'+self.frame[idx]
            xmin = max(bbox_raw[j, 0], 0)
            ymin = max(bbox_raw[j, 1], 0)
            
            if kp_box_name not in self.kp_pos_map:
                kp_score.append([-1]*self.N_KP)
                kp_loc.append([-1]*self.N_KP*2)
                # sum score < 0
                # do not regress

            else:
                kp_score.append(self.kp_score_map[kp_box_name][0])
                tmp_loc = self.kp_pos_map[kp_box_name][0]
                tmp_loc = np.array(tmp_loc)
                tmp_loc[::2] += int(xmin)
                tmp_loc[1::2] += int(ymin)
                kp_loc.append(tmp_loc.tolist())

        kp_loc_ = np.array(kp_loc)
        kp_loc_ = np.reshape(kp_loc_, (bbox_raw.shape[0], self.N_KP, 2))
        kp_score_ = np.array(kp_score)
        nks, kks = np.where(kp_score_<self.INVIS_THRESH)
        kp_score_[kp_score_<self.INVIS_THRESH] = 0
        for nk, kk in zip(nks, kks):
            kp_loc_[nk, kk, :] = 0
        keypoint_bbox = np.concatenate((kp_loc_, kp_score_[:, :, None]), axis=2).tolist()
        

        idd = bbox_arr[:, 0].astype(np.int32)

        label = torch.tensor([1]*idd.shape[0])

        difficult = torch.tensor([False]*idd.shape[0])
        # check if it is 0 or 1

        idd = torch.tensor(idd)

        imgname = torch.tensor([idx]*idd.shape[0])
        
        info = self.get_img_info(idx)
        
        cam = int(self.frame[idx][1])
        cam = torch.tensor([cam]*idd.shape[0])

        keypoint_bbox = PersonKeypoints(keypoint_bbox, (info["width"], info["height"]))

        boxlist = BoxList(bbox, (info["width"], info["height"]), mode="xyxy")
        boxlist.add_field("labels", label)
        boxlist.add_field("ids", idd)
        boxlist.add_field("cams", cam)
        boxlist.add_field("difficult", difficult)
        boxlist.add_field("keypoints", keypoint_bbox)
        boxlist.add_field("imgname", imgname)

        return boxlist


    def get_img_info(self, idx):
        shape = self.name_shape_map[self.frame[idx]]
        return {"height": shape[0], "width": shape[1]}


    def _get_kp_map(self, ds_dir):
        kp_score_path = os.path.join(ds_dir, 'scores_test_new.txt')
        kp_pred_path = os.path.join(ds_dir, 'pred_test_new.txt')
        f_score = open(kp_score_path)
        f_pred = open(kp_pred_path)

        score_map = defaultdict(list)
        pred_map = defaultdict(list)

        for line in f_score:
            a = line.split('\t')
            a[-1] = a[-1][:-1]
            tmp = []
            for k in a[1:]:
                tmp.append(float(k))
            score_map[a[0]].append(tmp)

        for line in f_pred:
            a = line.split('\t')
            a[-1] = a[-1][:-1]
            tmp = []
            for k in a[1:]:
                tmp.append(int(k))
            pred_map[a[0]].append(tmp)    
    
        f_score.close()
        f_pred.close()

        for k, v in score_map.items():
            l = len(v)
            if l > 1:
                #print(k)
                
                idx = 0
                conf = 0
                for j in range(l):
                    curr_conf = np.sum(v[j])
                    if curr_conf > conf:
                        conf = curr_conf
                        idx = j
                score_map[k] = [v[idx]]
                pred_map[k] = [pred_map[k][idx]]
        return score_map, pred_map


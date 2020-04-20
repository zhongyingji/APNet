from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from collections import defaultdict
from tqdm import tqdm

def eval_global(dataset, predictions, qdataset, query_predictions, output_folder, logger, query_pad_by_gt=None):
    # predictions
    # results_dict.update({img_id: result for img_id, result in zip(image_ids, output)}


    det_thresh = 0.5

    name_to_det_feat = {}

    FEAT_DIM = predictions[0].get_field("embeds").size(1)

    logger.info(
        "Dimension of Global Feature: " + str(FEAT_DIM)
    )

    print('Processing name_to_det_feat...')
    for image_id, prediction in enumerate(predictions):
        name = dataset.frame[image_id]
        gt_bboxlist = dataset.get_groundtruth(image_id)

        img_info = dataset.get_img_info(image_id)
        width = img_info['width']
        height = img_info['height']

        prediction = prediction.resize((width, height))
        det = np.array(prediction.bbox)
        det_feat = prediction.get_field("embeds")
        det_feat = np.array(det_feat)
        

        pids = np.array(gt_bboxlist.get_field("ids"))

        # CHECK
        #print(prediction.get_field("scores").shape)
        scores = np.array(prediction.get_field("scores"))
        inds = np.where(scores>=det_thresh)[0]

        if len(inds) > 0:
            name_to_det_feat[name] = (det[inds, :], det_feat[inds, :], pids)


    q_feat = []
    q_id = []
    q_imgname = []
    print('FOWARD QUERY...')
    for image_id, qpred in enumerate(query_predictions):

        gt_bboxlist = qdataset.get_groundtruth(image_id)

        qids = qpred.get_field("ids")
        qfeat = qpred.get_field("embeds")

        qimgname = qpred.get_field("imgname")
        
        q_feat.append(qfeat)
        q_id.extend(list(qids))
        q_imgname.extend(list(qimgname))

    q_feat = np.concatenate(q_feat, axis=0)
    q_id = np.array(q_id)
    q_imgname = np.array(q_imgname)


    aps = []
    accs = []
    topk = [1, 5, 10]

    for i in tqdm(range(q_feat.shape[0])):
        # if i == 3: break
        
        y_true, y_score = [], []
        imgs, rois = [], []
        count_gt, count_tp = 0, 0

        feat_p = q_feat[i, :]
        probe_imgname = qdataset.frame[q_imgname[i]]
        
        probe_pid = q_id[i]

        probe_gts = {}

        for image_id in range(len(dataset)):
            gt_bboxlist = dataset.get_groundtruth(image_id)
            name = dataset.frame[image_id]

            gt_ids = gt_bboxlist.get_field("ids")
            if probe_pid in gt_ids and name != probe_imgname:
                loc = np.where(gt_ids==probe_pid)[0]
                probe_gts[name] = np.array(gt_bboxlist.bbox)[loc]


        for image_id in range(len(dataset)):
            gallery_imgname = dataset.frame[image_id]
            if gallery_imgname == probe_imgname:
                continue
            count_gt += (gallery_imgname in probe_gts)

            if gallery_imgname not in name_to_det_feat:
                continue

            det, feat_g, pids_g = name_to_det_feat[gallery_imgname]
            sim = np.dot(feat_g, feat_p).ravel()
            label = np.zeros(len(sim), dtype=np.int32)

            if gallery_imgname in probe_gts:
                gt = probe_gts[gallery_imgname].ravel()
                w, h = gt[2] - gt[0], gt[3] - gt[1]

                iou_thresh = min(0.5, (w*h*1.0)/((w+10)*(h+10)))
                inds = np.argsort(sim)[::-1]
                sim = sim[inds]
                det = det[inds]

                for j, roi in enumerate(det[:, :]):
                    if compute_iou(roi, gt) >= iou_thresh:
                        label[j] = 1
                        count_tp += 1
                        break

            y_true.extend(list(label))
            y_score.extend(list(sim))


        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)
        recall_rate = count_tp*1.0/count_gt
        ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score)*recall_rate
        aps.append(ap)

        inds = np.argsort(y_score)[::-1]
        y_score = y_score[inds]
        y_true = y_true[inds]
        accs.append([min(1, sum(y_true[:k])) for k in topk])

    mAP = np.mean(aps)
    accs_ = np.mean(accs, axis=0)
    
    log_result = 'Result: \nmAP: {:.2%}\n'.format(mAP)
    for i, k in enumerate(topk):
        log_result += ('top-{:2d} = {:.2%}\n'.format(k, accs_[i]))
        # print('top-{:2d} = {:.2%}'.format(k, accs_[i]))
    logger.info(log_result)

    return



def compute_iou(box1, box2):
    # (4, )
    # (xmin, ymin, xmax, ymax)
    w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if w <= 0 or h <= 0:
        return 0
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    cross = w*h
    return cross/(area1+area2-cross)












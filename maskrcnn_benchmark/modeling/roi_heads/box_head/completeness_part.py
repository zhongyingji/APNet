import torch
import torch.nn.functional as F
import numpy as np

from maskrcnn_benchmark.layers.estimator_loss import est_decode
from maskrcnn_benchmark.structures.bounding_box import BoxList


class Ratio(object): 
	def __init__(self, discard_nokp=False):

		self.thrx_idx = 7
		self.rhip_idx = 2
		self.lhip_idx = 3
		self.rknee_idx = 1
		self.lknee_idx = 4
		self.rank_idx = 0
		self.lank_idx = 5

		self.r_head2thrx = 0.24
		self.r_thrx2hip = 0.32
		self.r_hip2knee = 0.22
		self.r_knee2ank = 0.21


		self._rat2 = [1., self.r_thrx2hip/self.r_head2thrx, self.r_hip2knee/self.r_head2thrx, self.r_knee2ank/self.r_head2thrx]
		self._pratio = [self.r_head2thrx, self.r_thrx2hip, self.r_hip2knee, self.r_knee2ank, 0.]
		self._idx = [(self.thrx_idx, ), (self.rhip_idx, self.lhip_idx), (self.rknee_idx, self.lknee_idx), (self.rank_idx, self.lank_idx)]
		
		
		self.INVIS_THRSH = 0.1
		self.discard_nokp = discard_nokp

	def get_ratio_by_est(self, boxlist, is_train):

		return_boxlist = []

		device = boxlist[0].bbox.device 

		for target in boxlist:
			target_bbox = target.bbox
			n = target_bbox.shape[0]
			img_size = target.size
			regvals = target.get_field("reg_vals")
			reg_vals = est_decode(regvals)
			
			new_bbox = []
			for k in range(n):
				p_bbox = target_bbox[k, :]
				h = p_bbox[3]-p_bbox[1]
				new_h = h*(1./(1.-reg_vals[k]))
				p_bbox[3] = p_bbox[1] + new_h
				new_bbox.append(p_bbox.tolist())

			new_bboxlist = BoxList(new_bbox, img_size, mode="xyxy")
			new_bboxlist._copy_extra_fields(target)
			new_bboxlist.add_field("pad_ratio", reg_vals)

			return_boxlist.append(new_bboxlist)

		return_boxlist = [return_box.to(device) for return_box in return_boxlist]

		return return_boxlist


	def get_ratio(self, boxlist, is_train):
		"""

			boxlist: [Bbox, Bbox, ...]

		"""	
		"""
			for those without keypoints:
				global, not partition
		
		"""
		return_boxlist = []
		device = boxlist[0].bbox.device

		for target in boxlist:
			target_bbox = target.bbox
			keypoint = target.get_field("keypoints")
			kp = keypoint.keypoints
			n, _, _ = kp.shape
			bbox = target.bbox
			img_size = target.size

			new_bbox = []
			new_pad = []
			for k in range(n):
				p_kp = kp[k]
				
				pad = 1.0 if is_train else 0.
			
				if p_kp.sum().item() > 0:
					pad = 0.
					for iteration, i in enumerate(self._idx[::-1][:-1]):
						# assume thorax exists
						vis = False
						store_y = None
						for j in i:
							if p_kp[j][2] > self.INVIS_THRSH:
								vis = True

						if vis: 
							store_y = max(p_kp[i[0]][1], p_kp[i[1]][1])
							break



					if not vis:
						# hips, knees, ankles not visible
						pad += sum(self._pratio[2:])
						res = F.relu(target_bbox[k, 3]-p_kp[self.thrx_idx, 1])
						known = F.relu(p_kp[self.thrx_idx, 1]-target_bbox[k, 1])
						tmp = F.relu((self.r_thrx2hip/self.r_head2thrx)*known-res) # pixel
						pad += self.r_thrx2hip*tmp/(tmp+res).item()

						if p_kp[self.thrx_idx, 1].item() == 0:
							pad = 1.0
				
					elif iteration == 0:
						pad = 0.
				
					else:
						pad += sum(self._pratio[::-1][:iteration])
						res = F.relu(target_bbox[k, 3]-store_y)
						known = F.relu(p_kp[self.thrx_idx, 1]-target_bbox[k, 1])
						tmp = F.relu((self._pratio[::-1][iteration]/self.r_head2thrx)*known-res)
						pad += (self._pratio[::-1][iteration]*tmp/(tmp+res)).item()
					
						if p_kp[self.thrx_idx, 1].item() == 0:
							pad = 1.0
							

				p_bbox = 1.*bbox[k, :]
				h = p_bbox[3] - p_bbox[1]
				if pad == 1.0:
					new_h = h
					if not is_train:
						pad = 0.
				else:
					new_h = h*(1./(1.-pad))
				p_bbox[3] = p_bbox[1] + new_h
				new_bbox.append(p_bbox.tolist())
				new_pad.append(pad)



			new_bboxlist = BoxList(new_bbox, img_size, mode="xyxy")
			new_bboxlist._copy_extra_fields(target)
			new_bboxlist.add_field("pad_ratio", torch.tensor(new_pad))
			return_boxlist.append(new_bboxlist)

		return_boxlist = [return_box.to(device) for return_box in return_boxlist]


		return return_boxlist
#coding=utf-8
import torch

def area_of(left_top,right_bottom) ->torch.Tensor:
	"""
	计算单个bbox的面积
	args：
		left-top [N,2]
		right-bottom [N,2]
	return areas [N]
	"""
	hw=torch.clamp(right_bottom-left_top,min=0.0)
	return hw[...,0]*hw[...,1]

def iou_of(boxes0,boxes1,eps=1e-5):
	"""Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlaps_left_top=torch.max(boxes0[...,:2],boxes1[...,:2])
    overlaps_right_bottom=torch.min(boxes0[...,2:],boxes1[...,2:])

    overlaps_area=area_of(overlaps_left_top,overlaps_right_bottom)
    area0=area_of(boxes0[...,:2],boxes0[...,2:])
    area1=area_of(boxes1[...,:2],boxes1[...,2:])
    return overlaps_area/(area0+area1-overlaps_area+eps)

 def hard_nms(box_scores,iou_threshold,top_k=-1,candidate_size=200):
 	"""
    Args:
	    box_scores (N, 5): box的集合，N为框的数量，5即4(位置信息)+1(可能为物体的概率)
	    iou_threshold: 我们用IOU标准去除多余检测框的阈值
	    top_k: 保留多少个计算后留下来的候选框，如果为-1则全保留
	    candidate_size: 参与计算的boxes数量
    Returns:
         picked: 经过nms计算后保留下来的box
 	"""
 	scores=box_scores[:,-1]   #取出所有bbox的scores
 	boxes=box_scores[:,:-1]   #取出所有的bbox的坐标
 	picked=[]

 	_,indexes=scores.sort(descending=True)  #降序排列
 	indexes=indexes[:candidate_size] # 取出前 candidate个

 	while len(indexes)>0:
 		current=indexes[0] # 每次取当前 indexes中概率最大的那个
 		picked.append(current.item())
 		if 0 < top_k ==len(picked) or len(indexes) ==-1:
 			break
 		current_box=boxes[current,:] #得分最高的bbox
 		indexes=indexes[1:]
 		rest_boxes=boxes[indexes,:] #剩下的bbox
 		iou=iou_of(# 计算 最高分数和余下的bbox的 交并比
 				rest_boxes,
 				current_box.unsqueeze(0)
 			)

 		indexes=indexes[iou<=iou_threshold] #保留iou小于阈值的bbox
 	return box_scores[picked,:]


def soft_nms(box_scores,score_threshold,sigma=0.5,top_k=-1):
	"""Soft NMS implementation.

    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores=[]
    while box_scores.size(0)>0:
    	max_score_index=torch.argmax(box_scores[:,4])#取最大的分数的scores
    	cur_box_prob=torch.tensor(box_scores[max_score_index,:])
    	picked_box_scores.append(cur_box_prob)
    	if len(picked_box_scores)==top_k >0 or box_scores.size(0)==1:
    		break
    	cur_box=cur_box_prob[:-1]
    	box_scores[max_score_index,:]=box_scores[-1,:]
    	box_scores=box_scores[:-1,:]
    	ious=iou_of(cur_box.unsqueeze(0),box_scores[:,:-1])

    	box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma) 
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    if len(picked_box_scores) > 0:
        return torch.stack(picked_box_scores)
    else:
        return torch.tensor([])
	
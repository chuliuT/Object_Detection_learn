#coding=utf-8
#
# IOU calculating bounding box intersect of union
# we will calculating between pred boxes and gt boxes

from __future__ import print_function,absolute_import
import numpy as np 

def jaccard_overlaps(pred_boxes,gt_box):
	"""
	input: pred_boxes:[xmin,ymin,xmax,ymax]
		   gt_boxe:[xmin,ymin,xmax,ymax]
	output: iou of two box
	输入 两个 box 来计算 IOU交并比，这里我们参考SSD里面的 jaccard 计算
	选取两个bbox的 左上角的最大值 和 左下角的最小值来计算 相交的部分
	并集的部分按正常的逻辑计算即可， iou = inter/union-inter
	"""
	# first step
	# inter left up point and right bottom point
	left_up_x=max(pred_boxes[0],gt_box[0])
	left_up_y=max(pred_boxes[1],gt_box[1])
	right_bottom_x=min(pred_boxes[2],gt_box[2])
	right_bottom_y=min(pred_boxes[3],gt_box[3])
	# second step
	# intersect area calculate
	intersect=(right_bottom_x-left_up_x)*(right_bottom_y-left_up_y)

	# third step
	# union area calculate
	area_pred=(pred_boxes[2]-pred_boxes[0])*(pred_boxes[3]-pred_boxes[1])
	area_gt=(gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
	union=(area_pred+area_gt-intersect)

	# four step
	# iou= inter /union
	iou=intersect/union
	return iou

def Jaccard_max_iou(pred_boxes,gt_box):
	"""
	input: pred_boxes:[xmin,ymin,xmax,ymax] (multi)
		   gt_boxe:[xmin,ymin,xmax,ymax]
	output: iou of gt box(single) and pred_boxes(multi)
	输入 box 和多个box 来计算 IOU交并比，这里我们参考SSD里面的 jaccard 计算
	选取两个bbox的 左上角的最大值 和 左下角的最小值来计算 相交的部分
	并集的部分按正常的逻辑计算即可， iou = inter/union-inter
	只是换成了矩阵的形式而已，核心思想还是上面的
	"""
	# first step
	# inter left up point and right bottom point
	# 这里 因为是矩阵的计算所以换了 np的函数，如果用 max min会报错
	left_up_x=np.maximum(pred_boxes[:,0],gt_box[0])
	left_up_y=np.maximum(pred_boxes[:,1],gt_box[1])
	right_bottom_x=np.minimum(pred_boxes[:,2],gt_box[2])
	right_bottom_y=np.minimum(pred_boxes[:,3],gt_box[3])
	# second step
	# intersect area calculate
	intersect=(right_bottom_x-left_up_x)*(right_bottom_y-left_up_y)

	# third step
	# union area calculate
	area_pred=(pred_boxes[:,2]-pred_boxes[:,0])*(pred_boxes[:,3]-pred_boxes[:,1])
	area_gt=(gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
	# broadcast
	union=(area_pred+area_gt-intersect)

	# four step
	# iou= inter /union
	iou=intersect/union
	max_iou=np.max(iou)
	index_maxiou=np.argmax(iou)
	return iou,max_iou,index_maxiou

if __name__ == '__main__':
    pred_bbox = np.array([0, 0, 30, 30])   
    gt_bbox = np.array([10, 10, 40, 40])
    print (jaccard_overlaps(pred_bbox, gt_bbox))

    from matplotlib import pyplot as plt 
    import cv2
    plt.figure()
    image=np.ones((60,60,3))*255
    # print(image)
    cv2.rectangle(image,(pred_bbox[0],pred_bbox[1]),(pred_bbox[2],pred_bbox[3]),(0,255,0),1)
    cv2.rectangle(image,(gt_bbox[0],gt_bbox[1]),(gt_bbox[2],gt_bbox[3]),(0,0,255),1)
    plt.imshow(image)
    plt.show()
    # a=[10]*10
    # print(a)
    pred_bbox = np.array([[0, 0, 20, 20],
    [10, 10, 20, 20]])  
    gt_bbox = np.array([10, 10, 20, 20])
    print (Jaccard_max_iou(pred_bbox, gt_bbox))



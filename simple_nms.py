#coding=utf-8
import numpy as np
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
def py_cpu_nms(dets,thresh):
	"""Pure Python NMS baseline."""
	# 坐标和置信度赋值
	x1=dets[:,0]
	y1=dets[:,1]
	x2=dets[:,2]
	y2=dets[:,3]
	scores=dets[:,4]

	#bbox area caculate
	areas=(x2-x1+1)*(y2-y1+1)

	#order descend argsort 默认是升序排列 [::-1] 相当于 [-1:-N:-1]
	order= scores.argsort()[::-1]

	keep=[]
	while order.size>0:
		# 保留最大scores的bbox
		i=order[0]
		keep.append(i)
		#比较 最大scores 和剩下的bbox相交区域的左上角点和右下角点
		xx1=np.maximum(x1[i],x1[order[1:]])
		yy1=np.maximum(y1[i],y1[order[1:]])
		xx2=np.minimum(x2[i],x2[order[1:]])
		yy2=np.minimum(y2[i],y2[order[1:]])
		#计算重叠面积
		w=np.maximum(0.0,xx2-xx1+1)
		h=np.maximum(0.0,yy2-yy1+1)
		inter=w*h
		#计算交并比
		union=areas[i]+areas[order[1:]]-inter
		iou=inter/union
		#保留iou小的值，因为nms 其实就是为了去除重叠的bbox，而这些bbox的iou会很高
		inds=np.where(iou<=thresh)[0]
		# 下标的后移，也就是对剩下的dier高的scores 进行一次nms，依次类推
		order=order[inds+1]
	return keep

# def soft_nms(dets,thresh,sigma,top_k=-1,method=1):
# 	# 坐标和置信度赋值
# 	x1=dets[:,0]
# 	y1=dets[:,1]
# 	x2=dets[:,2]
# 	y2=dets[:,3]
# 	scores=dets[:,4]

# 	#bbox area caculate
# 	areas=(x2-x1+1)*(y2-y1+1)

# 	#order descend argsort 默认是升序排列 [::-1] 相当于 [-1:-N:-1]
# 	order= scores.argsort()[::-1]

# 	keep=[]
# 	while order.size>0:
# 		# 保留最大scores的bbox
# 		i=order[0]
# 		keep.append(i)
		
# 		#比较 最大scores 和剩下的bbox相交区域的左上角点和右下角点
# 		xx1=np.maximum(x1[i],x1[order[1:]])
# 		yy1=np.maximum(y1[i],y1[order[1:]])
# 		xx2=np.minimum(x2[i],x2[order[1:]])
# 		yy2=np.minimum(y2[i],y2[order[1:]])
# 		#计算重叠面积
# 		w=np.maximum(0.0,xx2-xx1+1)
# 		h=np.maximum(0.0,yy2-yy1+1)
# 		inter=w*h
# 		#计算交并比
# 		union=areas[i]+areas[order[1:]]-inter
# 		iou=inter/union
# 		#保留iou小的值，因为nms 其实就是为了去除重叠的bbox，而这些bbox的iou会很高
# 		inds=np.where(iou<=thresh)[0]
# 		# 下标的后移，也就是对剩下的dier高的scores 进行一次nms，依次类推
# 		order=order[inds+1]
# 	return keep

if __name__ == '__main__':
    dets = np.array([[100,120,170,200,0.98],
                     [20,40,80,90,0.99],
                     [20,38,82,88,0.96],
                     [200,380,282,488,0.9],
                     [19,38,75,91, 0.8]])
    print(py_cpu_nms(dets, 0.5))
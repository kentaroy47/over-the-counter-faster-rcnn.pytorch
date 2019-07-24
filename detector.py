# -*- coding: utf-8 -*-
"""
kentaroy47
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
from util import *
from util2 import draw_bboxes

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3
    
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    
def _get_image_blob(im):
  """
  Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  print(im.shape)
  im_orig = im[:,:,:].astype(np.float32, copy=True)
#  im_orig -= cfg.PIXEL_MEANS
# changed to use pytorch models
  im_orig /= 255. # Convert range to [0,1]
  pixel_means = [0.485, 0.456, 0.406]
  im_orig -= pixel_means # Minus mean
  pixel_stdens = [0.229, 0.224, 0.225]
  im_orig /= pixel_stdens # divide by stddev
  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])
  processed_ims = []
  im_scale_factors = []
  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)
  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)
  return blob, np.array(im_scale_factors)

class faster_rcnn(object):
    def __init__(self, net, cfgfile, weightfile, namesfile, use_cuda=True, is_plot=False, is_xywh=False):
     # misc settings
     pascal_classes = np.asarray(['__background__',
                           'aeroplane', 'bicycle', 'bear', 'boat',
                           'bottle', 'bus', 'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse',
                           'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'truck'])
     cfg_from_file(cfgfile)
     set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
     cfg_from_list(set_cfgs)
     cfg.ANCHOR_SCALES = [4, 8, 16, 32]
     cfg.ANCHOR_RATIOS = [0.5,1,2]
#     cfg.TRAIN.SCALES = (int(600),)
#     cfg.TEST.SCALES = (int(600),)
     
     # set network
     class_agnostic = False
     if net == 'vgg16':
        self.net = vgg16(pascal_classes, pretrained=False, class_agnostic=class_agnostic)
     elif net == 'res101':
        self.net = resnet(pascal_classes, 101, pretrained=False, class_agnostic=class_agnostic)
     elif net == 'res50':
        self.net = resnet(pascal_classes, 50, pretrained=False, class_agnostic=class_agnostic)
     elif net == 'res152':
        self.net = resnet(pascal_classes, 152, pretrained=False, class_agnostic=class_agnostic)
     elif net == 'res34':
        self.net = resnet(pascal_classes, 34, pretrained=True, class_agnostic=class_agnostic)
     elif net == 'res18':
        self.net = resnet(pascal_classes, 18, pretrained=False, class_agnostic=class_agnostic)
     
     # set cuda
     self.device = "cuda" if use_cuda else "cpu"
     
     # load the weights
     self.net.create_architecture()
     if self.device == "cuda":
       checkpoint = torch.load(weightfile)
     else:
       checkpoint = torch.load(weightfile, map_location=(lambda storage, loc: storage))
     self.net.load_state_dict(checkpoint['model'])
     if 'pooling_mode' in checkpoint.keys():
       cfg.POOLING_MODE = checkpoint['pooling_mode']
     
     print('Loading weights from %s... Done!' % (weightfile))
     
     print("setting device to: ", self.device)
     self.net.to(self.device)
     self.net.eval()
     
    def __call__(self, ori_img):
      thresh = 0.5
      allbox = []
      
      assert isinstance(ori_img, np.ndarray), "input must be a numpy array!"
      if len(ori_img.shape) == 2:
          ori_img = ori_img[:,:,np.newaxis]
          ori_img = np.concatenate((ori_img,ori_img,ori_img), axis=2)

      blobs, im_scales = _get_image_blob(ori_img)
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

      im_data_pt = torch.from_numpy(im_blob)
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)
      im_info_pt = torch.from_numpy(im_info_np)
      # initilize the tensor holder here.
      im_data = torch.FloatTensor(1)
      im_info = torch.FloatTensor(1)
      num_boxes = torch.LongTensor(1)
      gt_boxes = torch.FloatTensor(1)
    
      # ship to cuda
      if self.device == "cuda":
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
    
      # make variable
      im_data = Variable(im_data, volatile=True)
      im_info = Variable(im_info, volatile=True)
      num_boxes = Variable(num_boxes, volatile=True)
      gt_boxes = Variable(gt_boxes, volatile=True)

      with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()
      
      # infer
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = self.net(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]
      
      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if self.device == "cuda":
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
            box_deltas = box_deltas.view(1, -1, 4 * 21)
          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= im_scales[0]

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()

      for j in xrange(1, 21):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            a = cls_dets.cpu().numpy()
#            print(a)
            allbox.append(a)
      else:
                empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
#                allbox.append(empty_array)
#      try:
      
      allbox = np.asarray(allbox)
      print(allbox)
      print(allbox.shape)
      try:
          xyxy = allbox[0, :, 0:4]
          conf = allbox[0, :, 4]
      except:
          xyxy = None
          conf = None
      return xyxy, conf

if __name__ == '__main__':
    yolo3 = faster_rcnn("res18", 'cfgs/vgg16.yml', "../models/faster_rcnn_hitachi_0718.pth", "aa", is_plot=True)
#    print("yolo3.size =",yolo3.size)
    import os
    root = "../scene4img"
    files = [os.path.join(root,file) for file in os.listdir(root)]
    files.sort()
    for filename in files:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = yolo3(img)
        # save results
#        cv2.imwrite("../results/{}".format(os.path.basename(filename)),res[:,:,(2,1,0)])
        # imshow
        cv2.namedWindow("yolo3", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("yolo3", 600,600)
        ori_im = draw_bboxes(img, np.asarray(res[0, :, 0:4]), None, offset=(0,0))
        cv2.imshow("yolo3", ori_im)
        cv2.waitKey(0)

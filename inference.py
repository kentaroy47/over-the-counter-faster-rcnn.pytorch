# -*- coding: utf-8 -*-
"""
Kentaro Yoshioka 2019/7/19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import cv2
import numpy as np
import argparse
from detector import faster_rcnn
import time
from scipy.misc import imread

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_false')
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  vis = True
  # define network
  fasterRCNN = faster_rcnn("res18", 'cfgs/res18.yml', "./faster_rcnn_500_40_625.pth", "aa", is_plot=True)

  args = parse_args() 
  webcam_num = args.webcam_num
  # Set up webcam or get image directories
  if webcam_num >= 0 :
    cap = cv2.VideoCapture(webcam_num)
    num_images = 0
  else:
    import re
    p = re.compile(r'\d+')
    imglist = sorted(os.listdir(args.image_dir), key=lambda s: int(p.search(s).group()), reverse = True)
    print(imglist)
    num_images = len(imglist)
    num_images2=num_images
  print('Loaded Photo: {} images.'.format(num_images))
  
  
  i = -1
  
  while (num_images > 00):
      i += 1
      total_tic = time.time()
      if webcam_num == -1:
        num_images -= 1

      # Get image from the webcam
      if webcam_num >= 0:
        if not cap.isOpened():
          raise RuntimeError("Webcam could not open. Please check connection.")
        ret, frame = cap.read()
        im_in = np.array(frame)
      # Load the demo image
      else:
        im_file = os.path.join(args.image_dir, imglist[num_images])
        print(im_file)
        im_in = np.array(imread(im_file))
      if len(im_in.shape) == 2:
        im_in = im_in[:,:,np.newaxis]
        im_in = np.concatenate((im_in,im_in,im_in), axis=2)
      im = im_in
      
      # infer network
      im2show, _, _, dets = fasterRCNN(im)
      print(dets)
      if vis and webcam_num == -1:
          # cv2.imshow('test', im2show)
          # cv2.waitKey(0)
          im2show = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
          result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
          cv2.imwrite(result_path, im2show)
      else:
          if vis:
              cv2.imshow("frame", im2show)
          total_toc = time.time()
          total_time = total_toc - total_tic
          frame_rate = 1 / total_time
#          print('Frame rate:', frame_rate)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
  if webcam_num >= 0:
      cap.release()
      cv2.destroyAllWindows()
  

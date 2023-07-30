import cv2
import numpy as np
import scipy
import pickle
from skimage.feature import hog

from ClassicDetector import ClassicDetector

class SVMDetector:
  def __init__(self, img=None, tuning=False):    
    
    self._classic_detector = ClassicDetector()
    
    self._calssic_mapping = {
      0: 'red',
      1: 'yellow',
      2: 'green'
    }
    
    self._colors = {
      'red' : (0,0,255),
      'yellow' : (0,255,255),
      'green' : (0,255,0)
    }
    
    with open('model/svm.p', 'rb') as handle:
        self._svm = pickle.load(handle)
    
    with open('model/mapping.p', 'rb') as handle:
        self._mapping = pickle.load(handle)
    
    self._img = img
    if tuning and img is None:
      raise ValueError('While tuning you also have to set image')
      
  
  def _check_svm(self, slice):
    slice = cv2.resize(slice, (32,96))
    fd = hog(slice, orientations=4, 
             pixels_per_cell=(4, 4), 
             cells_per_block=(2, 2), 
             visualize=False, channel_axis=-1)
    pred = self._svm.predict(fd.reshape(-1,1).T)
    return self._mapping[pred[0]]

  def detect(self, img=None):
    if img is None:
      img = np.copy(self._img)
    else:
      img = np.copy(img)
      
    bboxes, colors, _ = self._classic_detector.detect(img, carla=False)
    
    for bbox, color in zip(bboxes, colors):
      x_cent = bbox[0] + bbox[2] / 2
      y_cent = bbox[1] + bbox[3] / 2
      
      x_min_crop = int(x_cent - bbox[2] / 2 * 1.4)
      x_max_crop = int(x_cent + bbox[2] / 2 * 1.4)
      y_min_crop = int(y_cent - bbox[3] / 2 * 1.4)
      y_max_crop = int(y_cent + bbox[3] / 2 * 1.4)
      
      if x_min_crop < 0:
          x_max_crop -= x_min_crop
          x_min_crop = 0
          
      if x_max_crop > img.shape[1]:
          x_min_crop -= x_max_crop - img.shape[1]
          x_max_crop = img.shape[1]
          
      if y_min_crop < 0:
          y_max_crop -= y_min_crop
          y_min_crop = 0
          
      if y_max_crop > img.shape[0]:
          y_min_crop -= y_max_crop - img.shape[0]
          y_max_crop = img.shape[0]
                
      light = np.copy(img[y_min_crop:y_max_crop,x_min_crop:x_max_crop,:])
      label = self._check_svm(light)
      if self._calssic_mapping[color] == label:
        img = cv2.rectangle(img, (x_min_crop, y_min_crop), (x_max_crop, y_max_crop), self._colors[label], 1)
        cv2.putText(img, f'{label}', (x_min_crop, y_min_crop-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self._colors[label], 2)
      
    return img


if __name__ == '__main__':
  img = cv2.imread('images/000210.png')
  
  tuning = False
  
  detector = SVMDetector(img, tuning)
  if not tuning:
    sign = detector.detect(img=img)
    cv2.imshow('result', sign)
  
  cv2.waitKey(0)
  cv2.destroyAllWindows()
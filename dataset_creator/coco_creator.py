from pycocotools import mask as maskUtils
from datetime import datetime
import os
import cv2
import json
import numpy as np
 

class CocoCreator:
  def __init__(self, path_name=None, classes=None):
    if path_name is None:
      self._dataset_path = datetime.now().strftime('%Y_%m_%d_%H_%M')
    else:
      self._dataset_path = path_name
      
    self._img_path = 'images'
    self._annotations_path = 'annotations'
    
    self._img_path = os.path.join(self._dataset_path, self._img_path)
    self._annotations_path = os.path.join(self._dataset_path, self._annotations_path)
    
    if not os.path.exists(self._dataset_path):
      os.mkdir(self._dataset_path)
        
    if not os.path.exists(self._img_path):
      os.mkdir(self._img_path)
        
    if not os.path.exists(self._annotations_path):
      os.mkdir(self._annotations_path)
        
    
    if classes is None:
      raise ValueError
    else:
      self._classes = classes
      
    self._dataset = {
      'info': self._generate_info(),
      'license': self._generate_license(),
      'categories': self._generate_categories(),
      'images': [],
      'annotations': [],
    }
    
    self._idx = 0
    
  def add_image(self, img, img_id):
    file_name = str(img_id) + '.png'
    definition = self._generate_image_definition(img_id, img, file_name)  
    self._dataset['images'].append(definition)
    cv2.imwrite(os.path.join(self._img_path, file_name), img)
  
  def annonate_image(self, img_id, bbox, category, mask=None):
    category_id = self._classes[category]
    annotation = self._generate_annotation(self._idx, mask, bbox, img_id, category_id)
    self._dataset['annotations'].append(annotation)
    self._idx += 1
    
  def dump_json(self):
    dataset = json.dumps(self._dataset)
    file_name = 'annotation.json'
    with open(os.path.join(self._annotations_path, file_name), 'w') as outfile:
      outfile.write(dataset)
    return dataset
   
  def _generate_info(self):
    info =  {
      'year': datetime.now().strftime('%Y'),
      'version': '0.0',
      'description': 'Carla dataset',
      'contributor': 'Krzysztof Wojciechowski',
      'url': '---',
      'date_created': datetime.now().strftime('%Y-%m-%d')
    }
    return info
  
  
  def _generate_license(self):
    license =  {
      'id': 1,
      'name': 'MIT',
      'url': 'https://opensource.org/licenses/MIT'
    }
    return license
    
    
  def _generate_categories(self):
    categories = []
    for key, value in self._classes.items():
      categories.append({
        'id': value, 
        'name': key, 
        'supercategory': key, 
        'isthing': 1
      })
    return categories
    

  def _generate_annotation(self, idx, mask, bbox, image_id, category_id):
    if mask is not None:
      mask.reshape((mask.shape[0], mask.shape[1], 1))
      mask = mask.astype(np.uint8)
      c_rle = maskUtils.encode(np.asfortranarray(mask)) # Encoding it back to rle (coco format)
      c_rle['counts'] = c_rle['counts'].decode('utf-8') # converting from binary to utf-8
      area = maskUtils.area(c_rle).item() # calculating the area
      bbox = maskUtils.toBbox(c_rle).astype(int).tolist() # calculating the bboxes
    else:
      x1 = bbox[0]
      y1 = bbox[1]
      x2 = bbox[2]
      y2 = bbox[3]
      c_rle = [[x1,y1,x1,(y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]
      area = bbox[2] * bbox[3]
      
    if area < 10:
      raise Exception
    
    annotation = {
      'segmentation': c_rle,
      'bbox': bbox,
      'area': area,
      'image_id': image_id, 
      'category_id': category_id, 
      'iscrowd': 0, 
      'id': idx
    }
    return annotation
    
    
  def _generate_image_definition(self, image_id, img, file_name):
      annotation = {
        'license': 1,
        'file_name': file_name,
        'height': img.shape[0],
        'width': img.shape[1],
        'date_captured': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
        'id': image_id
      }
      return annotation
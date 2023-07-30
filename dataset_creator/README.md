# Install requirements

``` bash
pip3 install -r requirements.txt
```

# Usage

``` python
from coco_creator import CocoCreator

classes = {
  '__background__': 0 # not necessary, pytorch adds it itself, might even be confusing during loading
  'pedestrian': 1,
  'car': 2,
  'traffic_light': 3,
}
    
cc = CocoCreator(classes=classes)
img_cnt = 0

# assuming we have a mask with a single car
# add new annotation to the dataset and tie it with image with number img_cnt
cc.annonate_image(img_cnt, [], 'car', mask=car_mask)

# supposingly we also have mask for the same image with pedestrian
cc.annonate_image(img_cnt, [], 'pedestrian', mask=pedestrian_mask)

# now lets add the image itself and save it
cc.add_image(img, img_cnt)
img_cnt += 1


# for image without segmentation mask
cc.annonate_image(img_cnt, [x_min, y_min, w, h], 'traffic_light')
cc.add_image(img, img_cnt)


# saving the dataset
cc.dump_json()
```

By default if `CocoCreator` won't get parameter `path_name` it will create new folder with name being the date with year, month, day and than hour and minute when the program started. It will create folder structure as follows:
```
generator/
├─ 2023_01_24_13_30/
│  ├─ annotations/
│  │  ├─annotation.json
│  ├─ images/
│  │  ├─ 0.png
│  │  ├─ 1.png
│  │  ├─ 2.png
│  │  ├─ ...
```

In case no classes dictionary will be passed the constructor will raise `ValueError`.

## Segmentation masks

In case segmentation mask can be obtained for the image it is passed in the following way:
``` python
cc.annonate_image(image_id, None, category, mask)
```
Bounding box and area will be computed automatically and segmentation mask will be encoded into **annotation.json**.

## Bounding box only

``` python
cc.annonate_image(image_id, [x_min, y_min, w, h], category)
```

Setting mask to `None` disables it. Bounding box will be converted into COCO format. The area and segmentation mask will be computed automatically.

## Adding the image

In order to add image itself call
``` python
cc.add_image(img, image_id)
```
`image_id` parameter is unique ID given to the couple images with annotation. It's value have to be handled outside of the annotation object.

When called the image will be added to the JSON file and imminently saved with OpenCV so remember to us BGR encoding.

## Not supported

Specifying `supercategory` key.

# Testing

You can run `test_coco.py` to see if bounding boxes and masks are saved properly. This program randomly samples the dataset that... Has to be hardcoded into variable `dataset_date`... This shows all annotations for the image.

This program was not written in a maintanable way so it its requirements were also omitted in `requirements.txt`

# Loading into PyTorch

Following code shows how to setup dataset and dataloader for PyTorch.

``` python
from pycocotools import mask as maskUtils
import multiprocessing

path2data="/2023_01_23_23_00/images/"
path2json="/2023_01_23_23_00/annotations/annotation.json"

  
def collate_fn(batch):
    return tuple(zip(*batch))
  
class COCOBoundingBox:
    def __init__(self) -> None:
        pass

    def __call__(self, annotations):
        x, y = annotations[0]['segmentation']['size']
        masks = np.zeros((len(annotations), x, y))
        boxes = np.zeros((len(annotations), 4))
        
        for i, ann in enumerate(annotations):
            masks[i,:,:] = maskUtils.decode(ann['segmentation'])
            
            xmin = ann['bbox'][0]
            ymin = ann['bbox'][1]
            xmax = xmin + ann['bbox'][2]
            ymax = ymin + ann['bbox'][3]
            boxes[i,:] = np.array([xmin, ymin, xmax, ymax])
            
        
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.int64)

        labels = torch.tensor([int(ann['category_id']) for ann in annotations], dtype=torch.int64)
        img_id = torch.tensor([int(ann['image_id']) for ann in annotations], dtype=torch.int64)
        areas = torch.as_tensor([int(ann['area']) for ann in annotations], dtype=torch.float32)
        iscrowd = torch.as_tensor([int(ann['iscrowd']) for ann in annotations], dtype=torch.float32)
        
        # target in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = img_id
        target["area"] = areas
        target["iscrowd"] = iscrowd
        return target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"



trans = transforms.Compose([
    transforms.ToTensor()
])

coco_full_dataset = datasets.CocoDetection(root = path2data,
                                annFile = path2json,
                                transform=trans,
                                target_transform=COCOBoundingBox())


train_size = int(0.8 * len(coco_full_dataset))
test_size = len(coco_full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(coco_full_dataset, [train_size, test_size])

batch_size = 32

coco_train_dataloader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=multiprocessing.cpu_count(),
                                   collate_fn=collate_fn)

coco_test_dataloader =  DataLoader(test_dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=multiprocessing.cpu_count(),
                                   collate_fn=collate_fn)
```

In order to train:
``` python
device = 'cuda:0'
for images, targets in coco_train_dataloader:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    # ...
    
    with torch.set_grad_enabled(True):
        loss_dict = model(images, targets)
```

This dataset should work with examples shown in [classification references](https://github.com/pytorch/vision/tree/main/references/classification).
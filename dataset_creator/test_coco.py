from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np

dataset_date = '2023_01_23_16_02'
annFile=f'{dataset_date}/annotations/annotation.json'

# Initialize the COCO api for instance annotations
coco=COCO(annFile)

filterClasses = ['pedestrians']

# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds(catNms=None) 
# Get all images containing the above Category IDs
imgIds = coco.getImgIds(catIds=catIds)
print("Number of images containing all the  classes:", len(imgIds))


# load and display a random image
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
I = io.imread(f'{dataset_date}/images/' + img['file_name'])/255.0
plt.axis('off')
plt.imshow(I)

annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)

plt.show()
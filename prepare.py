import  xml.dom.minidom
import cv2
import os

image_path = './data/Images'
annotation_path = './data/Annotation'
root_path = './data/pytorch'

if not os.path.isdir(root_path):
    os.mkdir(root_path)

for name in os.listdir(annotation_path):
    target_annotation = os.path.join(annotation_path,name)
    target_image = os.path.join(image_path,name)
    save_path = os.path.join(root_path, name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for d in os.listdir(target_annotation):
        DOMTree = xml.dom.minidom.parse(os.path.join(target_annotation,d))
        collection = DOMTree.documentElement
        xmin = collection.getElementsByTagName('xmin')
        xmax = collection.getElementsByTagName('xmax')
        ymin = collection.getElementsByTagName('ymin')
        ymax = collection.getElementsByTagName('ymax')
        for i in range(len(xmin)):
            x_min = int(xmin[i].firstChild.data)
            y_min = int(ymin[i].firstChild.data)
            x_max = int(xmax[i].firstChild.data)
            y_max = int(ymax[i].firstChild.data)
            img = cv2.imread(os.path.join(target_image, d+'.jpg'))
            img = img[y_min:y_max, x_min:x_max]
            cv2.imwrite(os.path.join(save_path,d+'_%d.jpg'%i), img)






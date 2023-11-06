import os
import xml.etree.ElementTree as ET

sets=[('2012', 'train'),
    ('2012', 'val'), 
    ('2007', 'train'), 
    ('2007', 'val'), 
    ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", 
        "bottle", "bus", "car", "cat", "chair", 
        "cow", "diningtable", "dog", "horse", 
        "motorbike", "person", "pottedplant", 
        "sheep", "sofa", "train", "tvmonitor"]

# convert from minamx to xywh
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    # normalize
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def parse_xml(src_path, target_path):
    with open(src_path, 'r') as rf, open(target_path, 'w') as wf:
        tree=ET.parse(rf)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            wf.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

for year, image_set in sets:
    base_path = "./VOCdevkit/VOC{}".format(year)
    assert os.path.exists(base_path), "Have to download the VOC%s dataset first."%(year)
    labels_path = os.path.join(base_path, 'labels/')
    if not os.path.exists(labels_path): os.makedirs(labels_path)

    ids_path = os.path.join(base_path, "ImageSets/Main/{}.txt".format(image_set))
    image_ids=open(ids_path).read().strip().split()

    for image_id in image_ids:
        parse_xml(
            src_path=os.path.join(base_path, "Annotations/{}.xml".format(image_id)),
            target_path=os.path.join(labels_path, "{}.txt".format(image_id)))
        with open("./{}.txt".format(image_set), 'a') as f:
            f.write("{}\n".format(image_id))
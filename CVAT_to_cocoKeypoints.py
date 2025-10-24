import ast
import os.path
from xml.dom import minidom


out_dir = 'C:\\Users\\Mahin\\Desktop\\yolo_images\\saved_data'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

file = minidom.parse('C:\\Users\\Mahin\\Desktop\\yolo_images\\annotations.xml')

images = file.getElementsByTagName('image')

for image in images:

    width = int(image.getAttribute('width'))
    height = int(image.getAttribute('height'))
    name = image.getAttribute('name')

    # safely get box elements and skip the image if none exist
    boxes = image.getElementsByTagName('box')
    if not boxes:
        # no bounding box for this image; skip processing
        continue
    bbox = boxes[0]

    # parse bbox attributes safely
    try:
        xtl = int(float(bbox.getAttribute('xtl')))
        ytl = int(float(bbox.getAttribute('ytl')))
        xbr = int(float(bbox.getAttribute('xbr')))
        ybr = int(float(bbox.getAttribute('ybr')))
    except (ValueError, TypeError):
        # malformed bbox attributes; skip this image
        continue

    w = xbr - xtl
    h = ybr - ytl

    # get points elements; skip if none
    elem = image.getElementsByTagName('points')
    if not elem:
        continue

    out_path = os.path.join(out_dir, name[:-4] + '.txt')
    with open(out_path, 'w') as label_file:
        for e in elem:

            # write normalized bbox (class 0) in xywh format
            label_file.write('0 {} {} {} {} '.format(
                (xtl + (w / 2)) / width,
                (ytl + (h / 2)) / height,
                w / width,
                h / height
            ))

            points = e.attributes.get('points')
            if points is None:
                # no points attribute for this element
                label_file.write('\n')
                continue

            points = points.value.split(';')
            points_ = []
            for p in points:
                p = p.split(',')
                if len(p) != 2:
                    continue
                p1, p2 = p
                try:
                    points_.append([int(float(p1)), int(float(p2))])
                except (ValueError, TypeError):
                    continue

            for idx, p in enumerate(points_):
                label_file.write('{} {}'.format(p[0] / width, p[1] / height))
                if idx < len(points_) - 1:
                    label_file.write(' ')
                else:
                    label_file.write('\n')
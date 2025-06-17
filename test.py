# coding=utf-8
import numpy as np
import cv2
import os


image_dir = r""
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')][:5]
label_paths = [f.replace('.jpg', '.txt') for f in image_paths]


def xywhn2xyxy(x, w, h):
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2)  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2)  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2)  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2)  # bottom right y
    return y


def draw_labels(img, labels, w, h):
    for _, x in enumerate(labels):
        class_label = int(x[0])  # class
        cv2.rectangle(img, (int(x[1]), int(x[2])), (int(x[3]), int(x[4])), (0, 255, 0), 2)
        cv2.putText(img, str(class_label), (int(x[1]), int(x[2]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img

images_with_labels = []
for image_path, label_path in zip(image_paths, label_paths):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        lb[:, 1:] = xywhn2xyxy(lb[:, 1:], w, h) 

    img_with_labels = draw_labels(img.copy(), lb, w, h)
    images_with_labels.append((img, img_with_labels))

for i, (img, img_with_labels) in enumerate(images_with_labels):
    cv2.imshow(f'Image {i+1}', img)
    cv2.imshow(f'Image {i+1} with Labels', img_with_labels)

cv2.waitKey(0) 
cv2.destroyAllWindows()

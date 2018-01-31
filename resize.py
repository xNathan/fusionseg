# coding: utf-8
from __future__ import print_function, division
import os
from PIL import Image

resize_to = 380

for root, dirs, files in os.walk('./data'):
    if dirs:
        continue
    for filename in files:
        if not filename.lower().endswith(('.jpg', '.jpeg', '.gif', '.png', '.bmp')):
            continue
        file_path = os.path.join(root, filename)
        im = Image.open(file_path)
        (x, y) = im.size
        max_width = max(x, y)
        if max_width <= resize_to:
            continue
        scale_ratio = resize_to / max_width
        x_, y_ = x * scale_ratio, y * scale_ratio
        im.thumbnail((x_, y_), Image.ANTIALIAS)
        im.save(file_path)
        print(file_path)
        
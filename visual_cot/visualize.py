import argparse
import os.path

import numpy as np
from PIL import Image
import mmcv
import json
from tqdm import tqdm


def draw_box(image, box):
    image = np.array(image)
    image = mmcv.imshow_bboxes(img=image,
                               bboxes=np.array(box).reshape(1, 4),
                               colors=(0, 0, 255),
                               thickness=2,
                               show=False)

    return Image.fromarray(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--json_path', default='', type=str)
    parser.add_argument('--save_path', default='', type=str)
    args = parser.parse_args()

    with open(args.json_path, 'r') as f:
        data = json.load(f)

    for data_sample in tqdm(data):
        image = Image.open(data_sample['image']).convert('RGB')
        box = data_sample['box']
        image = draw_box(image, box)
        image.save(os.path.join(args.save_path, os.path.basename(data_sample['image'])))


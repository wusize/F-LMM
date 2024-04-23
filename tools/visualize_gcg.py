import json
import cv2
import pycocotools.mask as mask_utils
# json_file = 'data/GranDf/results/val/psg_2409246.json'
json_file = 'data/GranDf/llava_next_results/psg_2409242.json'
with open(json_file, 'r') as f:
    data = json.load(f)
print(data.keys())
masks = [mask_utils.decode(rle_mask) for rle_mask in data['pred_masks']]
image_file = 'data/GranDf_HA_images/val_test/psg_2409242.jpg'
image = cv2.imread(image_file)
# mask = masks[4]
mask = masks[2]
image[mask > 0] = 255


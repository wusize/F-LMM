import os
import json
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--split", required=True, help="Evaluation split, options are 'val', 'test'")
    parser.add_argument("--prediction_dir_path", required=True, help="The path where the inference results are stored.")
    parser.add_argument("--gt_dir_path", required=False, default="./data/GranDf/annotations/val_test",
                        help="The path containing GranD-f evaluation annotations.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Set the correct split
    split = args.split
    assert split == "val" or split == "test"  # GCG Evaluation has only val and test splits
    gt_mask_path = f"{args.gt_dir_path}/{split}_gcg_coco_mask_gt.json"
    gt_cap_path = f"{args.gt_dir_path}/{split}_gcg_coco_caption_gt.json"

    print(f"Starting evalution on {split} split.")

    # Get the image names of the split
    all_images_ids = []
    with open(gt_cap_path, 'r') as f:
        contents = json.load(f)
        for image in contents['images']:
            all_images_ids.append(image['id'])

    # The directory is used to store intermediate files
    tmp_dir_path = f"tmp/{os.path.basename(args.prediction_dir_path)}_{split}"
    os.makedirs(tmp_dir_path, exist_ok=True)  # Create directory if not exists already

    # Create predictions
    pred_save_path = f"{tmp_dir_path}/mask_pred_tmp_save.json"
    cap_pred_save_path = f"{tmp_dir_path}/cap_pred_tmp_save.json"
    coco_pred_file = []
    caption_pred_dict = {}
    for image_id in all_images_ids:
        prediction_path = f"{args.prediction_dir_path}/{image_id}.json"
        with open(prediction_path, 'r') as f:
            pred = json.load(f)
            bu = pred
            key = list(pred.keys())[0]
            pred = pred[key]
            try:
                caption_pred_dict[image_id] = {'caption': pred['caption'], 'labels': pred['phrases']}
            except Exception as e:
                pred = bu
                caption_pred_dict[image_id] = {'caption': pred['caption'], 'labels': pred['phrases']}
            for rle_mask in pred['pred_masks']:
                coco_pred_file.append({"image_id": image_id, "category_id": 1, "segmentation": rle_mask, "score": 1.0})

    # Save gcg_coco_predictions
    with open(pred_save_path, 'w') as f:
        json.dump(coco_pred_file, f)

    # Prepare the CAPTION predictions in COCO format
    cap_image_ids = []
    coco_cap_pred_file = []
    for image_id, values in caption_pred_dict.items():
        cap_image_ids.append(image_id)
        coco_cap_pred_file.append({"image_id": image_id, "caption": values['caption'], "labels": values['labels']})

    # Save gcg_caption_coco_predictions
    with open(cap_pred_save_path, 'w') as f:
        json.dump(coco_cap_pred_file, f)


    # # -------------------------------#
    # # 2. Evaluate Caption Quality
    coco_cap_gt = COCO(gt_cap_path)
    coco_cap_result = coco_cap_gt.loadRes(cap_pred_save_path)
    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco_cap_gt, coco_cap_result)
    coco_eval.params['image_id'] = coco_cap_result.getImgIds()
    coco_eval.evaluate()
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')


if __name__ == "__main__":
    main()

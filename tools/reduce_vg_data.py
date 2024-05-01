import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('json_path', default='', type=str)
    args = parser.parse_args()

    with open(args.json_path, 'r') as f:
        data = json.load(f)

    vg_data = [data_sample for data_sample in data
               if 'vg/VG_100K' in data_sample['image']]
    other_data = [data_sample for data_sample in data
                  if 'vg/VG_100K' not in data_sample['image']]

    with open(args.json_path.replace('.json', '-vg.json'), 'w') as f:
        json.dump(other_data, f)

    for x in [2, 5, 10, 50, 100]:
        with open(args.json_path.replace('.json', f'-{1/x}vg.json'), 'w') as f:
            json.dump(vg_data[::x] + other_data, f)

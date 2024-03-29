import torch
import argparse
from xtuner.model.utils import guess_load_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--deepspeed_path", default='', type=str)
    parser.add_argument("--torch_path", default='', type=str)

    args = parser.parse_args()
    state_dict = guess_load_checkpoint(args.deepspeed_path)
    torch.save(state_dict, args.torch_path)

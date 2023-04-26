import argparse
import os
import torch
from safetensors.torch import save_file

parser = argparse.ArgumentParser(
    description='Convert a RWKV pytorch checkpoint to a .safetensors file with the same name')
parser.add_argument('src_file', help='Source file')
args = parser.parse_args()

print('Loading checkpoint from %s' % args.src_file)
w = torch.load(args.src_file, map_location='cpu')

# from https://github.com/BlinkDL/ChatRWKV/blob/61c74030d5aee27675a80eb3b2648973b325a9f7/RWKV_in_150_lines.py#L33
for k in w.keys():
    if '.time_' in k:
        w[k] = w[k].squeeze()
    if '.time_decay' in k:
        w[k] = -torch.exp(w[k].float())  # the real time decay is like e^{-e^x}
    else:
        w[k] = w[k].float()  # convert to f32 type

    print(f'{k}, shape {w[k].shape}, type {w[k].dtype}')

file, _ = os.path.splitext(args.src_file)
file = file + '.safetensors'

print('Saving to %s' % file)
save_file(w, file)

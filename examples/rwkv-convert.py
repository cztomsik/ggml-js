import argparse
import os
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

parser = argparse.ArgumentParser(
    description='Convert a RWKV pytorch checkpoint to a .safetensors file with the same name')
parser.add_argument('src_file', help='Source file')
parser.add_argument('--mtype', default='f32',
                    help='Target data type for all matrices (f32 or f16)')
args = parser.parse_args()
mtype = dict(f32=torch.float32, f16=torch.float16)[args.mtype]

print('Loading checkpoint from %s' % args.src_file)
w = torch.load(args.src_file, map_location='cpu')

# from https://github.com/BlinkDL/ChatRWKV/blob/61c74030d5aee27675a80eb3b2648973b325a9f7/RWKV_in_150_lines.py#L33
for k in w.keys():
    if '.time_' in k:
        w[k] = w[k].squeeze()
    if '.time_decay' in k:
        # the real time decay is like e^{-e^x}
        w[k] = -torch.exp(w[k].float())

# precompute LN0 for embeddings
w['emb.weight'] = F.layer_norm(
    w['emb.weight'], (w['emb.weight'].shape[1],), w['blocks.0.ln0.weight'], w['blocks.0.ln0.bias'])
del w['blocks.0.ln0.weight']
del w['blocks.0.ln0.bias']

# convert tensors to target dtype
for k in w.keys():
    if len(w[k].shape) > 1:
        w[k] = w[k].to(mtype)
    else:
        w[k] = w[k].float()

# debug print
for k in w.keys():
    print(f'{k}, shape {w[k].shape}, type {w[k].dtype}')

file, _ = os.path.splitext(args.src_file)
file = file + '.safetensors'

print('Saving to %s' % file)
save_file(w, file)

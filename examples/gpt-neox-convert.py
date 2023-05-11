import argparse
import os
import json
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

parser = argparse.ArgumentParser(
    description='Convert a GPT-NeoX pytorch checkpoint to a .safetensors file with the same name')
parser.add_argument('src_file', help='Source file')
parser.add_argument('--mtype', default='f32',
                    help='Target data type for all matrices (f32 or f16)')
args = parser.parse_args()
mtype = dict(f32=torch.float32, f16=torch.float16)[args.mtype]

print('Loading checkpoint from %s' % args.src_file)
w = torch.load(args.src_file, map_location='cpu')

# strip prefix
w = {k.replace('gpt_neox.', ''): v for k, v in w.items()}

# prune unused weights
for k in list(w.keys()):
    if k.endswith('.masked_bias') or k.endswith('attention.bias') or k.endswith('.inv_freq'):
        del w[k]

# convert tensors to target dtype
for k in w.keys():
    if len(w[k].shape) > 1:
        w[k] = w[k].to(mtype)
    else:
        w[k] = w[k].float()

# debug print
for k in w.keys():
    print(f'{k}, shape {w[k].shape}, type {w[k].dtype}')

metadata = {
    'hparams': json.dumps({
        'vocab_size': w['embed_in.weight'].shape[0],
        'embed_dim': w['embed_in.weight'].shape[1],
        'num_layers': sum(x.endswith('.input_layernorm.weight') for x in w.keys()),
        # TODO
        'num_heads': 8,
        'num_rot': 2,
    })
}
print(metadata)

file, _ = os.path.splitext(args.src_file)
file = file + '.safetensors'

print('Saving to %s' % file)
save_file(w, file, metadata=metadata)

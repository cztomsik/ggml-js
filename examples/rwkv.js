// RWKV-LM Example Usage
// 1. Download the model and tokenizer files from the links below:
//    - https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v4neo/20B_tokenizer.json
//    - https://huggingface.co/BlinkDL/rwkv-4-pile-169m/blob/main/RWKV-4-Pile-169M-20220807-8023.pth
// 2. Run `python rwkv-convert.py <checkpoint>` to generate `.safetensors` file
//    where `<checkpoint>` is the path to the downloaded `.pth` file
// 3. Run `node rwkv.js <model> <tokenizer>` to run the model
//    where `<model>` is the path to the generated `.safetensors` file
//    and `<tokenizer>` is the path to the downloaded `.json` file

import { RWKV } from 'ggml-js/llms'
import { BPETokenizer } from 'ggml-js/tokenizers'

// Load the model and tokenizer
const model = RWKV.loadFromFile(process.argv[2])
const tokenizer = BPETokenizer.loadFromFile(process.argv[3])

// Generate text and print it one token at a time
for (const t of model.generate(tokenizer.encode('\nHello world!'))) {
  process.stdout.write(tokenizer.decodeOne(t))
}

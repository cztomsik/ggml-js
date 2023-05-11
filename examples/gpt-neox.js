// WIP: GPT-NeoX Example Usage
//
// this does not work yet!

import { GPTNeoX } from 'ggml-js/llms'
import { BPETokenizer } from 'ggml-js/tokenizers'

// Load the model and tokenizer
const model = GPTNeoX.loadFromFile(process.argv[2])
const tokenizer = BPETokenizer.loadFromFile('./20B_tokenizer.json')

// Generate text and print it one token at a time
for (const t of model.generate(tokenizer.encode('\nHello world!'))) {
  process.stdout.write(tokenizer.decodeOne(t))
}

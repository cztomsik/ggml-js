// WIP: GPT-NeoX Example Usage
//
// this does not work yet!

import { Context } from 'ggml-js/core'
import { GPTNeoX } from 'ggml-js/llms'
import { BPETokenizer } from 'ggml-js/tokenizers'

// Load the model and tokenizer
const model = GPTNeoX.loadFromFile(process.argv[2])
const tokenizer = BPETokenizer.loadFromFile('./20B_tokenizer.json')

// TODO
const ctx = Context.init()
const x = ctx.newTensor('i32', 1).set(0, 12092) // Hello
const y = model.forward(x)
const graph = ctx.buildForward(y)
graph.compute()

console.log(y.argmax()) // Should be 13

// Generate text and print it one token at a time
for (const t of model.generate(tokenizer.encode('\nHello world!'))) {
  process.stdout.write(tokenizer.decodeOne(t))
}

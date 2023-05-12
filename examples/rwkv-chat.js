// see examples/rwkv.js for instructions on how to use this script

import { moveCursor } from 'readline'
import { createInterface } from 'readline/promises'
import { RWKV } from 'ggml-js/llms'
import { BPETokenizer } from 'ggml-js/tokenizers'

const rl = createInterface({ input: process.stdin, output: process.stdout })
const model = RWKV.loadFromFile(process.argv[2])
const tokenizer = BPETokenizer.loadFromFile(process.argv[3])

while (true) {
  const instruction = await rl.question('User: ')
  const tokens = tokenizer.encode(`User: ${instruction}\nBot:`)

  moveCursor(process.stdout, 0, -1)

  for (const t of model.generate(tokens)) {
    process.stdout.write(tokenizer.decodeOne(t))
  }
}

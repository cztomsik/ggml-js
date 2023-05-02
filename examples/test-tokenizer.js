import { Tokenizer } from 'ggml-js'
import * as assert from 'node:assert'

const tokenizer = Tokenizer.fromFile('20B_tokenizer.json')

const text =
  '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.'
const tokens = tokenizer.encode(text)

assert.deepStrictEqual(
  tokens,
  [
    187, 688, 247, 29103, 4560, 13, 20687, 6888, 247, 33361, 273, 41705, 3811, 275, 247, 8905, 13, 3786, 35021, 2149,
    17836, 13, 275, 27061, 15, 4952, 625, 10084, 281, 253, 8607, 369, 253, 958, 326, 253, 41705, 7560, 3962, 5628, 15,
  ]
)

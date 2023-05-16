import { BPETokenizer } from 'ggml-js/tokenizers'
import * as assert from 'node:assert'

const tokenizer = BPETokenizer.loadFromFile('20B_tokenizer.json')

const expectTokens = (text, tokens) => assert.deepStrictEqual(tokenizer.encode(text), tokens)

expectTokens(
  '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.',
  [
    187, 688, 247, 29103, 4560, 13, 20687, 6888, 247, 33361, 273, 41705, 3811, 275, 247, 8905, 13, 3786, 35021, 2149,
    17836, 13, 275, 27061, 15, 4952, 625, 10084, 281, 253, 8607, 369, 253, 958, 326, 253, 41705, 7560, 3962, 5628, 15,
  ]
)

expectTokens(
  'Chinese: 我能吞下玻璃而不伤身体。',
  [
    27223, 27, 209, 15367, 20287, 7719, 241, 20720, 20241, 121, 163, 229, 214, 38328, 14274, 13127, 99, 45748, 29645,
    4340,
  ]
)

expectTokens(
  'Arabic(3): أنا قادر على أكل الزجاج و هذا لا يؤلمني.',
  [
    24850, 280, 9, 20, 2262, 23207, 43762, 47347, 40802, 6900, 25849, 38033, 23207, 12346, 4467, 19621, 112, 23072,
    10714, 107, 18152, 29802, 31602, 3142, 23630, 3142, 32443, 148, 99, 4467, 5843, 5846, 6463, 15,
  ]
)

expectTokens('<|endoftext|>', [0])

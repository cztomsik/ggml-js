// TODO: I'm not sure if this should be part of the project but HF tokenizers
//       cannot be installed right now.

import fs from 'node:fs'

export class Tokenizer {
  /**
   * Load a tokenizer from a file.
   * @param {string} path
   * @returns {Tokenizer}
   */
  static fromFile(path) {
    const { model } = JSON.parse(fs.readFileSync(path, 'utf8'))
    const { type, vocab, merges } = model

    switch (type) {
      case 'BPE':
        return new BPETokenizer(vocab, merges)
      default:
        throw new Error(`Unknown tokenizer type: ${type}`)
    }
  }

  /**
   * Encode a string into tokens.
   * @param {string} text
   * @returns {number[]}
   * @abstract
   */
  encode(text) {
    throw new Error('To be overridden')
  }

  /**
   * Decode tokens into a string.
   * @param {number[]} tokens
   * @returns {string}
   * @abstract
   */
  decode(tokens) {
    throw new Error('To be overridden')
  }
}

class BPETokenizer extends Tokenizer {
  constructor(vocab, merges) {
    super()
    this.vocab = vocab
    this.merges = merges
    this.tokenToId = new Map(Object.entries(vocab).map(([k, v]) => [v, k]))
  }

  /**
   * @override
   */
  decode(tokens) {
    return tokens.map(token => this.tokenToId.get(token)).join('')
  }
}

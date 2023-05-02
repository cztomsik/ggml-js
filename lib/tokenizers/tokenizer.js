// TODO: I'm not sure if this should be part of the project but HF tokenizers
//       cannot be installed right now.

import fs from 'node:fs'
import { BPETokenizer } from './bpe-tokenizer.js'

export class Tokenizer {
  /**
   * Load a tokenizer from a file.
   * @param {string} path
   * @returns {Tokenizer}
   */
  static loadFromFile(path) {
    const { model, added_tokens } = JSON.parse(fs.readFileSync(path, 'utf8'))

    switch (model.type) {
      case 'BPE':
        return new BPETokenizer({ ...model, added_tokens })
      default:
        throw new Error(`Unknown tokenizer type: ${model.type}`)
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
   */
  decode(tokens) {
    const chunks = tokens.map(token => this.decodeOne(token))
    const len = chunks.reduce((acc, chunk) => acc + chunk.length, 0)
    const buffer = new Uint8Array(len)
    let offset = 0

    for (const chunk of chunks) {
      buffer.set(chunk, offset)
      offset += chunk.length
    }

    return new TextDecoder('utf-8').decode(buffer)
  }

  /**
   * Decode one token into a Uint8Array.
   * @param {number} token
   * @returns {Uint8Array}
   * @abstract
   */
  decodeOne(token) {
    throw new Error('To be overridden')
  }
}

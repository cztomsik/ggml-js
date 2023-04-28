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

class BPETokenizer extends Tokenizer {
  constructor({ vocab, added_tokens }) {
    super()

    const entries = Object.entries(vocab)
    const te = new TextEncoder()
    this.tokenToChunk = Array(entries.length)

    for (const [token, id] of entries) {
      // TODO: is this the right way?
      this.tokenToChunk[id] = te.encode(token.replace('\u0120', ' '))
    }

    for (const { id, content } of added_tokens) {
      this.tokenToChunk[id] = te.encode(content)
    }
  }

  /**
   * @override
   */
  decodeOne(token) {
    return this.tokenToChunk[token]
  }
}

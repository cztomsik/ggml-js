// TODO: I'm not sure if this should be part of the project but HF tokenizers
//       cannot be installed right now.

import fs from 'node:fs'

/**
 * Base class for tokenizers.
 * @abstract
 */
export class Tokenizer {
  /**
   * Load a tokenizer from a file.
   * @type {<T extends typeof Tokenizer>(this: T, path: string) => InstanceType<T>}
   */
  static loadFromFile(path) {
    const { model, added_tokens } = JSON.parse(fs.readFileSync(path, 'utf8'))

    if (this === Tokenizer) {
      throw new Error('Tokenizer is an abstract class, use a subclass instead')
    }

    if (!this.name.startsWith(model.type)) {
      throw new Error(`File ${path} does not contain a ${this.name} tokenizer`)
    }

    // @ts-expect-error
    return new this({ ...model, added_tokens })
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

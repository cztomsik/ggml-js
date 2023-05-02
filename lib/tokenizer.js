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
  cache = new Map()
  vocab = new Map()
  ranks = new Map()
  chunks = new Map()
  // from https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py
  pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu

  constructor({ vocab, merges, added_tokens }) {
    super()

    // undo the weird byte encoding
    // ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠ!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁł¡¢£¤¥¦§¨©ª«¬Ń®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ
    const unWtf = str => str.replace(/[Ā-Ġġ-łŃ]/gu, c => String.fromCharCode(c.charCodeAt(0) - 256))

    const textEncoder = new TextEncoder()

    for (const k in vocab) {
      const tok = unWtf(k)
      // there are some duplicates in the vocab (13 and 221, 15 and 223)
      if (!this.vocab.has(tok)) {
        this.vocab.set(tok, vocab[k])
      }
      this.chunks.set(vocab[k], textEncoder.encode(tok))
    }

    for (const m of merges) {
      this.ranks.set(unWtf(m.replace(' ', '\0')), this.ranks.size)
    }

    for (const { id, content } of added_tokens) {
      const tok = unWtf(content)
      this.vocab.set(tok, id)
      this.chunks.set(id, textEncoder.encode(tok))
    }
  }

  /**
   * @override
   */
  encode(text) {
    const res = []

    for (const [word] of text.matchAll(this.pat)) {
      if (!this.cache.has(word)) {
        const toks = this.bpe([...word])
        const ids = toks.map(t => this.vocab.get(t))
        this.cache.set(word, ids)
      }

      res.push(...this.cache.get(word))
    }

    return res
  }

  bpe(word) {
    const bigrams = word.slice(1).map((it, i) => `${word[i]}\0${it}`)
    const bigram = bigrams.reduce(
      (min, bigram) => (this.ranks.get(bigram) < (this.ranks.get(min) ?? Infinity) ? bigram : min),
      ''
    )

    return bigram ? this.merge(word, bigram.split('\0')) : word
  }

  merge(word, bigram) {
    // Find the first occurrence and merge it
    const i = word.findIndex((part, i) => part === bigram[0] && word[i + 1] === bigram[1])
    word.splice(i, 2, bigram.join(''))

    return this.bpe(word)
  }

  /**
   * @override
   */
  decodeOne(token) {
    return this.chunks.get(token)
  }
}

import { Tokenizer } from './tokenizer.js'

export class BPETokenizer extends Tokenizer {
  cache = new Map()
  vocab = new Map()
  ranks = new Map()
  chunks = new Map()
  // from https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py
  pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu

  constructor({ vocab, merges, added_tokens }) {
    super()

    for (const k in vocab) {
      // there are some duplicates in the vocab (13 and 221, 15 and 223)
      if (!this.vocab.has(k)) {
        this.vocab.set(k, vocab[k])
      }
    }

    for (const m of merges) {
      this.ranks.set(m, this.ranks.size)
    }

    for (const { id, content } of added_tokens) {
      this.vocab.set(content, id)
    }

    for (const [k, v] of this.vocab.entries()) {
      // save decoded chunks
      this.chunks.set(v, textEncoder.encode(k.replace(/[Ā-Ġġ-łŃ]/gu, c => String.fromCharCode(c.charCodeAt(0) - 256))))
    }
  }

  /**
   * @override
   */
  encode(text) {
    const res = []

    for (const [word] of text.matchAll(this.pat)) {
      if (!this.cache.has(word)) {
        const toks = this.bpe(Array.from(textEncoder.encode(word), b => TABLE.charAt(b)))
        const ids = toks.map(t => this.vocab.get(t))
        this.cache.set(word, ids)
      }

      res.push(...this.cache.get(word))
    }

    return res
  }

  bpe(word) {
    const bigrams = word.slice(1).map((it, i) => `${word[i]} ${it}`)
    const bigram = bigrams.reduce(
      (min, bigram) => (this.ranks.get(bigram) < (this.ranks.get(min) ?? Infinity) ? bigram : min),
      ''
    )

    return bigram ? this.merge(word, bigram.split(' ')) : word
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

const textEncoder = new TextEncoder()
const TABLE =
  'ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠ!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁł¡¢£¤¥¦§¨©ª«¬Ń®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ'

import { Tokenizer } from './tokenizer.js'

const textEncoder = new TextEncoder()
// from https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py
const PAT = `'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+`
const TABLE =
  'ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠ!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁł¡¢£¤¥¦§¨©ª«¬Ń®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ'

export class BPETokenizer extends Tokenizer {
  cache = new Map()
  vocab = new Map()
  ranks = new Map()
  chunks = new Map()

  constructor({ vocab, merges, added_tokens }) {
    super()

    this.vocab = new Map(Object.entries(vocab))
    this.ranks = new Map(merges.map((m, i) => [m, i]))

    for (const [k, v] of this.vocab.entries()) {
      // save decoded chunks
      this.chunks.set(v, new Uint8Array(Array.from(k, ch => TABLE.indexOf(ch))))
    }

    for (const { id, content } of added_tokens) {
      this.cache.set(content, [id])
      this.chunks.set(id, content)
    }

    const escape = ({ content }) => content.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    this.pat = new RegExp(`${added_tokens.map(escape).join('|')}|${PAT}`, 'gu')
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

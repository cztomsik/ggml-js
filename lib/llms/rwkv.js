import { Context, Module, Embedding, LayerNorm, Linear, F } from 'ggml-js/core'
import { CausalLM } from './index.js'

/**
 * RWKV model
 * @see https://github.com/BlinkDL/RWKV-LM
 *
 * based on https://johanwind.github.io/2023/03/23/rwkv_details.html
 * and https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_in_150_lines.py
 * and https://github.com/saharNooby/rwkv.cpp
 */
export class RWKV extends CausalLM {
  #state

  /**
   * Create a new RWKV model.
   * @param {Context} context
   * @param {{ vocabSize: number, embedDim: number, numLayers: number }} hparams
   */
  constructor(context, { vocabSize, embedDim, numLayers }) {
    super(context)
    this[`blocks.0.ln0`] = new LayerNorm(this, embedDim)
    this.emb = new Embedding(this, vocabSize, embedDim)
    this.blocks = Array.from(Array(numLayers), _ => new Block(this, embedDim))
    this.ln_out = new LayerNorm(this, embedDim)
    this.head = new Linear(this, embedDim, vocabSize, { bias: false })
    this.#state = Array.from(Array(numLayers * 5), (_, i) =>
      this.context.newTensor1D('f32', embedDim).setAll(i % 5 === 4 ? -1e30 : 0)
    )
  }

  static loadHparams(header) {
    const [vocabSize, embedDim] = header[`emb.weight`].shape
    const numLayers = Object.keys(header).filter(k => k.endsWith('.ln1.weight')).length
    return { vocabSize, embedDim, numLayers }
  }

  forward(x, updates = []) {
    x = this[`blocks.0.ln0`].forward(this.emb.forward(x))
    x = this.blocks.reduce((x, block, i, _, o = i * 5) => block.forward(x, this.#state.slice(o, o + 5), updates), x)
    return this.head.forward(this.ln_out.forward(x))
  }
}

class Block extends Module {
  constructor(parentModule, dim) {
    super(parentModule)
    this.ln1 = new LayerNorm(this, dim)
    this.att = new TimeMix(this, dim)
    this.ln2 = new LayerNorm(this, dim)
    this.ffn = new ChannelMix(this, dim)
  }

  forward(x, state, updates) {
    x = F.add(x, this.att.forward(this.ln1.forward(x), state, updates))
    return F.add(x, this.ffn.forward(this.ln2.forward(x), state, updates))
  }
}

class TimeMix extends Module {
  constructor(parentModule, dim) {
    super(parentModule)
    this.time_decay = this.context.newTensor1D('f32', dim)
    this.time_first = this.context.newTensor1D('f32', dim)
    this.time_mix_k = this.context.newTensor1D('f32', dim)
    this.time_mix_v = this.context.newTensor1D('f32', dim)
    this.time_mix_r = this.context.newTensor1D('f32', dim)
    this.key = new Linear(this, dim, dim, { bias: false })
    this.value = new Linear(this, dim, dim, { bias: false })
    this.receptance = new Linear(this, dim, dim, { bias: false })
    this.output = new Linear(this, dim, dim, { bias: false })
  }

  forward(x, [_, prev_x, aa, bb, pp], updates) {
    const xk = F.add(F.mul(x, this.time_mix_k), F.mul(prev_x, F.oneMinusX(this.time_mix_k)))
    const xv = F.add(F.mul(x, this.time_mix_v), F.mul(prev_x, F.oneMinusX(this.time_mix_v)))
    const xr = F.add(F.mul(x, this.time_mix_r), F.mul(prev_x, F.oneMinusX(this.time_mix_r)))
    const r = F.sigmoid(this.receptance.forward(xr))
    const k = this.key.forward(xk)
    const v = this.value.forward(xv)

    let ww = F.add(this.time_first, k)
    let qq = F.max(pp, ww)
    let e1 = F.exp(F.sub(pp, qq))
    let e2 = F.exp(F.sub(ww, qq))
    const a = F.add(F.mul(e1, aa), F.mul(e2, v))
    const b = F.add(F.mul(e1, bb), e2)
    const wkv = F.div(a, b)
    ww = F.add(pp, this.time_decay)
    qq = F.max(ww, k)
    e1 = F.exp(F.sub(ww, qq))
    e2 = F.exp(F.sub(k, qq))

    updates.push(
      // dest, src
      [prev_x, x],
      [aa, F.add(F.mul(e1, aa), F.mul(e2, v))],
      [bb, F.add(F.mul(e1, bb), e2)],
      [pp, qq]
    )

    return this.output.forward(F.mul(r, wkv))
  }
}

class ChannelMix extends Module {
  constructor(parentModule, dim) {
    super(parentModule)
    this.time_mix_k = this.context.newTensor1D('f32', dim)
    this.time_mix_r = this.context.newTensor1D('f32', dim)
    this.key = new Linear(this, dim, 4 * dim, { bias: false })
    this.receptance = new Linear(this, dim, dim, { bias: false })
    this.value = new Linear(this, 4 * dim, dim, { bias: false })
  }

  forward(x, [prev_x], updates) {
    const xk = F.add(F.mul(x, this.time_mix_k), F.mul(prev_x, F.oneMinusX(this.time_mix_k)))
    const xr = F.add(F.mul(x, this.time_mix_r), F.mul(prev_x, F.oneMinusX(this.time_mix_r)))
    const r = F.sigmoid(this.receptance.forward(xr))
    const k = F.square(F.relu(this.key.forward(xk)))

    updates.push([prev_x, x])

    return F.mul(r, this.value.forward(k))
  }
}

// based on https://johanwind.github.io/2023/03/23/rwkv_details.html
// and https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_in_150_lines.py
// - download https://huggingface.co/BlinkDL/rwkv-4-pile-169m/blob/main/RWKV-4-Pile-169M-20220807-8023.pth
// - run `python convert.py <file>` to generate `.safetensors` file
// - run `node rwkv.js <file>` to run the model

import { Context, Model, Module, Embedding, LayerNorm, Linear, F } from 'ggml-js'

// TODO: model.hparams?
const [N_VOCAB, N_EMB, N_LAYER] = [50277, 768, 12]

class RWKV extends Model {
  [`blocks.0.ln0`] = new LayerNorm(this, N_EMB)
  emb = new Embedding(this, N_VOCAB, N_EMB)
  blocks = Array.from(Array(N_LAYER), _ => new Block(this))
  ln_out = new LayerNorm(this, N_EMB)
  head = new Linear(this, N_EMB, N_VOCAB, { bias: false })

  forward(x) {
    x = this[`blocks.0.ln0`].forward(this.emb.forward(x))
    this.blocks.forEach(b => (x = b.forward(x)))
    return this.head.forward(this.ln_out.forward(x))
  }
}

class Block extends Module {
  ln1 = new LayerNorm(this, N_EMB)
  att = new TimeMix(this)
  ln2 = new LayerNorm(this, N_EMB)
  ffn = new ChannelMix(this)
  #prev = new State(this)

  forward(x) {
    x = F.add(x, this.att.forward(this.ln1.forward(x), this.#prev))
    return F.add(x, this.ffn.forward(this.ln2.forward(x), this.#prev))
  }
}

class State extends Module {
  cx = this.context.newTensor1D('f32', N_EMB).setAll(0)
  tx = this.context.newTensor1D('f32', N_EMB).setAll(0)
  aa = this.context.newTensor1D('f32', N_EMB).setAll(0)
  bb = this.context.newTensor1D('f32', N_EMB).setAll(-1e30)
  pp = this.context.newTensor1D('f32', N_EMB).setAll(0)
}

class TimeMix extends Module {
  time_decay = this.context.newTensor1D('f32', N_EMB)
  time_first = this.context.newTensor1D('f32', N_EMB)
  time_mix_k = this.context.newTensor1D('f32', N_EMB)
  time_mix_v = this.context.newTensor1D('f32', N_EMB)
  time_mix_r = this.context.newTensor1D('f32', N_EMB)
  key = new Linear(this, N_EMB, N_EMB, { bias: false })
  value = new Linear(this, N_EMB, N_EMB, { bias: false })
  receptance = new Linear(this, N_EMB, N_EMB, { bias: false })
  output = new Linear(this, N_EMB, N_EMB, { bias: false })

  forward(x, prev) {
    const xk = F.add(F.mul(x, this.time_mix_k), F.mul(prev.tx, F.oneMinusX(this.time_mix_k)))
    const xv = F.add(F.mul(x, this.time_mix_v), F.mul(prev.tx, F.oneMinusX(this.time_mix_v)))
    const xr = F.add(F.mul(x, this.time_mix_r), F.mul(prev.tx, F.oneMinusX(this.time_mix_r)))
    prev.tx = x
    const r = F.sigmoid(this.receptance.forward(xr))
    const k = this.key.forward(xk)
    const v = this.value.forward(xv)

    let ww = F.add(this.time_first, k)
    let qq = F.max(prev.pp, ww)
    let e1 = F.exp(F.sub(prev.pp, qq))
    let e2 = F.exp(F.sub(ww, qq))
    const a = F.add(F.mul(e1, prev.aa), F.mul(e2, v))
    const b = F.add(F.mul(e1, prev.bb), e2)
    const wkv = F.div(a, b)
    ww = F.add(prev.pp, this.time_decay)
    qq = F.max(ww, k)
    e1 = F.exp(F.sub(ww, qq))
    e2 = F.exp(F.sub(k, qq))
    prev.aa = F.add(F.mul(e1, prev.aa), F.mul(e2, v))
    prev.bb = F.add(F.mul(e1, prev.bb), e2)
    prev.pp = qq
    return this.output.forward(F.mul(r, wkv))
  }
}

class ChannelMix extends Module {
  time_mix_k = this.context.newTensor1D('f32', N_EMB)
  time_mix_r = this.context.newTensor1D('f32', N_EMB)
  key = new Linear(this, N_EMB, 4 * N_EMB, { bias: false })
  receptance = new Linear(this, N_EMB, N_EMB, { bias: false })
  value = new Linear(this, 4 * N_EMB, N_EMB, { bias: false })

  forward(x, prev) {
    const k = this.key.forward(F.add(F.mul(x, this.time_mix_k), F.mul(prev.cx, F.oneMinusX(this.time_mix_k))))
    const r = this.receptance.forward(F.add(F.mul(x, this.time_mix_r), F.mul(prev.cx, F.oneMinusX(this.time_mix_r))))
    const vk = this.value.forward(F.square(F.relu(k)))
    prev.cx = x
    return F.mul(F.sigmoid(r), vk)
  }
}

// TODO: no_alloc: true, this is currently broken (unary functions like F.fun(mmappedX))
const ctx = Context.init({ mem_size: BigInt(700_000_000) })
const model = new RWKV(ctx)
model.loadFromFile(process.argv[2])
// model.print()

// RNN only has one input
const x = ctx.newTensor1D('i32', 1)

// Prepare the graph
const out = F.softmax(model.forward(x))
const graph = ctx.buildForward(out)

// push few tokens from from https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v4/20B_tokenizer.json
x.set(0, 12092) // `Hello`
graph.compute()
x.set(0, 1533) // `Ġworld`
graph.compute()
x.set(0, 2) // `!`
graph.compute()
x.set(0, 831) // `ĠThis`
graph.compute()
x.set(0, 310) // `Ġis`
graph.compute()
x.set(0, 247) // `Ġa`
graph.compute()

// sample
for (let i = 0; i < N_VOCAB; i++) {
  const p = out.get(i)
  if (p > 0.1) {
    console.log('p', p, 'i', i)
  }
}

console.log('done')

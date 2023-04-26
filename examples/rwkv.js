// based on https://johanwind.github.io/2023/03/23/rwkv_details.html
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
  tx = this.context.newTensor1D('f32', N_EMB)
  cx = this.context.newTensor1D('f32', N_EMB)
  num = this.context.newTensor1D('f32', N_EMB)
  den = this.context.newTensor1D('f32', N_EMB)
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
    const k = this.key.forward(F.add(F.mul(x, this.time_mix_k), F.mul(prev.tx, F.oneMinusX(this.time_mix_k))))
    const v = this.value.forward(F.add(F.mul(x, this.time_mix_v), F.mul(prev.tx, F.oneMinusX(this.time_mix_v))))
    const r = this.receptance.forward(F.add(F.mul(x, this.time_mix_r), F.mul(prev.tx, F.oneMinusX(this.time_mix_r))))

    const wkv = F.div(
      F.add(prev.num, F.mul(F.exp(F.add(this.time_first, k)), v)),
      F.add(prev.den, F.exp(F.add(this.time_first, k)))
    )
    const rwkv = F.mul(F.sigmoid(r), wkv)

    const num = F.add(F.mul(F.exp(F.neg(F.exp(this.time_decay))), prev.num), F.mul(F.exp(k), v))
    const den = F.add(F.mul(F.exp(F.neg(F.exp(this.time_decay))), prev.den), F.exp(k))

    // update state
    Object.assign(prev, { tx: x, num, den })

    return this.output.forward(rwkv)
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

    // update state
    prev.cx = x

    return F.mul(F.sigmoid(r), vk)
  }
}

// this one is mmapped
// TODO: no_alloc: true, this is currently broken (unary functions like F.fun(mmappedX))
const ctx = Context.init({ mem_size: BigInt(700_000_000) })
const model = new RWKV(ctx)
model.loadFromFile(process.argv[2])
model.print()

// this one allocates
const ctx2 = Context.init({ mem_size: BigInt(100_000_000) })
const x = ctx2.newTensor1D('i32', 1)
x.set(0, 1)

const out = model.forward(x)
const graph = ctx.buildForward(out)
console.log(out.get(0))

console.log('done')

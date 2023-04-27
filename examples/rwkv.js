// based on https://johanwind.github.io/2023/03/23/rwkv_details.html
// and https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_in_150_lines.py
// and https://github.com/saharNooby/rwkv.cpp
//
// - download https://huggingface.co/BlinkDL/rwkv-4-pile-169m/blob/main/RWKV-4-Pile-169M-20220807-8023.pth
// - run `python convert.py <file>` to generate `.safetensors` file
// - run `node rwkv.js <file>` to run the model

import { Context, LLM, Module, Embedding, LayerNorm, Linear, F } from 'ggml-js'

const [N_VOCAB, N_EMB, N_LAYER] = [50277, 768, 12]

class RWKV extends LLM {
  [`blocks.0.ln0`] = new LayerNorm(this, N_EMB)
  emb = new Embedding(this, N_VOCAB, N_EMB)
  blocks = Array.from(Array(N_LAYER), _ => new Block(this))
  ln_out = new LayerNorm(this, N_EMB)
  head = new Linear(this, N_EMB, N_VOCAB, { bias: false })
  #state = Array.from(Array(N_LAYER * 5), (_, i) =>
    this.context.newTensor1D('f32', N_EMB).setAll(i % 5 === 3 ? -1e30 : 0)
  )

  forward(x, updates = []) {
    x = this[`blocks.0.ln0`].forward(this.emb.forward(x))
    x = this.blocks.reduce((x, block, i, _, o = i * 5) => block.forward(x, this.#state.slice(o, o + 5), updates), x)
    return this.head.forward(this.ln_out.forward(x))
  }
}

class Block extends Module {
  ln1 = new LayerNorm(this, N_EMB)
  att = new TimeMix(this)
  ln2 = new LayerNorm(this, N_EMB)
  ffn = new ChannelMix(this)

  forward(x, state, updates) {
    x = F.add(x, this.att.forward(this.ln1.forward(x), state, updates))
    return F.add(x, this.ffn.forward(this.ln2.forward(x), state, updates))
  }
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
      F.cpy(prev_x, x),
      F.cpy(aa, F.add(F.mul(e1, aa), F.mul(e2, v))),
      F.cpy(bb, F.add(F.mul(e1, bb), e2)),
      F.cpy(pp, qq)
    )

    return this.output.forward(F.mul(r, wkv))
  }
}

class ChannelMix extends Module {
  time_mix_k = this.context.newTensor1D('f32', N_EMB)
  time_mix_r = this.context.newTensor1D('f32', N_EMB)
  key = new Linear(this, N_EMB, 4 * N_EMB, { bias: false })
  receptance = new Linear(this, N_EMB, N_EMB, { bias: false })
  value = new Linear(this, 4 * N_EMB, N_EMB, { bias: false })

  forward(x, [prev_x], updates) {
    const k = this.key.forward(F.add(F.mul(x, this.time_mix_k), F.mul(prev_x, F.oneMinusX(this.time_mix_k))))
    const r = this.receptance.forward(F.add(F.mul(x, this.time_mix_r), F.mul(prev_x, F.oneMinusX(this.time_mix_r))))
    const vk = this.value.forward(F.square(F.relu(k)))
    updates.push(F.cpy(prev_x, x))
    return F.mul(F.sigmoid(r), vk)
  }
}

// TODO: no_alloc: true, this is currently broken (unary functions like F.fun(mmappedX))
const ctx = Context.init({ mem_size: BigInt(700_000_000) })
const model = new RWKV(ctx)
model.loadFromFile(process.argv[2])
// model.print()

// push few tokens from from https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v4/20B_tokenizer.json
const tokens = [`Hello`, `Ġworld`, `!`, `ĠThis`, `Ġis`, `Ġa`]
const ids = [12092, 1533, 2, 831, 310, 247]

for (const predictions of model.generate(ids)) {
  console.log(predictions)
}

console.log('done')

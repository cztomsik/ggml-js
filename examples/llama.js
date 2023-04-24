import { Context, Model, F } from 'ggml-js'

const N_VOCAB = 32001
const N_CTX = 512
const N_EMBED = 4096
const N_HIDDEN = 11008

if (process.argv.length < 3) {
  console.log('Usage: node test-llama.js <path to model>')
  process.exit(1)
}

class LLaMA extends Model {
  constructor(ctx) {
    super()
    this.tok_embeddings = new Embedding(ctx, N_VOCAB, N_EMBED)
    this.norm = new RMSNorm(ctx)
    this.output = new Linear(ctx, N_EMBED, N_VOCAB)
    this.layers = Array.from(Array(32), (_, i) => new Layer(ctx, i))
  }

  forward(x) {
    x = this.tok_embeddings.forward(x)

    for (const layer of this.layers) {
      x = layer.forward(x)
    }

    return this.output.forward(this.norm.forward(x))
  }
}

class Embedding {
  constructor(ctx, n_vocab, n_embed) {
    this.weight = ctx.newTensor2D('q4_0', n_vocab, n_embed)
  }

  forward(x) {
    return F.embedding(x, this.weight)
  }
}

class Layer {
  constructor(ctx, i) {
    this.attention = new SelfAttention(ctx)
    this.attention_norm = new RMSNorm(ctx)
    this.feed_forward = new FeedForward(ctx)
    this.ffn_norm = new RMSNorm(ctx)
  }

  forward(x) {
    x = F.add(x, this.attention.forward(this.attention_norm.forward(x)))
    x = F.add(x, this.feed_forward.forward(this.ffn_norm.forward(x)))
    return x
  }
}

class SelfAttention {
  constructor(ctx) {
    this.wq = new Linear(ctx, N_EMBED, N_EMBED)
    this.wk = new Linear(ctx, N_EMBED, N_EMBED)
    this.wv = new Linear(ctx, N_EMBED, N_EMBED)
    this.wo = new Linear(ctx, N_EMBED, N_EMBED)
  }

  forward(x) {
    return x
  }
}

class RMSNorm {
  constructor(ctx) {
    this.weight = ctx.newTensor1D('f32', N_EMBED)
  }

  forward(x) {
    return F.mul(F.rmsNorm(x), this.weight)
  }
}

class FeedForward {
  constructor(ctx) {
    this.w1 = new Linear(ctx, N_EMBED, N_HIDDEN)
    this.w2 = new Linear(ctx, N_HIDDEN, N_EMBED)
    this.w3 = new Linear(ctx, N_EMBED, N_HIDDEN)
  }

  forward(x) {
    return this.w2.forward(F.mul(this.w1.forward(x), this.w3.forward(x)))
  }
}

export class Linear {
  constructor(ctx, inputDim, outputDim) {
    this.weight = ctx.newTensor2D('q4_0', inputDim, outputDim)
  }

  forward(x) {
    return F.mul(x, this.weight)
  }
}

const ctx = Context.init({ mem_size: BigInt(100_000), no_alloc: true })
const model = new LLaMA(ctx)
model.print()
ctx.printObjects()

const input = ctx.newTensor1D('i32', N_CTX)
const output = model.forward(input)

const graph = ctx.buildForward(output)
graph.print()
graph.compute()

//native.foo(process.argv[2])

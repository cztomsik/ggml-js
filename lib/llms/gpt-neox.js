import { Embedding, F, LayerNorm, Linear, Module } from 'ggml-js/core'
import { CausalLM } from './causal.js'

export class GPTNeoX extends CausalLM {
  constructor(context, { vocab_size, embed_dim, num_layers, num_heads, num_rot }) {
    super(context)
    this.embed_in = new Embedding(this, vocab_size, embed_dim)
    this.layers = Array.from(Array(num_layers), _ => new Block(this, embed_dim, num_heads, num_rot))
    this.final_layer_norm = new LayerNorm(this, embed_dim)
    this.embed_out = new Linear(this, embed_dim, vocab_size, { bias: false })
  }

  forward(x, state, updates = []) {
    x = this.embed_in.forward(x)
    x = this.layers.reduce((x, layer) => layer.forward(x), x)
    return this.embed_out.forward(this.final_layer_norm.forward(x))
  }

  getInitialState(ctx) {
    return {}
  }
}

class Block extends Module {
  constructor(parentModule, embed_dim, num_heads, num_rot) {
    super(parentModule)
    this.input_layernorm = new LayerNorm(this, embed_dim)
    this.attention = new Attention(this, embed_dim, num_heads, num_rot)
    this.post_attention_layernorm = new LayerNorm(this, embed_dim)
    this.mlp = new MLP(this, embed_dim)
  }

  forward(x) {
    // parallel residual
    // https://github.com/huggingface/transformers/blob/3cf01b2060625de99f19f29e48f889c76cbf5cdf/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L344
    const att = this.attention.forward(this.input_layernorm.forward(x))
    return F.add(x, F.add(att, this.mlp.forward(this.post_attention_layernorm.forward(x))))
  }
}

class Attention extends Module {
  constructor(parentModule, embed_dim, num_heads, num_rot) {
    super(parentModule)

    this.embed_dim = embed_dim
    this.num_heads = num_heads
    this.head_size = embed_dim / num_heads
    this.num_rot = num_rot

    this.query_key_value = new Linear(this, embed_dim, embed_dim * 3)
    this.dense = new Linear(this, embed_dim, embed_dim)
  }

  // TODO: KV cache[N] from state
  // TODO: copy K/V to K/V[N-1]
  // TODO: shift KV to left in update step
  // TODO: pad during init
  forward(x) {
    const scale = x.context.newTensor('f32', 1).set(0, 1 / Math.sqrt(this.embed_dim / this.num_heads))

    let [q, k, v] = this.query_key_value
      .forward(x)
      .view(3 * this.head_size, this.num_heads)
      .split(this.head_size)

    // ROPE
    q = F.rope(q, 1, this.num_rot, 2)
    k = F.rope(k, 1, this.num_rot, 2)

    let kq = F.matmul(k, q)
    kq = F.scale(kq, scale)
    kq = F.softmax(kq)

    x = F.matmul(v.permute(1, 0, 2, 3).contiguous(), kq)

    return this.dense.forward(x.view(-1))
  }
}

class MLP extends Module {
  constructor(parentModule, dim) {
    super(parentModule)
    this.dense_h_to_4h = new Linear(this, dim, dim * 4)
    this.dense_4h_to_h = new Linear(this, dim * 4, dim)
  }

  forward(x) {
    return this.dense_4h_to_h.forward(F.gelu(this.dense_h_to_4h.forward(x)))
  }
}

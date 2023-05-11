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
    x = F.add(x, this.attention.forward(this.input_layernorm.forward(x)))
    return F.add(x, this.mlp.forward(this.post_attention_layernorm.forward(x)))
  }
}

class Attention extends Module {
  constructor(parentModule, embed_dim, num_heads, num_rot) {
    super(parentModule)

    this.num_heads = num_heads
    this.num_rot = num_rot

    this.query_key_value = new Linear(this, embed_dim, embed_dim * 3)
    this.dense = new Linear(this, embed_dim, embed_dim)
  }

  // TODO
  forward(x) {
    const [T, C] = x.shape

    return this.dense.forward(x)
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

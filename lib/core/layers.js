import { Module } from './module.js'
import * as F from './functional.js'

/**
 * Embedding layer.
 */
export class Embedding extends Module {
  /**
   * @param {Module} parentModule - A parent module.
   * @param {number} numEmbeddings - The size of the vocabulary.
   * @param {number} dim - The embedding dimension.
   */
  constructor(parentModule, numEmbeddings, dim) {
    super(parentModule)
    // torch.nn.Embedding(num, dim).weight.shape == (num, dim)
    // but ggml shape is in reverse order so it's (dim, num)
    this.weight = this.context.newTensor2D('f32', dim, numEmbeddings)
  }

  forward(x) {
    return F.embedding(x, this.weight)
  }
}

/**
 * Layer normalization.
 */
export class LayerNorm extends Module {
  /**
   * @param {Module} parentModule - A parent module.
   * @param {number} dim - The embedding dimension.
   */
  constructor(parentModule, dim) {
    super(parentModule)
    this.weight = this.context.newTensor1D('f32', dim)
    this.bias = this.context.newTensor1D('f32', dim)
  }

  forward(x) {
    // TODO: there's a small difference between torch and ggml (1e-7)
    return F.add(F.mul(F.norm(x), this.weight), this.bias)
  }
}

/**
 * Linear layer.
 */
export class Linear extends Module {
  /**
   * @param {Module} parentModule - A parent module.
   * @param {number} inputDim - The input dimension.
   * @param {number} outputDim - The output dimension.
   * @param {{bias?: boolean}} [options] - Options.
   */
  constructor(parentModule, inputDim, outputDim, { bias = true } = {}) {
    super(parentModule)
    // torch.nn.Linear(in, out).weight.shape == (out, in)
    // but ggml shape is in reverse order so it's (in, out)
    this.weight = this.context.newTensor2D('f32', inputDim, outputDim)
    this.bias = bias ? this.context.newTensor1D('f32', outputDim) : null
  }

  forward(x) {
    const res = F.matmul(this.weight, x)

    return this.bias ? F.add(res, this.bias) : res
  }
}

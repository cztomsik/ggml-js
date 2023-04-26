import { Module } from './module.js'
import * as F from './functional.js'

/**
 * Embedding layer.
 */
export class Embedding extends Module {
  /**
   * @param {Module} parentModule - A parent module.
   * @param {number} vocabSize - The size of the vocabulary.
   * @param {number} dim - The embedding dimension.
   */
  constructor(parentModule, vocabSize, dim) {
    super(parentModule)
    this.weight = this.context.newTensor2D('f32', dim, vocabSize)
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
    this.weight = this.context.newTensor2D('f32', inputDim, outputDim)
    this.bias = bias ? this.context.newTensor1D('f32', outputDim) : null
  }

  forward(x) {
    const res = F.mul(x, this.weight)

    return this.bias ? F.add(res, this.bias) : res
  }
}

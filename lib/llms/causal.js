import { Tensor, Model, F } from 'ggml-js/core'

/**
 * Base class for all causal language models.
 */
export class CausalLM extends Model {
  /**
   * @param {Tensor} x - Input tensor.
   * @param {Tensor[]} updates - Array of tensors that should be updated after the forward pass.
   * @returns {Tensor} Output tensor.
   * @abstract
   */
  forward(x, updates = []) {
    throw new Error('To be implemented by subclasses')
  }

  *generate(input, n = 100) {
    const updates = []
    const x = this.context.newTensor1D('i32', 1)
    const out = F.softmax(this.forward(x, updates))
    const [_, N_VOCAB] = out.shape
    const graph = this.context.buildForward(out)
    const update = () => updates.forEach(([dst, src]) => dst.copyFrom(src))

    // make sure the update tensors will be computed
    updates.forEach(([_, src]) => graph.forwardExpand(src))

    // feed the input one by one
    for (const v of input) {
      x.set(0, v)
      graph.compute()
      yield v
      update()
    }

    // generate
    for (let i = 0; i < n; i++) {
      // TODO: sampling
      let res = out.argmax()

      yield res

      x.set(0, res)
      graph.compute()
      update()
    }
  }
}

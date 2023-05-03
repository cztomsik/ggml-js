import { Tensor, Model, Context, F } from 'ggml-js/core'

/**
 * Base class for all causal language models.
 */
export class CausalLM extends Model {
  /**
   * @param {Tensor} x - Input tensor.
   * @param {any} state
   * @param {Tensor[]} updates - Array of tensors that should be updated after the forward pass.
   * @returns {Tensor} Output tensor.
   * @abstract
   */
  forward(x, state, updates = []) {
    throw new Error('To be implemented by subclasses')
  }

  /**
   * Prepare the initial state.
   * @param {Context} cx
   * @returns {any} Initial state.
   * @abstract
   */
  getInitialState(cx) {
    throw new Error('To be implemented by subclasses')
  }

  *generate(input, n = 100) {
    const cx = Context.init({ mem_size: BigInt(100_000_0000) })
    const x = cx.newTensor1D('i32', 1)
    const state = this.getInitialState(cx)
    const updates = []
    const out = F.softmax(this.forward(x, state, updates))
    const [_, N_VOCAB] = out.shape
    const graph = cx.buildForward(out)
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

import { Tensor, Model, Context, F } from 'ggml-js/core'
import { native } from '../core/native.js'

if ('SEED' in process.env) {
  native.sample_seed(BigInt(process.env.SEED))
} else if (!process.env.NO_SEED) {
  native.sample_seed(BigInt(Date.now()))
}

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

  /**
   * @param {number[]} input - Array of input token IDs.
   * @param {object} [options]
   * @param {number} [options.max_tokens=100] - Maximum number of tokens to generate.
   * @param {number} [options.top_k=40] - Top-k sampling cutoff.
   * @param {number} [options.top_p=0.85] - Top-p sampling cutoff.
   * @param {number} [options.temperature=1.0] - Sampling temperature.
   */
  *generate(input, { max_tokens = 100, top_k = 40, top_p = 0.85, temperature = 1.0 } = {}) {
    const cx = Context.init({ mem_size: BigInt(100_000_0000) })
    const x = cx.newTensor1D('i32', 1)
    const state = this.getInitialState(cx)
    const updates = []
    const logits = this.forward(x, state, updates)
    const probs = F.softmax(logits)
    const graph = cx.buildForward(probs)
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
    for (let i = 0; i < max_tokens; i++) {
      let res = native.sample_top_k_top_p(probs, top_k, top_p, temperature)

      yield res

      x.set(0, res)
      graph.compute()
      update()
    }
  }
}

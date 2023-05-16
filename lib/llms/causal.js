import { Tensor, Model, Context, F } from 'ggml-js/core'
import { native } from '../core/native.js'

if ('SEED' in process.env) {
  native.sample_seed(BigInt(process.env.SEED))
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
   * @param {number} [options.stop_token=null] - Stop token ID.
   */
  *generate(input, { max_tokens = 100, top_k = 40, top_p = 0.85, temperature = 1.0, stop_token = null } = {}) {
    const cx = Context.init({ mem_size: BigInt(100_000_0000) })
    const x = cx.newTensor('i32', 1)
    const state = this.getInitialState(cx)
    const updates = []
    const logits = this.forward(x, state, updates)
    const probs = F.softmax(logits)
    const graph = cx.buildForward(probs)

    // make sure the updates will be applied
    updates.forEach(([dest, src]) => graph.buildForwardExpand(F.cpy(src, dest)))

    // feed the input one by one
    for (const v of input) {
      x.set(0, v)
      graph.compute()
      yield v
    }

    // generate
    for (let i = 0; i < max_tokens; i++) {
      let res = native.sample_top_k_top_p(probs, top_k, top_p, temperature)

      if (res === stop_token) {
        break
      }

      yield res

      x.set(0, res)
      graph.compute()
    }
  }
}

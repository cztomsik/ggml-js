import assert from 'assert'
import { native, wrap } from './native.js'
import { Tensor } from './tensor.js'

export const GGML_TYPE = Object.fromEntries(
  Object.entries(native)
    .filter(([k, _]) => k.startsWith('GGML_TYPE_'))
    .map(([k, v]) => [k.slice(10).toLowerCase(), v])
)

export const GGML_TYPE_NAMES = new Map(Object.entries(GGML_TYPE).map(([k, v]) => [v, k]))

/**
 * GGML context.
 */
export class Context {
  /**
   * Initialize the context.
   */
  static init(opts = {}) {
    opts = {
      mem_size: BigInt(1_000_000),
      mem_buffer: null,
      no_alloc: false,
      ...opts,
    }

    return wrap(Context, native.ggml_init(opts))
  }

  // internal
  callOp(nativeFun, ...args) {
    return wrap(Tensor, nativeFun(this, ...args), { context: this })
  }

  /**
   * Create a new tensor (shape in reverse order).
   * @param {string} type - Tensor type.
   * @param {...number} dims - Tensor shape.
   */
  newTensor(type, ...dims) {
    assert(dims.length <= 4, 'GGML only supports up to 4 dimensions')
    const fun = native[`ggml_new_tensor_${dims.length}d`]
    return this.callOp(fun, GGML_TYPE[type], ...dims.map(BigInt))
  }

  /**
   * Build a graph for forward computation.
   * @param {Tensor} tensor
   */
  buildForward(tensor) {
    return wrap(Graph, native.ggml_build_forward(tensor), { context: this })
  }

  /**
   * Print GGML objects
   **/
  printObjects() {
    native.ggml_print_objects(this)
  }
}

/**
 * GGML Graph.
 */
class Graph {
  context = null

  /**
   * Compute the graph.
   */
  compute(n_threads = 4) {
    native.ggml_graph_compute_with_ctx(this.context, this, n_threads)
  }

  /**
   * Reset the graph.
   */
  reset() {
    native.ggml_graph_reset(this)
  }

  /**
   * Print info about the graph.
   */
  print() {
    native.ggml_graph_print(this)
  }

  /**
   * TODO: document this
   */
  buildForwardExpand(tensor) {
    native.ggml_build_forward_expand(this, tensor)
  }
}

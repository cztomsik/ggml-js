import assert from 'assert'
import { native, wrap } from './native.js'
import { Tensor } from './tensor.js'

export const GGML_TYPE = Object.fromEntries(
  Object.entries(native)
    .filter(([k, v]) => k.startsWith('GGML_TYPE_'))
    .map(([k, v]) => [k.slice(10).toLowerCase(), v])
)

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
   * Create a new tensor.
   * @param {string} type - Tensor type.
   * @param {...number} dims - Tensor shape.
   */
  newTensor(type, ...dims) {
    assert(dims.length <= 4, 'GGML supports up to 4 dimensions')
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
  compute() {
    native.ggml_graph_compute(this.context, this)
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

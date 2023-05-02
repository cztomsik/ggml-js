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
  newTensor(nativeFun, type, ...dims) {
    return wrap(Tensor, nativeFun(this, GGML_TYPE[type], ...dims.map(BigInt)), { context: this })
  }

  /**
   * Create a new 1D tensor.
   */
  newTensor1D(type, dim) {
    return this.newTensor(native.ggml_new_tensor_1d, type, dim)
  }

  /**
   * Create a new 2D tensor.
   * Note that GGML shape is in reverse order.
   */
  newTensor2D(type, dim1, dim2) {
    return this.newTensor(native.ggml_new_tensor_2d, type, dim1, dim2)
  }

  /**
   * Create a new 3D tensor.
   * Note that GGML shape is in reverse order.
   */
  newTensor3D(type, dim1, dim2, dim3) {
    return this.newTensor(native.ggml_new_tensor_3d, type, dim1, dim2, dim3)
  }

  /**
   * Create a new 4D tensor.
   * Note that GGML shape is in reverse order.
   */
  newTensor4D(type, dim1, dim2, dim3, dim4) {
    return this.newTensor(native.ggml_new_tensor_4d, type, dim1, dim2, dim3, dim4)
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
  forwardExpand(tensor) {
    native.ggml_build_forward_expand(this, tensor)
  }
}

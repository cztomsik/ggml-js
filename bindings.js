import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)

const targets = {
  darwin: 'macos',
  linux: 'linux',
  win32: 'windows',
}
const native = require(`./zig-out/lib/ggml.${targets[process.platform]}.node`)

export const GGML_TYPE = Object.fromEntries(
  Object.entries(native)
    .filter(([k, v]) => k.startsWith('GGML_TYPE_'))
    .map(([k, v]) => [k.slice(10).toLowerCase(), v])
)

export class Context {
  /**
   * Initialize the context.
   * @returns {Context}
   */
  static init(opts = {}) {
    opts = {
      mem_size: BigInt(1024),
      mem_buffer: null,
      no_alloc: false,
      ...opts,
    }

    return wrap(native.ggml_init(opts), Context)
  }

  wrapTensor(obj, type) {
    return (obj.context = this), (obj.type = type), wrap(obj, Tensor)
  }

  wrapGraph(obj) {
    return (obj.context = this), wrap(obj, Graph)
  }

  /**
   * Create a new 1D tensor.
   * @returns {Tensor}
   */
  newTensor1D(type, dim) {
    return this.wrapTensor(native.ggml_new_tensor_1d(this, GGML_TYPE[type], big(dim)), type)
  }

  /**
   * Create a new 2D tensor.
   * @returns {Tensor}
   */
  newTensor2D(type, dim1, dim2) {
    return this.wrapTensor(native.ggml_new_tensor_2d(this, GGML_TYPE[type], big(dim1), big(dim2)), type)
  }

  /**
   * Create a new 3D tensor.
   * @returns {Tensor}
   */
  newTensor3D(type, dim1, dim2, dim3) {
    return this.wrapTensor(native.ggml_new_tensor_3d(this, GGML_TYPE[type], big(dim1), big(dim2), big(dim3)), type)
  }

  /**
   * Create a new 4D tensor.
   * @returns {Tensor}
   */
  newTensor4D(type, dim1, dim2, dim3, dim4) {
    return this.wrapTensor(
      native.ggml_new_tensor_4d(this, GGML_TYPE[type], big(dim1), big(dim2), big(dim3), big(dim4)),
      type
    )
  }

  /**
   * Create a new 32-bit integer.
   * @returns {Tensor}
   */
  newI32(value) {
    return this.wrapTensor(native.ggml_new_i32(this, value))
  }

  /**
   * Create a new 32-bit float.
   * @returns {Tensor}
   */
  newF32(value) {
    return this.wrapTensor(native.ggml_new_f32(this, value))
  }
}

export class Tensor {
  type = null
  context = null

  wrapNew(obj) {
    return this.context.wrapTensor(obj, this.type)
  }

  /**
   * Set all elements to the given value.
   * @returns {Tensor}
   */
  setAll(value) {
    if (value === 0) {
      native.ggml_set_zero(this)
    } else {
      if (this.type === 'i32') {
        native.ggml_set_i32(this, value)
      } else if (this.type === 'f32') {
        native.ggml_set_f32(this, value)
      } else {
        throw new Error(`TODO`)
      }
    }
    return this
  }

  /**
   * Get value at the given index.
   * @returns {number}
   */
  get(index) {
    if (this.type === 'i32') {
      return native.ggml_get_i32_1d(this, index)
    } else if (this.type === 'f32') {
      return native.ggml_get_f32_1d(this, index)
    } else {
      throw new Error(`TODO`)
    }
  }

  /**
   * Set value at the given index.
   * @returns {Tensor}
   */
  set(index, value) {
    if (this.type === 'i32') {
      native.ggml_set_i32_1d(this, index, value)
    } else if (this.type === 'f32') {
      native.ggml_set_f32_1d(this, index, value)
    } else {
      throw new Error(`TODO`)
    }
    return this
  }

  /**
   * Add two tensors.
   * @param {Tensor} other
   * @returns {Tensor}
   */
  add(other) {
    return this.wrapNew(native.ggml_add(this.context, this, other))
  }

  /**
   * Subtract two tensors.
   * @param {Tensor} other
   * @returns {Tensor}
   */
  sub(other) {
    return this.wrapNew(native.ggml_sub(this.context, this, other))
  }

  /**
   * Multiply two tensors.
   * @param {Tensor} other
   * @returns {Tensor}
   */
  mul(other) {
    return this.wrapNew(native.ggml_mul(this.context, this, other))
  }

  /**
   * Divide two tensors.
   * @param {Tensor} other
   * @returns {Tensor}
   */
  div(other) {
    return this.wrapNew(native.ggml_div(this.context, this, other))
  }

  /**
   * Compute square of a tensor.
   * @returns {Tensor}
   */
  sqr() {
    return this.wrapNew(native.ggml_sqr(this.context, this))
  }

  /**
   * Compute square root of a tensor.
   * @returns {Tensor}
   */
  sqrt() {
    return this.wrapNew(native.ggml_sqrt(this.context, this))
  }

  /**
   * Compute sum of all elements.
   * @returns {Tensor}
   */
  sum() {
    return this.wrapNew(native.ggml_sum(this.context, this))
  }

  /**
   * Compute mean of all elements.
   * @returns {Tensor}
   */
  mean() {
    return this.wrapNew(native.ggml_mean(this.context, this))
  }

  /**
   * Compute absolute value of a tensor.
   * @returns {Tensor}
   */
  abs() {
    return this.wrapNew(native.ggml_abs(this.context, this))
  }

  /**
   * Compute sign of a tensor.
   * @returns {Tensor}
   */
  sgn() {
    return this.wrapNew(native.ggml_sgn(this.context, this))
  }

  /**
   * Negate a tensor.
   * @returns {Tensor}
   */
  neg() {
    return this.wrapNew(native.ggml_neg(this.context, this))
  }

  /**
   * Compute RELU of a tensor.
   * @returns {Tensor}
   */
  relu() {
    return this.wrapNew(native.ggml_relu(this.context, this))
  }

  /**
   * Compute GELU of a tensor.
   * @returns {Tensor}
   */
  gelu() {
    return this.wrapNew(native.ggml_gelu(this.context, this))
  }

  /**
   * Compute SiLU of a tensor.
   * @returns {Tensor}
   */
  silu() {
    return this.wrapNew(native.ggml_silu(this.context, this))
  }

  /**
   * Normalize a tensor.
   * @returns {Tensor}
   */
  norm() {
    return this.wrapNew(native.ggml_norm(this.context, this))
  }

  /**
   * Compute RMS Norm of a tensor.
   * @returns {Tensor}
   */
  rmsNorm() {
    return this.wrapNew(native.ggml_rms_norm(this.context, this))
  }

  /**
   * Build a graph for forward computation.
   * @returns {Graph}
   */
  buildForward() {
    return this.context.wrapGraph(native.ggml_build_forward(this))
  }
}

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
}

const wrap = (obj, Clz) => (Object.setPrototypeOf(obj, Clz.prototype), obj)
const big = v => BigInt(v)

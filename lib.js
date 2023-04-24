import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)

const targets = { darwin: 'macos', linux: 'linux', win32: 'windows' }
const native = require(`./zig-out/lib/ggml.${targets[process.platform]}.node`)

export const DEBUG = Symbol()

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
   * @returns {Context}
   */
  static init(opts = {}) {
    opts = {
      mem_size: BigInt(1_000_000),
      mem_buffer: null,
      no_alloc: false,
      ...opts,
    }

    return wrap(native.ggml_init(opts), Context)
  }

  // internal
  wrapTensor(obj, type) {
    return (obj.context = this), (obj.type = type), wrap(obj, Tensor)
  }

  // internal
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

  /**
   * Build a graph for forward computation.
   * @param {Tensor} tensor
   * @returns {Graph}
   */
  buildForward(tensor) {
    return this.wrapGraph(native.ggml_build_forward(tensor))
  }

  /**
   * Print GGML objects
   **/
  printObjects() {
    native.ggml_print_objects(this)
  }
}

/**
 * GGML tensor.
 */
export class Tensor {
  type
  context

  /**
   * @internal
   * @returns {Tensor} */
  wrapNew(obj) {
    return this.context.wrapTensor(obj, this.type)
  }

  /**
   * @internal
   * @returns {number} */
  nElements() {
    return native.ggml_nelements(this)
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
   * Debug repr.
   */
  get [DEBUG]() {
    return `Tensor(${this.type})`
  }
}

/**
 * Functional interface.
 * @satisfies {{ [k: string]: (...args: Tensor[]) => Tensor} }}
 */
export const F = {
  /**
   * Add two tensors.
   */
  add: (a, b) => a.wrapNew(native.ggml_add(a.context, a, b)),

  /**
   * Subtract two tensors.
   */
  sub: (a, b) => a.wrapNew(native.ggml_sub(a.context, a, b)),

  /**
   * Multiply two tensors.
   */
  mul: (a, b) => a.wrapNew(native.ggml_mul(a.context, a, b)),

  /**
   * Divide two tensors.
   */
  div: (a, b) => a.wrapNew(native.ggml_div(a.context, a, b)),

  /**
   * Compute square of a tensor.
   */
  sqr: input => input.wrapNew(native.ggml_sqr(input.context, input)),

  /**
   * Compute square root of a tensor.
   */
  sqrt: input => input.wrapNew(native.ggml_sqrt(input.context, input)),

  /**
   * Compute sum of all elements in a tensor.
   */
  sum: input => input.wrapNew(native.ggml_sum(input.context, input)),

  /**
   * Compute mean of all elements in a tensor.
   */
  mean: input => input.wrapNew(native.ggml_mean(input.context, input)),

  /**
   * Compute absolute value of a tensor.
   */
  abs: input => input.wrapNew(native.ggml_abs(input.context, input)),

  /**
   * Compute sign of a tensor.
   */
  sgn: input => input.wrapNew(native.ggml_sgn(input.context, input)),

  /**
   * Negate a tensor.
   */
  neg: input => input.wrapNew(native.ggml_neg(input.context, input)),

  /**
   * Compute RELU of a tensor.
   */
  relu: input => input.wrapNew(native.ggml_relu(input.context, input)),

  /**
   * Compute GELU of a tensor.
   */
  gelu: input => input.wrapNew(native.ggml_gelu(input.context, input)),

  /**
   * Compute SiLU of a tensor.
   */
  silu: input => input.wrapNew(native.ggml_silu(input.context, input)),

  /**
   * Normalize a tensor.
   */
  norm: input => input.wrapNew(native.ggml_norm(input.context, input)),

  /**
   * Compute RMS Norm of a tensor.
   */
  rmsNorm: input => input.wrapNew(native.ggml_rms_norm(input.context, input)),

  /**
   * Retrieve word embeddings from tensor of i32 indices.
   */
  embedding: (input, weight) => weight.wrapNew(native.ggml_get_rows(weight.context, weight, input)),
}

// debug
for (const [k, f] of Object.entries(F)) {
  F[k] = (...args) => (console.log(k, ...args.map(a => a[DEBUG] ?? a)), f(...args))
}

/**
 * Utility base class. Provides loading & debug printing.
 */
export class Model {
  /**
   * Go through all tensors in the model and collect them with their names.
   * @returns {Array<[string, Tensor]>}
   */
  collectTensors() {
    const layers = []
    const visit = (obj, prefix) => {
      switch (true) {
        case obj instanceof Tensor:
          return layers.push([prefix.slice(0, -1), obj])
        case obj instanceof Array:
          return obj.forEach((v, i) => visit(v, `${prefix}${i}.`))
        case obj instanceof Object:
          return Object.getOwnPropertyNames(obj).forEach(k => visit(obj[k], `${prefix}${k}.`))
        default:
          throw new Error('Unknown type')
      }
    }
    visit(this, '')

    return layers
  }

  /**
   * Print all tensors in the model.
   */
  print() {
    const tensors = this.collectTensors()
    const longestName = tensors.reduce((a, [k]) => Math.max(a, k.length), 0)

    tensors.forEach(([k, v], i) => {
      const num = `#${i}`.padEnd(5)
      const name = k.padEnd(longestName)
      const type = v.type.padEnd(5)
      const count = v.nElements()
      console.log(`${num} ${name} ${type} ${count}`)
    })
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
}

const wrap = (obj, Clz) => (Object.setPrototypeOf(obj, Clz.prototype), obj)
const big = v => BigInt(v)

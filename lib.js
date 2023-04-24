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
   */
  newTensor2D(type, dim1, dim2) {
    return this.newTensor(native.ggml_new_tensor_2d, type, dim1, dim2)
  }

  /**
   * Create a new 3D tensor.
   */
  newTensor3D(type, dim1, dim2, dim3) {
    return this.newTensor(native.ggml_new_tensor_3d, type, dim1, dim2, dim3)
  }

  /**
   * Create a new 4D tensor.
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
 * GGML tensor.
 */
export class Tensor {
  context

  /**
   * Tensor type.
   */
  get type() {
    return native.ggml_tensor_type(this)
  }

  /**
   * Tensor shape.
   */
  get shape() {
    return native.ggml_tensor_shape(this)
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
    return `${this.type}[${this.shape}]`
  }
}

// internal
const callTensorOp = (nativeFun, a, ...rest) => wrap(Tensor, nativeFun(a.context, a, ...rest), { context: a.context })

/**
 * Functional interface.
 * @satisfies {{ [k: string]: (...args: Tensor[]) => Tensor} }}
 */
export const F = {
  /**
   * Add two tensors.
   */
  add: (a, b) => callTensorOp(native.ggml_add, a, b),

  /**
   * Subtract two tensors.
   */
  sub: (a, b) => callTensorOp(native.ggml_sub, a, b),

  /**
   * Multiply two tensors.
   */
  mul: (a, b) => callTensorOp(native.ggml_mul, a, b),

  /**
   * Divide two tensors.
   */
  div: (a, b) => callTensorOp(native.ggml_div, a, b),

  /**
   * Compute square of a tensor.
   */
  sqr: input => callTensorOp(native.ggml_sqr, input),

  /**
   * Compute square root of a tensor.
   */
  sqrt: input => callTensorOp(native.ggml_sqrt, input),

  /**
   * Compute sum of all elements in a tensor.
   */
  sum: input => callTensorOp(native.ggml_sum, input),

  /**
   * Compute mean of all elements in a tensor.
   */
  mean: input => callTensorOp(native.ggml_mean, input),

  /**
   * Compute absolute value of a tensor.
   */
  abs: input => callTensorOp(native.ggml_abs, input),

  /**
   * Compute sign of a tensor.
   */
  sgn: input => callTensorOp(native.ggml_sgn, input),

  /**
   * Negate a tensor.
   */
  neg: input => callTensorOp(native.ggml_neg, input),

  /**
   * Compute RELU of a tensor.
   */
  relu: input => callTensorOp(native.ggml_relu, input),

  /**
   * Compute GELU of a tensor.
   */
  gelu: input => callTensorOp(native.ggml_gelu, input),

  /**
   * Compute SiLU of a tensor.
   */
  silu: input => callTensorOp(native.ggml_silu, input),

  /**
   * Normalize a tensor.
   */
  norm: input => callTensorOp(native.ggml_norm, input),

  /**
   * Compute RMS Norm of a tensor.
   */
  rmsNorm: input => callTensorOp(native.ggml_rms_norm, input),

  /**
   * Retrieve word embeddings from tensor of i32 indices.
   */
  embedding: (input, weight) => callTensorOp(native.ggml_get_rows, weight, input),

  /**
   * TODO: document this
   */
  repeat: (a, b) => callTensorOp(native.ggml_repeat, a, b),

  /**
   * Matrix multiplication.
   */
  matmul: (a, b) => callTensorOp(native.ggml_mul_mat, a, b),
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
      console.log(`${num} ${name} ${type} ${v.shape}`)
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

/** @type {<T extends new (...args: any[]) => any>(Clz: T, obj: any, ...extra: any[]) => InstanceType<T>} */
const wrap = (Clz, obj, ...extra) => (Object.setPrototypeOf(obj, Clz.prototype), Object.assign(obj, ...extra))
const big = v => BigInt(v)

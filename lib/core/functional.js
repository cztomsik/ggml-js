import { native, wrap } from './native.js'
import { Tensor } from './tensor.js'

const unaryOp = op => (/** @type Tensor */ input) => input.context.callOp(op, input)
const binaryOp = op => (/** @type Tensor */ a, /** @type Tensor */ b) => a.context.callOp(op, a, b)

/**
 * Add two tensors.
 */
export const add = binaryOp(native.ggml_add)

/**
 * Subtract two tensors.
 */
export const sub = binaryOp(native.ggml_sub)

/**
 * Multiply two tensors.
 */
export const mul = binaryOp(native.ggml_mul)

/**
 * Divide two tensors.
 */
export const div = binaryOp(native.ggml_div)

/**
 * Compute square of a tensor.
 */
export const square = unaryOp(native.ggml_sqr)

/**
 * Compute square root of a tensor.
 */
export const sqrt = unaryOp(native.ggml_sqrt)

/**
 * Compute sum of all elements in a tensor.
 */
export const sum = unaryOp(native.ggml_sum)

/**
 * Compute mean of all elements in a tensor.
 */
export const mean = unaryOp(native.ggml_mean)

/**
 * Compute absolute value of a tensor.
 */
export const abs = unaryOp(native.ggml_abs)

/**
 * Compute sign of a tensor.
 */
export const sgn = unaryOp(native.ggml_sgn)

/**
 * Negate a tensor.
 */
export const neg = unaryOp(native.ggml_neg)

/**
 * Compute RELU of a tensor.
 */
export const relu = unaryOp(native.ggml_relu)

/**
 * Compute GELU of a tensor.
 */
export const gelu = unaryOp(native.ggml_gelu)

/**
 * Compute SiLU of a tensor.
 */
export const silu = unaryOp(native.ggml_silu)

/**
 * Normalize a tensor.
 */
export const norm = unaryOp(native.ggml_norm)

/**
 * Compute exponential of a tensor.
 */
export const exp = unaryOp(native.ggml_exp)

/**
 * Compute sigmoid of a tensor.
 */
export const sigmoid = unaryOp(native.ggml_sigmoid)

/**
 * Compute softmax of a tensor.
 */
export const softmax = unaryOp(native.ggml_soft_max)

/**
 * Get the index of the maximum value.
 * @returns {Tensor}
 */
export const argmax = unaryOp(native.ggml_argmax)

/**
 * Compute max of two tensors.
 */
export const max = binaryOp(native.ggml_max)

/**
 * Compute RMS Norm of a tensor.
 */
export const rmsNorm = unaryOp(native.ggml_rms_norm)

/**
 * Retrieve word embeddings from tensor of i32 indices.
 * @type {(a: Tensor, b: Tensor) => Tensor}
 */
export const embedding = (input, weight) => input.context.callOp(native.ggml_get_rows, weight, input)

/**
 * TODO: document this
 */
export const scale = binaryOp(native.ggml_scale)

/**
 * TODO: document this
 */
export const repeat = (a, b) => a.context.callOp(native.ggml_repeat, a, b)

/**
 * TODO: document this
 */
export const rope = (a, n_past, n_rot, mode) => a.context.callOp(native.ggml_rope, a, n_past, n_rot, mode)

/**
 * TODO: document this
 */
export const flashAttention = (q, k, v, masked) => q.context.callOp(native.ggml_flash_attn, q, k, v, masked)

/**
 * Matrix multiplication.
 * @type {(a: Tensor, b: Tensor) => Tensor}
 */
export const matmul = (a, b) => b.context.callOp(native.ggml_mul_mat, a, b)

/**
 * Copy data from one tensor to another. Returns a view of the current tensor
 * with the copy operation. Note that you still need to call
 * `graph.buildForwardExpand()` on the returned tensor.
 */
export const cpy = binaryOp(native.ggml_cpy)

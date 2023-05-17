import assert from 'assert'
import { Context, GGML_TYPE_NAMES } from './context.js'
import { native } from './native.js'

/**
 * GGML tensor.
 */
export class Tensor {
  /** @type {Context} */
  context

  /**
   * Tensor type.
   */
  get type() {
    return GGML_TYPE_NAMES.get(native.ggml_tensor_type(this))
  }

  /**
   * Tensor shape.
   */
  get shape() {
    // ggml reports shape in reverse order
    return native.ggml_tensor_shape(this).map(Number).reverse()
  }

  /**
   * Set all elements to the given value.
   * @returns {Tensor}
   */
  setAll(value) {
    if (this.type === 'i32') {
      native.ggml_set_i32(this, value)
    } else if (this.type === 'f32') {
      native.ggml_set_f32(this, value)
    } else {
      throw new Error(`TODO`)
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
   * @param {number} index
   * @param {number} value
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
   * Get a view of the tensor.
   * @param {...number} dims
   * @returns {Tensor}
   */
  view(...dims) {
    assert(dims.length === 1, 'It is only possible to view a tensor as a 1D tensor for now')

    if (dims[0] === -1) {
      dims[0] = this.shape.reduce((a, b) => a * b, 1)
    }

    return this.context.callOp(native.ggml_view_1d, this, BigInt(dims[0]), BigInt(0))
  }

  /**
   * Reshape the tensor.
   * @param {...number} dims
   * @returns {Tensor}
   */
  reshape(...dims) {
    assert(dims.length <= 4, 'GGML only supports up to 4 dimensions')
    const fun = native[`ggml_reshape_${dims.length}d`]
    return this.context.callOp(fun, this, ...dims.map(BigInt))
  }

  /**
   * Transpose the tensor.
   * @returns {Tensor}
   */
  transpose() {
    return this.context.callOp(native.ggml_transpose, this)
  }

  /**
   * Split a tensor into chunks of the given size.
   * @param {number} size
   * @returns {Tensor[]}
   */
  split(size) {
    assert(this.shape.length === 1, 'Splitting is only supported for 1D tensors')

    const numChunks = Math.ceil(this.shape[0] / size)

    return Array.from(Array(numChunks), (_, i) =>
      this.context.callOp(native.ggml_view_1d, this, BigInt(size), BigInt(i * size))
    )
  }

  argmax() {
    return native.ggml_argmax(this)
  }

  /**
   * Debug string.
   */
  toString() {
    return `${this.type}[${this.shape}]`
  }
}

import { native } from './native.js'

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
   * Copy data from another tensor.
   */
  copyFrom(other) {
    native.ggml_memcpy(this, other)
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

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
   * @returns {string}
   */
  get type() {
    return GGML_TYPE_NAMES.get(native.ggml_tensor_type(this))
  }

  /**
   * Tensor shape (in reverse order).
   * @returns {number[]}
   */
  get shape() {
    return native.ggml_tensor_shape(this).map(Number)
  }

  /**
   * Size of one element in bytes.
   * @returns {BigInt}
   */
  get elementSize() {
    return native.ggml_element_size(this)
  }

  /**
   * Set all elements to the given value.
   * @param {number} value
   * @returns {Tensor}
   */
  setAll(value) {
    if (value === 0) {
      native.ggml_set_zero(this)
    } else if (this.type === 'i32') {
      native.ggml_set_i32(this, value)
    } else {
      native.ggml_set_f32(this, value)
    }

    return this
  }

  /**
   * Get value at the given index.
   * @param {number} index
   * @returns {number}
   */
  get(index) {
    if (this.type === 'i32') {
      return native.ggml_get_i32_1d(this, index)
    }

    return native.ggml_get_f32_1d(this, index)
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
    } else {
      native.ggml_set_f32_1d(this, index, value)
    }

    return this
  }

  /**
   * Get a view of the tensor (shape in reverse order).
   * @param {...number} dims
   * @returns {Tensor}
   */
  view(...dims) {
    assert(dims.length <= 3, 'It is only possible to view up to 3 dimensions for now')

    if (dims.length === 1 && dims[0] === -1) {
      dims[0] = this.shape.reduce((a, b) => a * b, 1)
    }

    switch (dims.length) {
      case 1:
        return this.context.callOp(native.ggml_view_1d, this, BigInt(dims[0]), BigInt(0))
      case 2:
        return this.context.callOp(native.ggml_view_2d, this, BigInt(dims[0]), BigInt(dims[1]), BigInt(0), BigInt(0))
      case 3:
        return this.context.callOp(
          native.ggml_view_3d,
          this,
          BigInt(dims[0]),
          BigInt(dims[1]),
          BigInt(dims[2]),
          BigInt(0),
          BigInt(0),
          BigInt(0)
        )
      default:
        throw new Error(`TODO`)
    }
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
   * Permute the tensor.
   * @param {number} axis0
   * @param {number} axis1
   * @param {number} axis2
   * @param {number} axis3
   * @returns {Tensor}
   */
  permute(axis0, axis1, axis2, axis3) {
    return this.context.callOp(native.ggml_permute, this, axis0, axis1, axis2, axis3)
  }

  /**
   * Transpose the tensor.
   * @returns {Tensor}
   */
  transpose() {
    return this.context.callOp(native.ggml_transpose, this)
  }

  /**
   * Split a tensor into chunks of the given size along the first dimension.
   * @param {number} chunkSize
   * @returns {Tensor[]}
   */
  split(chunkSize) {
    assert(this.shape.length <= 2, 'Splitting is only supported for tensors with up to 2 dimensions')

    const numChunks = Math.ceil(this.shape[0] / chunkSize)
    const bytes = Number(this.elementSize)
    const offset = chunkSize * bytes

    return Array.from(Array(numChunks), (_, i) => {
      switch (this.shape.length) {
        case 1:
          return this.context.callOp(native.ggml_view_1d, this, BigInt(chunkSize), BigInt(i * offset))
        case 2:
          return this.context.callOp(
            native.ggml_view_2d,
            this,
            BigInt(chunkSize),
            BigInt(this.shape[1]),
            BigInt(this.shape[0] * bytes),
            BigInt(i * offset)
          )
      }
    })
  }

  /**
   * Make the tensor contiguous in memory.
   * @returns {Tensor}
   */
  contiguous() {
    return this.context.callOp(native.ggml_cont, this)
  }

  /**
   * Debug string.
   */
  toString() {
    return `${this.type}[${this.shape}]`
  }
}

import { native } from './native.js'
import { Tensor } from './tensor.js'
import { Module } from './module.js'
import * as F from './functional.js'

/**
 * Base class for all models.
 */
export class Model extends Module {
  constructor(context) {
    // @ts-ignore
    super({})
    this.parentModule = null
    this.context = context
  }

  loadFromFile(path) {
    const header = JSON.parse(native.safetensors_read_header(path), (k, v) => (typeof v === 'number' ? BigInt(v) : v))
    const tensors = this.collectTensors()
    const keysLeft = new Set(Object.keys(header))
    const mappings = []

    const extraTensors = tensors.filter(([k, _]) => !keysLeft.has(k)).map(v => v[0])
    if (extraTensors.length) {
      throw new Error(`Model contains tensors that are not in the file: ${extraTensors.join(', ')}`)
    }

    // prepare mappings
    for (const [name, tensor] of tensors) {
      if (keysLeft.has(name)) {
        const { dtype, shape, data_offsets } = header[name]

        if (tensor.type !== dtype.toLowerCase()) {
          throw new Error(`Tensor ${name} has type ${tensor.type} but the file contains ${dtype}`)
        }

        if (!tensor.shape.every((v, i) => v === shape[i])) {
          throw new Error(`Tensor ${name} has shape ${tensor.shape} but the file contains ${shape}`)
        }

        mappings.push({ tensor, start: data_offsets[0], end: data_offsets[1] })
        keysLeft.delete(name)
      }
    }

    // ignore optional __metadata__ key
    keysLeft.delete('__metadata__')

    if (keysLeft.size > 0) {
      throw new Error(`Could not find the following tensors in the model: ${Array.from(keysLeft).join(', ')}`)
    }

    // load tensor data
    native.safetensors_mmap(path, mappings)
  }

  /**
   * Go through all tensors in the model and collect them with their names.
   * @returns {Array<[string, Tensor]>}
   */
  collectTensors() {
    const layers = []
    const visit = (obj, prefix) => {
      switch (true) {
        case obj === null:
          return
        case obj instanceof Tensor:
          return layers.push([prefix.slice(0, -1), obj])
        case obj instanceof Array:
          return obj.forEach((v, i) => visit(v, `${prefix}${i}.`))
        case obj instanceof Object:
          return Object.getOwnPropertyNames(obj)
            .filter(k => k !== 'parentModule')
            .forEach(k => visit(obj[k], `${prefix}${k}.`))
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
 * Base class for all causal language models.
 */
export class CausalLM extends Model {
  /**
   * @param {Tensor} x - Input tensor.
   * @param {Tensor[]} updates - Array of tensors that should be updated after the forward pass.
   * @returns {Tensor} Output tensor.
   * @abstract
   */
  forward(x, updates = []) {
    throw new Error('To be implemented by subclasses')
  }

  *generate(input, n = 100) {
    const updates = []
    const x = this.context.newTensor1D('i32', 1)
    const out = F.softmax(this.forward(x, updates))
    const [_, N_VOCAB] = out.shape
    const graph = this.context.buildForward(out)
    const update = () => updates.forEach(([dst, src]) => dst.copyFrom(src))

    // make sure the update tensors will be computed
    updates.forEach(([_, src]) => graph.forwardExpand(src))

    // feed the input one by one
    for (const v of input) {
      x.set(0, v)
      graph.compute()
      update()
    }

    // generate
    for (let i = 0; i < n; i++) {
      // TODO: sampling
      let res = out.argmax()

      yield res

      x.set(0, res)
      graph.compute()
      update()
    }
  }
}

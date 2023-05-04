import { native } from './native.js'
import { Context } from './context.js'
import { Tensor } from './tensor.js'
import { Module } from './module.js'

/**
 * Base class for all models.
 */
export class Model extends Module {
  /**
   * Create an empty model.
   * @param {Context} context
   * @param {Object} hparams
   */
  constructor(context, hparams = {}) {
    // @ts-expect-error
    super({})
    this.parentModule = null
    this.context = context
  }

  /**
   * Load a model from a file.
   * @type {<T extends typeof Model>(this: T, path: string) => InstanceType<T>}
   */
  static loadFromFile(path) {
    const safetensors = native.safetensors_open(path)
    const { __metadata__, ...header } = JSON.parse(native.safetensors_header(safetensors))

    const context = Context.init({ no_alloc: true })
    const model = new this(context, this.loadHparams(header))

    const tensors = model.collectTensors()
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

        mappings.push({ tensor, start: BigInt(data_offsets[0]), end: BigInt(data_offsets[1]) })
        keysLeft.delete(name)
      }
    }

    if (keysLeft.size > 0) {
      throw new Error(`Could not find the following tensors in the model: ${Array.from(keysLeft).join(', ')}`)
    }

    // load tensor data
    native.safetensors_load_tensors(safetensors, mappings)

    // @ts-expect-error
    return model
  }

  /**
   * Load model hyperparameters from the safetensors header.
   * @param {Object} header
   * @returns {Object}
   * @abstract
   */
  static loadHparams(header) {
    return {}
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

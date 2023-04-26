import { native } from './native.js'
import { Tensor } from './tensor.js'
import { Module } from './module.js'

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
    const header = JSON.parse(native.safetensors_read_header(path))

    const tensors = this.collectTensors()
    const keysLeft = new Set(Object.keys(header))
    const mappings = []

    const extraTensors = tensors.filter(([k, _]) => !keysLeft.has(k)).map(v => v[0])
    if (extraTensors.length) {
      throw new Error(`Model contains tensors that are not in the header: ${extraTensors.join(', ')}`)
    }

    // prepare mappings
    for (const [name, tensor] of tensors) {
      if (keysLeft.has(name)) {
        const [start, end] = header[name].data_offsets.map(BigInt)
        mappings.push({ tensor, start, end })
        keysLeft.delete(name)
      }
    }

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
 * Base class for all LLMs.
 */
export class LLM extends Model {
  constructor(context, tokenizer) {
    super(context)
    this.tokenizer = tokenizer
  }

  generate(input, options = {}) {
    const { maxLen = 512, temperature = 1, topK = 0, topP = 0 } = options
    const output = native.ggml_llm_generate(this.context, this, input, maxLen, temperature, topK, topP)
    return this.tokenizer.decode(output)
  }
}

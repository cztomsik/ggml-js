> This project is temporarily **on hold**.\
> I am now working on **[Ava, Personal Language Server](http://www.avapls.com/)**, a GUI app for running LLMs.

# ggml-js

JavaScript bindings for the [GGML](https://github.com/ggerganov/ggml) library, a
fast and lightweight tensor/machine-learning library implemented in C.

[RWKV example](https://github.com/cztomsik/ggml-js/blob/main/examples/rwkv.js)

https://user-images.githubusercontent.com/3526922/236536800-fca4d729-3479-471a-aea5-2d0ae5df3fdf.mov

## Installation

You can install ggml-js via npm:

```bash
npm install ggml-js
```

## Basic Usage

Here's an example of how to use ggml-js in your JavaScript code:

```js
import { Context, F } from 'ggml-js/core'

// Create context, two 1D tensors and multiply them
const ctx = Context.init()
const a = ctx.newTensor1D('f32', 1)
const b = ctx.newTensor1D('f32', 1)
const ab = F.mul(a, b)

// Build the computation graph
const graph = ctx.buildForward(ab)

// Set values & compute the graph
a.set(0, 1.5)
b.set(0, 2)
graph.compute()

// Get result
console.log(ab.get(0))
```

## Advanced Usage

ggml-js also provides modules for working with pre-trained models and tokenizers. Here's an example of how to use the RWKV model and BPETokenizer:

```js
import { RWKV } from 'ggml-js/llms'
import { BPETokenizer } from 'ggml-js/tokenizers'

// see examples/rwkv.js for full example
const model = RWKV.loadFromFile(...)
const tokenizer = BPETokenizer.loadFromFile(...)

for (const t of model.generate(tokenizer.encode('Hello world!'))) {
  process.stdout.write(tokenizer.decodeOne(t))
}
```

## Building From Source

If you want to build ggml-js from source, you can clone the repository and run the following commands:

```bash
zig build
```

## License

This project is licensed under the MIT License.

This project bundles [GGML](https://github.com/ggerganov/ggml) library by **Georgi Gerganov**, which is also licensed under the MIT License.

# ggml-js

JavaScript bindings for the [ggml](https://github.com/ggerganov/ggml) library.

[RWKV example](https://github.com/cztomsik/ggml-js/blob/main/examples/rwkv.js)
https://user-images.githubusercontent.com/3526922/235147373-1d3b7205-8c4c-4654-940c-78a4baeb4fad.mov

## Installation

You can install ggml-js via npm:

```bash
npm install ggml-js
```

## Usage

Here's an example of how to use ggml-js in your JavaScript code:

```js
import { Context, F } from 'ggml-js'

// Initialize the context
const ctx = Context.init()

// Create new 1D tensors a and b
const a = ctx.newTensor1D('f32', 1)
const b = ctx.newTensor1D('f32', 1)

// Perform element-wise multiplication of a and b
const ab = F.mul(a, b)

// Build the forward computation graph
const graph = ctx.buildForward(ab)

// Print the graph structure
graph.print()

// Set values for tensors a and b
a.set(0, 1.5)
b.set(0, 2)

// Compute the graph
graph.compute()

// Print the result of the multiplication
console.log(ab.get(0))
```

## License

This project is licensed under the MIT License.

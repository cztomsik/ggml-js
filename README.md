# ggml-js

JavaScript bindings for the [ggml](https://github.com/ggerganov/ggml) library.

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

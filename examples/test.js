import { Context, F } from 'ggml-js/core'

// Initialize the context
const ctx = Context.init()

// Create 1D tensors and multiply them
const a = ctx.newTensor1D('f32', 1)
const b = ctx.newTensor1D('f32', 1)
const ab = F.mul(a, b)

// Build the forward computation graph
const graph = ctx.buildForward(ab)

// Print the graph structure
graph.print()

// Set values & compute the graph
a.set(0, 1.5)
b.set(0, 2)
graph.compute()

// Get result
console.log(ab.get(0))

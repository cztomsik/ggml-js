import { Context } from 'ggml-js';

// Initialize the context
const ctx = Context.init();

// Create new 1D tensors a and b
const a = ctx.newTensor1D('f32', 1);
const b = ctx.newTensor1D('f32', 1);

// Perform element-wise multiplication of a and b
const ab = a.mul(b);

// Build the forward computation graph
const graph = ab.buildForward();

// Print the graph structure
graph.print();

// Set values for tensors a and b
a.set(0, 1.5);
b.set(0, 2);

// Compute the graph
graph.compute();

// Print the result of the multiplication
console.log(ab.get(0));

// see examples/rwkv.js for instructions on how to use this script

import { RWKV } from 'ggml-js/llms'
import { BPETokenizer } from 'ggml-js/tokenizers'

// Load the model and tokenizer
const model = RWKV.loadFromFile(process.argv[2])
const tokenizer = BPETokenizer.loadFromFile(process.argv[3])

// Taken from SQuAD 2.0
const context = `Steam engines are external combustion engines, where the working fluid is separate from the combustion products. Non-combustion heat sources such as solar power, nuclear power or geothermal energy may be used. The ideal thermodynamic cycle used to analyze this process is called the Rankine cycle. In the cycle, water is heated and transforms into steam within a boiler operating at a high pressure. When expanded through pistons or turbines, mechanical work is done. The reduced-pressure steam is then condensed and pumped back into the boiler.`

// Raven prompt
const prompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.
# Instruction:
Write at least 5 questions that can be answered by the following text:
${context}
# Response:`

// Generate text and print it one token at a time
for (const t of model.generate(tokenizer.encode(prompt), {
  max_tokens: 500,
  stop_token: 0,
  // SEED=1
  // typical: 0.9,
  temperature: 1.2,
  top_k: 50_000,
  top_p: 0.5,
})) {
  process.stdout.write(tokenizer.decodeOne(t))
}

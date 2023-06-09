import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)

// TODO: arch

const targets = {
  darwin: 'macos',
  linux: 'linux',
  win32: 'windows',
}

export const native = require(`../../zig-out/lib/ggml.${targets[process.platform]}.node`)

/** @type {<T extends new (...args: any[]) => any>(Clz: T, obj: any, ...extra: any[]) => InstanceType<T>} */
export const wrap = (Clz, obj, ...extra) => (Object.setPrototypeOf(obj, Clz.prototype), Object.assign(obj, ...extra))

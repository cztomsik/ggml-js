import { Context } from './context.js'

/**
 * Base module class.
 */
export class Module {
  /** @type {Module} */
  parentModule
  /** @type {Context} */
  context

  /**
   * Create a new module.
   * @param {Module} parentModule
   */
  constructor(parentModule) {
    this.parentModule = parentModule
    this.context = this.parentModule.context
  }
}

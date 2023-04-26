/**
 * Base module class.
 */
export class Module {
  /**
   * Create a new module.
   * @param {Module} parentModule
   */
  constructor(parentModule) {
    this.parentModule = parentModule
    this.context = this.parentModule.context
  }
}

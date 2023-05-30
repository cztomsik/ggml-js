const std = @import("std");
const napigen = @import("napigen");
const ggml = @import("ggml.zig");
const safetensors = @import("safetensors.zig");
const sampling = @import("sampling.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
pub const allocator = gpa.allocator();

comptime {
    napigen.defineModule(initModule);
}

fn initModule(js: *napigen.JsContext, exports: napigen.napi_value) !napigen.napi_value {
    @setEvalBranchQuota(100_000);
    inline for (comptime std.meta.declarations(ggml)) |d| {
        if (comptime !std.ascii.startsWithIgnoreCase(d.name, "ggml_") or
            std.mem.eql(u8, d.name, "GGML_RESTRICT") or
            std.mem.eql(u8, d.name, "GGML_ASSERT") or
            std.mem.eql(u8, d.name, "ggml_internal_get_quantize_fn") or
            @TypeOf(@field(ggml, d.name)) == type) continue;

        try js.exportOne(exports, d.name, @field(ggml, d.name));
    }

    // extensions
    try js.exportAll(exports, safetensors);
    try js.exportAll(exports, sampling);

    return exports;
}

pub fn napigenWrite(js: *napigen.JsContext, value: anytype) !napigen.napi_value {
    return switch (@TypeOf(value)) {
        // alloc a pointer and write the value to it
        safetensors.SafeTensors, ggml.ggml_cgraph => {
            var ptr = napigen.allocator.create(@TypeOf(value)) catch unreachable;
            ptr.* = value;
            return js.write(ptr);
        },
        else => js.defaultWrite(value),
    };
}

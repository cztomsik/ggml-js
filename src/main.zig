const std = @import("std");
const napigen = @import("napigen");
const ggml = @import("ggml.zig");
const safetensors = @import("safetensors.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
pub const allocator = gpa.allocator();

comptime {
    napigen.defineModule(initModule);
}

fn initModule(js: *napigen.JsContext, exports: napigen.napi_value) !napigen.napi_value {
    @setEvalBranchQuota(100_000);
    inline for (comptime std.meta.declarations(ggml)) |d| {
        // constants
        if (comptime std.mem.startsWith(u8, d.name, "GGML_")) {
            if (comptime std.mem.eql(u8, d.name, "GGML_RESTRICT")) continue;

            try js.setNamedProperty(exports, d.name ++ "", try js.write(@field(ggml, d.name)));
        }

        // functions
        if (comptime std.mem.startsWith(u8, d.name, "ggml_")) {
            if (comptime std.mem.eql(u8, d.name, "ggml_internal_get_quantize_fn")) continue;

            const T = @TypeOf(@field(ggml, d.name));

            if (@typeInfo(T) == .Fn) {
                try js.setNamedProperty(exports, d.name ++ "", try js.createNamedFunction(d.name ++ "", @field(ggml, d.name)));
            }
        }
    }

    // extensions
    inline for (comptime std.meta.declarations(safetensors)) |d| {
        if (comptime std.mem.startsWith(u8, d.name, "safetensors_")) {
            try js.setNamedProperty(exports, d.name ++ "", try js.createNamedFunction(d.name ++ "", @field(safetensors, d.name)));
        }
    }

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

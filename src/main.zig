const std = @import("std");
const napigen = @import("napigen");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

const ggml = @cImport({
    @cUndef("__ARM_NEON");
    @cInclude("ggml.h");
});

comptime {
    napigen.defineModule(initModule);
}

fn initModule(js: *napigen.JsContext, exports: napigen.napi_value) !napigen.napi_value {
    @setEvalBranchQuota(100_000);
    inline for (comptime std.meta.declarations(ggml)) |d| {
        // constants
        if (comptime std.mem.startsWith(u8, d.name, "GGML_")) {
            if (comptime std.mem.eql(u8, d.name, "GGML_RESTRICT")) continue;

            try js.setNamedProperty(exports, "" ++ d.name, try js.write(@field(ggml, d.name)));
        }

        // functions
        if (comptime std.mem.startsWith(u8, d.name, "ggml_")) {
            if (comptime std.mem.eql(u8, d.name, "ggml_internal_get_quantize_fn")) continue;

            const T = @TypeOf(@field(ggml, d.name));

            if (@typeInfo(T) == .Fn) {
                try js.setNamedProperty(exports, "" ++ d.name, try js.createNamedFunction("" ++ d.name, @field(ggml, d.name)));
            }
        }
    }

    // extensions
    inline for (.{ "ggml_tensor_type", "ggml_tensor_shape" }) |name| {
        try js.setNamedProperty(exports, name, try js.createNamedFunction(name, @field(@This(), name)));
    }

    return exports;
}

pub fn napigenWrite(js: *napigen.JsContext, value: anytype) !napigen.napi_value {
    return switch (@TypeOf(value)) {
        // whenever we return a ggml_cgraph, we don't want to return a copy of it, but the pointer to it
        ggml.ggml_cgraph => {
            var ptr = napigen.allocator.create(ggml.ggml_cgraph) catch unreachable;
            ptr.* = value;
            return js.write(ptr);
        },
        else => js.defaultWrite(value),
    };
}

pub fn ggml_tensor_type(tensor: *ggml.ggml_tensor) []const u8 {
    return std.mem.span(ggml.ggml_type_name(tensor.type));
}

pub fn ggml_tensor_shape(tensor: *ggml.ggml_tensor) []const i64 {
    return tensor.ne[0..@intCast(usize, tensor.n_dims)];
}

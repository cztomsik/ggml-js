const std = @import("std");
const napigen = @import("napigen");

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
                try js.setNamedProperty(exports, "" ++ d.name, try js.createFunction(@field(ggml, d.name)));
            }
        }
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

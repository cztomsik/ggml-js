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
    inline for (.{
        "ggml_tensor_type",
        "ggml_tensor_shape",
        "ggml_max",
        "ggml_exp",
        "ggml_sigmoid",
        "ggml_one_minus_x",
        "ggml_memcpy",
        "safetensors_read_header",
        "safetensors_mmap",
    }) |name| {
        try js.setNamedProperty(exports, name, try js.createNamedFunction(name, @field(@This(), name)));
    }

    return exports;
}

pub fn napigenWrite(js: *napigen.JsContext, value: anytype) !napigen.napi_value {
    return switch (@TypeOf(value)) {
        // alloc a pointer and write the value to it
        MmappedFile, ggml.ggml_cgraph => {
            var ptr = napigen.allocator.create(@TypeOf(value)) catch unreachable;
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

pub fn ggml_max(context: *ggml.ggml_context, a: *ggml.ggml_tensor, b: *ggml.ggml_tensor) *ggml.ggml_tensor {
    return ggml.ggml_map_binary_f32(context, a, b, &max);
}

fn max(cols: c_int, dest: [*c]f32, a: [*c]const f32, b: [*c]const f32) callconv(.C) void {
    for (0..@intCast(usize, cols)) |i| {
        dest[i] = @max(a[i], b[i]);
    }
}

pub fn ggml_exp(context: *ggml.ggml_context, tensor: *ggml.ggml_tensor) *ggml.ggml_tensor {
    return ggml.ggml_map_unary_f32(context, tensor, &exp);
}

fn exp(cols: c_int, dest: [*c]f32, src: [*c]const f32) callconv(.C) void {
    for (0..@intCast(usize, cols)) |i| {
        dest[i] = @exp(src[i]);
    }
}

pub fn ggml_sigmoid(context: *ggml.ggml_context, tensor: *ggml.ggml_tensor) *ggml.ggml_tensor {
    return ggml.ggml_map_unary_f32(context, tensor, &sigmoid);
}

fn sigmoid(cols: c_int, dest: [*c]f32, src: [*c]const f32) callconv(.C) void {
    for (0..@intCast(usize, cols)) |i| {
        dest[i] = 1 / (1 + @exp(-src[i]));
    }
}

pub fn ggml_one_minus_x(context: *ggml.ggml_context, tensor: *ggml.ggml_tensor) *ggml.ggml_tensor {
    return ggml.ggml_map_unary_f32(context, tensor, &one_minus_x);
}

fn one_minus_x(cols: c_int, dest: [*c]f32, src: [*c]const f32) callconv(.C) void {
    for (0..@intCast(usize, cols)) |i| {
        dest[i] = 1 - src[i];
    }
}

pub fn ggml_memcpy(dest: *ggml.ggml_tensor, src: *ggml.ggml_tensor) void {
    for (0..ggml.ggml_nbytes(dest)) |i| {
        @ptrCast([*c]u8, dest.data.?)[i] = @ptrCast([*c]u8, src.data.?)[i];
    }
}

pub fn safetensors_read_header(path: []const u8) ![]const u8 {
    const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    defer file.close();

    var reader = file.reader();
    const header_size = try reader.readIntLittle(u64);

    var buf = try allocator.alloc(u8, header_size);
    _ = try reader.read(buf);

    // TODO: allocator.free(buf);
    return buf;
}

pub fn safetensors_mmap(path: []const u8, mappings: []const TensorMapping) !MmappedFile {
    // open the file and mmap it
    const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    const memory = try std.os.mmap(
        null,
        try file.getEndPos(),
        std.os.PROT.READ,
        std.os.MAP.PRIVATE,
        file.handle,
        0,
    );

    // go through the mappings and set the data pointers
    for (mappings, 0..) |m, i| {
        if (ggml.ggml_nbytes(m.tensor) != (m.end - m.start)) {
            std.debug.print("tensor #{d} {s}{any} has size {d} it should be {d}\n", .{
                i,
                ggml.ggml_type_name(m.tensor.type),
                ggml_tensor_shape(m.tensor),
                ggml.ggml_nbytes(m.tensor),
                (m.end - m.start),
            });
            return error.TensorSizeMismatch;
        }

        m.tensor.data = memory[m.start..m.end].ptr;
    }

    return .{ .file = file, .memory = memory };
}

// TODO: call this from JS (when the tensors are no longer needed)
pub fn safetensors_mmap_close(file: MmappedFile) !void {
    try std.os.munmap(file.memory);
    file.file.close();
}

const TensorMapping = struct { tensor: *ggml.ggml_tensor, start: usize, end: usize };
const MmappedFile = struct { file: std.fs.File, memory: []const u8 };

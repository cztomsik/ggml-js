const std = @import("std");

const ggml = @cImport({
    // NEON is supported, this is just for the @cImport
    @cUndef("__ARM_NEON");
    @cDefine("GGML_ASSERT(x)", "(x || abort())");
    @cInclude("stdlib.h");
    @cInclude("ggml.h");
});

// re-export everything
pub usingnamespace ggml;

pub fn ggml_tensor_type(tensor: *ggml.ggml_tensor) ggml.ggml_type {
    return tensor.type;
}

pub fn ggml_tensor_shape(tensor: *ggml.ggml_tensor) []const i64 {
    return tensor.ne[0..@intCast(usize, tensor.n_dims)];
}

pub fn ggml_argmax(tensor: *ggml.ggml_tensor) !u32 {
    if (tensor.n_dims > 2 or (tensor.n_dims == 2 and tensor.ne[1] != 1)) {
        return error.NotImplemented;
    }

    var ptr = ggml.ggml_get_data_f32(tensor);
    var data = ptr[0..@intCast(usize, tensor.ne[0])];
    return @intCast(u32, std.sort.argMax(f32, data, {}, std.sort.asc(f32)) orelse 0);
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

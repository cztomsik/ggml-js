const std = @import("std");
const ggml = @import("ggml.zig");

pub const SafeTensors = struct {
    file: std.fs.File,
    memory: []const u8,
    header: []u8,
    data: []u8,
};

pub const TensorMapping = struct {
    tensor: *ggml.ggml_tensor,
    type: ggml.ggml_type,
    start: usize,
    end: usize,
};

pub fn safetensors_open(path: []const u8) !SafeTensors {
    // open the file and mmap it
    const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    const memory = try std.posix.mmap(
        null,
        try file.getEndPos(),
        std.posix.PROT.READ,
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );

    var reader = file.reader();
    const header_start = @sizeOf(u64);
    const header_size: usize = @intCast(try reader.readInt(u64, .little));

    return .{
        .file = file,
        .memory = memory,
        .header = memory[header_start .. header_start + header_size],
        .data = memory[header_start + header_size ..],
    };
}

pub fn safetensors_header(file: *SafeTensors) []const u8 {
    return file.header;
}

pub fn safetensors_load_tensors(file: *SafeTensors, mappings: []const TensorMapping) !void {
    // go through the mappings and set the data pointers
    for (mappings, 0..) |m, i| {
        // create a copy with correct data type
        var copy = m.tensor.*;
        copy.type = m.type;
        copy.data = file.data[m.start..m.end].ptr;
        copy.nb = std.mem.zeroes(@TypeOf(copy.nb));
        copy.nb[0] = ggml.ggml_type_size(m.type);
        copy.nb[1] = copy.nb[0] * @as(usize, @intCast(@divTrunc(copy.ne[0], @as(i64, ggml.ggml_blck_size(m.type)))));
        for (2..copy.nb.len) |bi| {
            copy.nb[bi] = copy.nb[bi - 1] * @as(usize, @intCast(copy.ne[bi - 1]));
        }

        if (ggml.ggml_nbytes(&copy) != (m.end - m.start)) {
            std.debug.print("tensor #{d} {s}{any} has size {d} it should be {d}\n", .{
                i,
                ggml.ggml_type_name(m.tensor.type),
                ggml.ggml_tensor_shape(m.tensor),
                ggml.ggml_nbytes(m.tensor),
                (m.end - m.start),
            });
            return error.TensorSizeMismatch;
        }

        // write the copy back
        m.tensor.* = copy;
    }
}

// TODO: call this from JS (when none of the tensors are used anymore)
// pub fn safetensors_mmap_close(file: *SafeTensors) void {
//     std.os.munmap(file.memory);
//     file.file.close();
// }

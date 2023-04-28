// TODO: this is just PoC, maybe we could change the API to something like:
//
//       safetensors_open(path: []const u8) !SafetensorsFile
//       safetensors_close(file: SafetensorsFile) !void
//       safetensors_get_header(file: SafetensorsFile) ![]const u8
//       ...
//
//       that way we could have just a single handle to the file and we could
//       avoid that extra allocation/freeing for the header

const std = @import("std");
const ggml = @import("ggml.zig");
const allocator = @import("root").allocator;

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

    var reader = file.reader();
    const offset = @intCast(usize, try reader.readIntLittle(u64)) + @sizeOf(u64);

    // go through the mappings and set the data pointers
    for (mappings, 0..) |m, i| {
        if (ggml.ggml_nbytes(m.tensor) != (m.end - m.start)) {
            std.debug.print("tensor #{d} {s}{any} has size {d} it should be {d}\n", .{
                i,
                ggml.ggml_type_name(m.tensor.type),
                ggml.ggml_tensor_shape(m.tensor),
                ggml.ggml_nbytes(m.tensor),
                (m.end - m.start),
            });
            return error.TensorSizeMismatch;
        }

        m.tensor.data = memory[offset + m.start .. offset + m.end].ptr;
    }

    return .{ .file = file, .memory = memory };
}

// TODO: call this from JS (when the tensors are no longer needed)
// pub fn safetensors_mmap_close(file: MmappedFile) void {
//     std.os.munmap(file.memory);
//     file.file.close();
// }

pub const TensorMapping = struct { tensor: *ggml.ggml_tensor, start: usize, end: usize };
pub const MmappedFile = struct { file: std.fs.File, memory: []align(std.mem.page_size) const u8 };

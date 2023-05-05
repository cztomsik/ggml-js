const std = @import("std");
const root = @import("root");
const ggml = @import("ggml.zig");

var prng = std.rand.DefaultPrng.init(123);
var random = prng.random();

pub fn sample_top_p(probs_tensor: *ggml.ggml_tensor, top_p: f32, temperature: f32) !u32 {
    var ptr = @ptrCast([*]f32, @alignCast(@alignOf(f32), probs_tensor.data));
    var probs = ptr[0..@intCast(usize, probs_tensor.ne[0])];

    if (top_p < 1) {
        var sorted_probs = try root.allocator.dupe(f32, probs);
        defer root.allocator.free(sorted_probs);

        var cutoff: f32 = 0;
        std.sort.sort(f32, sorted_probs, {}, std.sort.desc(f32));

        for (sorted_probs) |prob| {
            cutoff += prob;

            if (cutoff > top_p) {
                for (probs) |*p| {
                    if (p.* < cutoff) p.* = 0;
                }

                break;
            }
        }
    }

    if (temperature != 1.0) {
        for (probs[0..]) |*p| {
            p.* = std.math.pow(f32, p.*, 1 / temperature);
        }
    }

    return @truncate(u32, random.weightedIndex(f32, probs));
}

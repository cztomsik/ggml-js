const std = @import("std");
const root = @import("root");
const ggml = @import("ggml.zig");

var prng = std.rand.DefaultPrng.init(123);
var random = prng.random();

pub fn sample_top_k_top_p(probs_tensor: *ggml.ggml_tensor, top_k: u32, top_p: f32, temperature: f32) !u32 {
    var ptr = @ptrCast([*]f32, @alignCast(@alignOf(f32), probs_tensor.data));
    var probs = ptr[0..@intCast(usize, probs_tensor.ne[0])];

    if (temperature != 1.0) {
        for (probs) |*p| {
            p.* /= temperature;
        }
    }

    var sorted_probs = try root.allocator.dupe(f32, probs);
    defer root.allocator.free(sorted_probs);

    std.sort.sort(f32, sorted_probs, {}, std.sort.desc(f32));
    var cutoff: f32 = 0;
    var cumsum: f32 = 0;

    // find cutoff value (either top_k or top_p)
    for (0..top_k, sorted_probs) |_, p| {
        cutoff = p;
        cumsum += p;

        if (cumsum >= top_p) break;
    }

    // erase everything below
    for (probs) |*p| {
        if (p.* < cutoff) p.* = 0;
        p.* /= cumsum;
    }

    return @truncate(u32, random.weightedIndex(f32, probs));
}

const std = @import("std");
const root = @import("root");
const ggml = @import("ggml.zig");

var prng: std.rand.DefaultPrng = undefined;
var random = std.crypto.random;

pub fn sample_seed(seed: u64) void {
    prng = std.rand.DefaultPrng.init(seed);
    random = prng.random();
}

pub fn sample_top_k_top_p(probs_tensor: *ggml.ggml_tensor, top_k: u32, top_p: f32, temperature: f32) !u32 {
    var ptr = ggml.ggml_get_data_f32(probs_tensor);
    var probs = ptr[0..@intCast(usize, probs_tensor.ne[0])];

    if (temperature != 1.0) {
        for (probs) |*p| {
            p.* = std.math.pow(f32, p.*, 1 / temperature);
        }
    }

    var sorted_probs = try root.allocator.dupe(f32, probs);
    defer root.allocator.free(sorted_probs);
    std.sort.sort(f32, sorted_probs, {}, std.sort.desc(f32));

    // find cutoff value (either top_k or top_p)
    var cumsum: f32 = 0;
    for (sorted_probs, 1..) |p, i| {
        cumsum += p;

        if (i == top_k or cumsum >= top_p) {
            break cut_off(probs, p);
        }
    }

    return @truncate(u32, random.weightedIndex(f32, probs));
}

fn cut_off(probs: []f32, cutoff: f32) void {
    for (probs) |*p| {
        if (p.* < cutoff) p.* = 0;
    }
}

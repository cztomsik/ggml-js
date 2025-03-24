const std = @import("std");
const root = @import("root");
const ggml = @import("ggml.zig");

const Candidate = struct { id: u32, prob: f32, score: f32 = undefined };

var prng: std.Random.DefaultPrng = undefined;
var random = std.crypto.random;

pub fn sample_seed(seed: u64) void {
    prng = std.Random.DefaultPrng.init(seed);
    random = prng.random();
}

pub fn sample_typical(tensor: *ggml.ggml_tensor, temperature: f32, tau: f32) !u32 {
    var list = try prepare_candidates(tensor);
    defer list.deinit();

    // compute entropy
    var entropy: f32 = 0;
    for (list.items) |it| entropy += -it.prob * @log(it.prob);

    // compute shifted_scores & sort
    for (list.items) |*it| it.score = @abs(-@log(it.prob) - entropy);
    sort_by(list.items, .score, std.sort.desc(f32));

    if (find_cutoff(list.items, tau)) |i| {
        list.shrinkRetainingCapacity(i + 1);
    }

    if (temperature != 1.0) {
        for (list.items) |*it| {
            it.prob = std.math.pow(f32, it.prob, 1 / temperature);
        }
    }

    // restore original order
    sort_by(list.items, .id, std.sort.asc(u32));

    return pick(list.items).id;
}

pub fn sample_top_k_top_p(tensor: *ggml.ggml_tensor, top_k: u32, top_p: f32, temperature: f32) !u32 {
    var list = try prepare_candidates(tensor);
    defer list.deinit();

    sort_by(list.items, .prob, std.sort.desc(f32));

    if (top_k < list.items.len) {
        list.shrinkRetainingCapacity(top_k);
    }

    if (top_p < 1) {
        if (find_cutoff(list.items, top_p)) |i| {
            list.shrinkRetainingCapacity(i + 1);
        }
    }

    if (temperature != 1.0) {
        for (list.items) |*it| {
            it.prob = std.math.pow(f32, it.prob, 1 / temperature);
        }
    }

    // restore original order
    sort_by(list.items, .id, std.sort.asc(u32));

    return pick(list.items).id;
}

fn prepare_candidates(tensor: *ggml.ggml_tensor) !std.ArrayList(Candidate) {
    var ptr = ggml.ggml_get_data_f32(tensor);
    const probs = ptr[0..@intCast(tensor.ne[0])];
    const items = try root.allocator.alloc(Candidate, probs.len);

    if (probs.len == 0) {
        return error.InvalidInput;
    }

    for (items, 0..) |*it, i| {
        it.* = .{
            .id = @intCast(i),
            .prob = probs[i],
        };
    }

    return std.ArrayList(Candidate).fromOwnedSlice(root.allocator, items);
}

fn sort_by(items: []Candidate, comptime field: std.meta.FieldEnum(Candidate), comptime comparator: anytype) void {
    const Sort = struct {
        pub fn compare(context: void, a: Candidate, b: Candidate) bool {
            return comparator(
                context,
                @field(a, @tagName(field)),
                @field(b, @tagName(field)),
            );
        }
    };
    std.mem.sort(Candidate, items, {}, Sort.compare);
}

fn pick(items: []const Candidate) Candidate {
    var sum: f32 = 0;
    for (items) |it| sum += it.prob;

    const i = find_cutoff(
        items,
        @min(random.float(f32) * sum, sum - std.math.floatEps(f32)),
    ).?;

    return items[i];
}

fn find_cutoff(items: []const Candidate, point: f32) ?usize {
    var sum: f32 = 0;

    for (items, 0..) |it, i| {
        sum += it.prob;
        if (sum > point) return i;
    }

    return null;
}

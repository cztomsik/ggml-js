const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addSharedLibrary(.{
        .name = "ggml-js",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // weak-linkage
    lib.linker_allow_shlib_undefined = true;

    // ggml
    lib.addIncludePath("deps/ggml/include/ggml");
    lib.addCSourceFile("deps/ggml/src/ggml.c", &.{ "-std=c11", "-pthread" });

    const napigen = b.createModule(.{ .source_file = .{ .path = "deps/napigen/napigen.zig" } });
    lib.addModule("napigen", napigen);

    // build .dylib & copy as .node
    b.installArtifact(lib);
    const copy_node_step = b.addInstallLibFile(lib.getOutputSource(), "ggml.node");
    b.getInstallStep().dependOn(&copy_node_step.step);
}

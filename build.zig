const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addSharedLibrary(.{
        .name = "ggml_js",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // libc
    lib.linkLibC();

    // Use weak-linkage
    lib.linker_allow_shlib_undefined = true;

    // ggml
    lib.addIncludePath(b.path("deps/ggml/include/ggml"));
    lib.addIncludePath(b.path("deps/ggml/src"));
    lib.addCSourceFile(.{ .file = b.path("deps/ggml/src/ggml.c"), .flags = &.{ "-std=c11", "-pthread" } });

    // Use Metal on macOS
    if (target.result.os.tag == .macos) {
        // lib.defineCMacroRaw("GGML_USE_METAL");
        // lib.defineCMacroRaw("GGML_METAL_NDEBUG");
        lib.addCSourceFiles(.{ .files = &.{"deps/ggml/src/ggml-metal.m"}, .flags = &.{"-std=c11"} });
        lib.linkFramework("Foundation");
        lib.linkFramework("Metal");
        lib.linkFramework("MetalKit");
        lib.linkFramework("MetalPerformanceShaders");

        // copy the *.metal file so that it can be loaded at runtime
        const copy_metal_step = b.addInstallLibFile(b.path("deps/ggml/src/ggml-metal.metal"), "ggml-metal.metal");
        b.getInstallStep().dependOn(&copy_metal_step.step);
    }

    const napigen = b.dependency("napigen", .{});
    lib.root_module.addImport("napigen", napigen.module("napigen"));

    // build .dylib & copy as .node
    b.installArtifact(lib);
    const copy_node_step = b.addInstallLibFile(lib.getEmittedBin(), try std.fmt.allocPrint(
        b.allocator,
        "ggml.{s}.node",
        .{@tagName(target.result.os.tag)},
    ));
    b.getInstallStep().dependOn(&copy_node_step.step);
}

const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addSharedLibrary(.{
        .name = "ggml-js",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // libc
    lib.linkLibC();

    // weak-linkage
    lib.linker_allow_shlib_undefined = true;

    // ggml
    lib.addIncludePath(.{ .path = "deps/ggml/include/ggml" });
    lib.addIncludePath(.{ .path = "deps/ggml/src" });
    lib.addCSourceFile(.{ .file = .{ .path = "deps/ggml/src/ggml.c" }, .flags = &.{ "-std=c11", "-pthread" } });

    // Use Metal on macOS
    if (target.getOsTag() == .macos) {
        lib.defineCMacroRaw("GGML_USE_METAL");
        lib.defineCMacroRaw("GGML_METAL_NDEBUG");
        lib.addCSourceFiles(&.{"deps/ggml/src/ggml-metal.m"}, &.{"-std=c11"});
        lib.linkFramework("Foundation");
        lib.linkFramework("Metal");
        lib.linkFramework("MetalKit");
        lib.linkFramework("MetalPerformanceShaders");

        // copy the *.metal file so that it can be loaded at runtime
        const copy_metal_step = b.addInstallLibFile(.{ .path = "deps/ggml/src/ggml-metal.metal" }, "ggml-metal.metal");
        b.getInstallStep().dependOn(&copy_metal_step.step);
    }

    const napigen = b.createModule(.{ .source_file = .{ .path = "deps/napigen/napigen.zig" } });
    lib.addModule("napigen", napigen);

    // build .dylib & copy as .node
    b.installArtifact(lib);
    const copy_node_step = b.addInstallLibFile(lib.getOutputSource(), try std.fmt.allocPrint(
        b.allocator,
        "ggml.{s}.node",
        .{@tagName(target.getOsTag())},
    ));
    b.getInstallStep().dependOn(&copy_node_step.step);
}

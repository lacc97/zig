const builtin = @import("builtin");

comptime {
    if (builtin.cpu.arch.isX86())
        _ = @import("cpu_model/x86.zig");
}

const builtin = @import("builtin");

const common = @import("../common.zig");

comptime {
    @export(cpu, .{ .name = "__cpu_model", .linkage = common.linkage, .visibility = common.visibility });
    @export(init, .{ .name = "__cpu_indicator_init", .linkage = common.linkage, .visibility = common.visibility });
}

// Based on LLVM's compiler-rt implementation.

const ProcessorVendor = enum(u32) {
    unknown = 0,
    intel,
    amd,
    other,
};
const ProcessorType = enum(u32) {
    unknown = 0,
    intel_bonnell,
    intel_core2,
    intel_corei7,
    amdfam10h,
    amdfam15h,
    intel_silvermont,
    intel_knl,
    amd_btver1,
    amd_btver2,
    amdfam17h,
    intel_knm,
    intel_goldmont,
    intel_goldmont_plus,
    intel_tremont,
    amdfam19h,
    zhaoxin_fam7h,
    intel_sierraforest,
    intel_grandridge,
    intel_clearwaterforest,
};
const ProcessorSubtype = enum(u32) {
    unknown = 0,
    intel_corei7_nehalem,
    intel_corei7_westmere,
    intel_corei7_sandybridge,
    amdfam10h_barcelona,
    amdfam10h_shanghai,
    amdfam10h_istanbul,
    amdfam15h_bdver1,
    amdfam15h_bdver2,
    amdfam15h_bdver3,
    amdfam15h_bdver4,
    amdfam17h_znver1,
    intel_corei7_ivybridge,
    intel_corei7_haswell,
    intel_corei7_broadwell,
    intel_corei7_skylake,
    intel_corei7_skylake_avx512,
    intel_corei7_cannonlake,
    intel_corei7_icelake_client,
    intel_corei7_icelake_server,
    amdfam17h_znver2,
    intel_corei7_cascadelake,
    intel_corei7_tigerlake,
    intel_corei7_cooperlake,
    intel_corei7_sapphirerapids,
    intel_corei7_alderlake,
    amdfam19h_znver3,
    intel_corei7_rocketlake,
    zhaoxin_fam7h_lujiazui,
    amdfam19h_znver4,
    intel_corei7_graniterapids,
    intel_corei7_graniterapids_d,
    intel_corei7_arrowlake,
    intel_corei7_arrowlake_s,
    intel_corei7_pantherlake,
};

const ProcessorFeature = enum(u8) {
    cmov = 0,
    mmx,
    popcnt,
    sse,
    sse2,
    sse3,
    ssse3,
    sse4_1,
    sse4_2,
    avx,
    avx2,
    sse4_a,
    fma4,
    xop,
    fma,
    avx512f,
    bmi,
    bmi2,
    aes,
    pclmul,
    avx512vl,
    avx512bw,
    avx512dq,
    avx512cd,
    avx512er,
    avx512pf,
    avx512vbmi,
    avx512ifma,
    avx5124vnniw,
    avx5124fmaps,
    avx512vpopcntdq,
    avx512vbmi2,
    gfni,
    vpclmulqdq,
    avx512vnni,
    avx512bitalg,
    avx512bf16,
    avx512vp2intersect,

    cmpxchg16b = 46,
    f16c = 49,
    lahf_lm = 54,
    lm,
    wp,
    lzcnt,
    movbe,

    x86_64_baseline = 95,
    x86_64_v2,
    x86_64_v3,
    x86_64_v4,
};

const ProcessorModel = extern struct {
    vendor: ProcessorVendor = .unknown,
    type: ProcessorType = .unknown,
    subtype: ProcessorSubtype = .unknown,
    features: [1]u32 = .{0},
};

var cpu: ProcessorModel = .{};
var cpu_extra_features: [3]u32 = [_]u32{0} ** 3;

fn init() callconv(.C) c_int {
    if (cpu.vendor != .unknown) return 0;

    const vendor, const max_leaf = blk: {
        const leaf = cpuid(0, 0);

        const vendor = leaf.ebx;
        const max_leaf = leaf.eax;

        break :blk .{ vendor, max_leaf };
    };
    const family, const model, const features = blk: {
        const leaf = cpuid(1, 0);

        var family: u32 = (leaf.eax >> 8) & 0xf;
        var model: u32 = (leaf.eax >> 4) & 0xf;
        if (family == 0x6 or family == 0xf) {
            if (family == 0xf) family += (leaf.eax >> 20) & 0xff;
            model += ((leaf.eax >> 16) & 0xf) << 4;
        }

        break :blk .{ family, model, getAvailableFeatures(max_leaf, leaf.ecx, leaf.edx) };
    };

    cpu.features[0] = features.data[0];
    cpu_extra_features[0] = features.data[1];
    cpu_extra_features[1] = features.data[2];
    cpu_extra_features[2] = features.data[3];

    cpu.vendor = switch (vendor) {
        0x756e6547 => blk: {
            break :blk .intel;
        },
        0x68747541 => blk: {
            cpu.type, cpu.subtype = getAmdTypeAndSubtype(family, model, &features);
            break :blk .amd;
        },
        else => .other,
    };

    return 0;
}

fn getAmdTypeAndSubtype(
    family: u32,
    model: u32,
    features: *const Features,
) struct { ProcessorType, ProcessorSubtype } {
    _ = features;

    var t: ProcessorType = .unknown;
    var s: ProcessorSubtype = .unknown;

    switch (family) {
        16 => {
            t = .amdfam10h;
            switch (model) {
                2 => s = .amdfam10h_barcelona,
                4 => s = .amdfam10h_shanghai,
                8 => s = .amdfam10h_istanbul,
                else => {},
            }
        },
        20 => t = .amd_btver1,
        21 => {
            t = .amdfam15h;
            switch (model) {
                0x60...0x7f => s = .amdfam15h_bdver4,
                0x30...0x3f => s = .amdfam15h_bdver3,
                0x10...0x1f => s = .amdfam15h_bdver2,
                0x00...0x0f => s = if (model == 0x02) .amdfam15h_bdver2 else .amdfam15h_bdver1,
                else => {},
            }
        },
        22 => t = .amd_btver2,
        23 => {
            t = .amdfam17h;
            switch (model) {
                0x30...0x3f, 0x71 => s = .amdfam17h_znver2,
                0x00...0x0f => s = .amdfam17h_znver1,
                else => {},
            }
        },
        25 => {
            t = .amdfam19h;
            switch (model) {
                0x00...0x0f, 0x20...0x5f => s = .amdfam19h_znver3,
                0x10...0x1f, 0x60...0x74, 0x78...0x7b, 0xa0...0xaf => s = .amdfam19h_znver4,
                else => {},
            }
        },
        else => {},
    }

    return .{ t, s };
}

const Features = struct {
    data: [4]u32 = [_]u32{0} ** 4,

    comptime {
        inline for (@typeInfo(ProcessorFeature).Enum.fields) |f| {
            if (f.value >= @bitSizeOf(Features))
                @compileError(@import("std").fmt.comptimePrint("ProcessorFeature.{s} ({}) bitindex too large", .{ f.name, f.value }));
        }
    }

    inline fn has(self: Features, f: ProcessorFeature) bool {
        const f_value = @intFromEnum(f);
        const bit = self.data[f_value / 32] & (1 << @intCast(f_value % 32));
        return bit != 0;
    }
    inline fn set(self: *Features, f: ProcessorFeature) void {
        const f_value = @intFromEnum(f);
        self.data[f_value / 32] |= (1 << @intCast(f_value % 32));
    }
};

fn getAvailableFeatures(max_leaf: u32, ecx: u32, edx: u32) Features {
    const bit = struct {
        inline fn bit(value: u32, bit_index: u5) bool {
            return (value & (1 << bit_index)) != 0;
        }
    }.bit;
    const set = struct {
        inline fn set(features: *Features, f: ProcessorFeature, enabled: bool) void {
            if (enabled) features.set(f);
        }
    }.set;

    var features: Features = .{};

    var leaf: CpuidLeaf = .{ .eax = 0, .ebx = 0, .ecx = ecx, .edx = edx };

    set(&features, .cmov, bit(leaf.edx, 15));
    set(&features, .mmx, bit(leaf.edx, 23));
    set(&features, .sse, bit(leaf.edx, 25));
    set(&features, .sse2, bit(leaf.edx, 26));

    set(&features, .sse3, bit(leaf.ecx, 0));
    set(&features, .pclmul, bit(leaf.ecx, 1));
    set(&features, .ssse3, bit(leaf.ecx, 9));
    set(&features, .fma, bit(leaf.ecx, 12));
    set(&features, .cmpxchg16b, bit(leaf.ecx, 13));
    set(&features, .sse4_1, bit(leaf.ecx, 19));
    set(&features, .sse4_2, bit(leaf.ecx, 20));
    set(&features, .movbe, bit(leaf.ecx, 22));
    set(&features, .popcnt, bit(leaf.ecx, 23));
    set(&features, .aes, bit(leaf.ecx, 25));
    set(&features, .f16c, bit(leaf.ecx, 29));

    const has_avx = blk: {
        const avx_bits = (1 << 27) | (1 << 28);
        if ((leaf.ecx & avx_bits) != avx_bits) break :blk false;
        break :blk ((getXCR0() & 0x6) == 0x6);
    };

    const has_avx512_save = builtin.os.tag.isDarwin() or (has_avx and ((leaf.eax & 0xe0) == 0xe0));

    const has_leaf7 = max_leaf >= 7;
    if (has_leaf7) {
        leaf = cpuid(7, 0);

        set(&features, .bmi, bit(leaf.ebx, 3));
        set(&features, .avx2, bit(leaf.ebx, 5) and has_avx);
        set(&features, .bmi2, bit(leaf.ebx, 8));
        set(&features, .avx512f, bit(leaf.ebx, 16) and has_avx512_save);
        set(&features, .avx512dq, bit(leaf.ebx, 17) and has_avx512_save);
        set(&features, .avx512ifma, bit(leaf.ebx, 21) and has_avx512_save);
        set(&features, .avx512pf, bit(leaf.ebx, 26) and has_avx512_save);
        set(&features, .avx512er, bit(leaf.ebx, 27) and has_avx512_save);
        set(&features, .avx512cd, bit(leaf.ebx, 28) and has_avx512_save);
        set(&features, .avx512bw, bit(leaf.ebx, 30) and has_avx512_save);
        set(&features, .avx512vl, bit(leaf.ebx, 31) and has_avx512_save);

        set(&features, .avx512vbmi, bit(leaf.ecx, 1) and has_avx512_save);
        set(&features, .avx512vbmi2, bit(leaf.ecx, 6) and has_avx512_save);
        set(&features, .gfni, bit(leaf.ecx, 8));
        set(&features, .vpclmulqdq, bit(leaf.ecx, 10) and has_avx);
        set(&features, .avx512vnni, bit(leaf.ecx, 11) and has_avx512_save);
        set(&features, .avx512bitalg, bit(leaf.ecx, 12) and has_avx512_save);
        set(&features, .avx512vpopcntdq, bit(leaf.ecx, 14) and has_avx512_save);

        set(&features, .avx5124vnniw, bit(leaf.edx, 2) and has_avx512_save);
        set(&features, .avx5124fmaps, bit(leaf.edx, 3) and has_avx512_save);
        set(&features, .avx512vp2intersect, bit(leaf.edx, 8) and has_avx512_save);

        const has_leaf7_subleaf1 = leaf.eax >= 1;
        if (has_leaf7_subleaf1) {
            leaf = cpuid(7, 1);

            set(&features, .avx512bf16, bit(leaf.eax, 5) and has_avx512_save);
        }
    }

    const max_ext_level = cpuid(0x80000000, 0).eax;
    const has_ext_leaf1 = max_ext_level >= 0x80000001;
    if (has_ext_leaf1) {
        leaf = cpuid(0x80000001, 0);

        set(&features, .lahf_lm, bit(leaf.ecx, 0));
        set(&features, .lzcnt, bit(leaf.ecx, 5));
        set(&features, .sse4_a, bit(leaf.ecx, 6));
        set(&features, .xop, bit(leaf.ecx, 11));
        set(&features, .fma4, bit(leaf.ecx, 16));

        set(&features, .lm, bit(leaf.edx, 29));
    }

    if (features.has(.lm) and features.has(.sse2)) {
        features.set(.x86_64_baseline);

        if (features.has(.cmpxchg16b) and features.has(.popcnt) and
            features.has(.lahf_lm) and features.has(.sse4_2))
        {
            features.set(.x86_64_v2);

            if (features.has(.avx2) and features.has(.bmi) and
                features.has(.bmi2) and features.has(.f16c) and
                features.has(.fma) and features.has(.lzcnt) and
                features.has(.movbe))
            {
                features.set(.x86_64_v3);

                if (features.has(.avx512bw) and features.has(.avx512cd) and
                    features.has(.avx512dq) and features.has(.avx512vl))
                {
                    features.set(.x86_64_v4);
                }
            }
        }
    }

    return features;
}

const CpuidLeaf = packed struct {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
};

inline fn cpuid(leaf_id: u32, subid: u32) CpuidLeaf {
    // valid for both x86 and x86_64
    var eax: u32 = undefined;
    var ebx: u32 = undefined;
    var ecx: u32 = undefined;
    var edx: u32 = undefined;

    asm volatile ("cpuid"
        : [_] "={eax}" (eax),
          [_] "={ebx}" (ebx),
          [_] "={ecx}" (ecx),
          [_] "={edx}" (edx),
        : [_] "{eax}" (leaf_id),
          [_] "{ecx}" (subid),
    );

    return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
}

inline fn getXCR0() u32 {
    return asm volatile (
        \\ xor %%ecx, %%ecx
        \\ xgetbv
        : [_] "={eax}" (-> u32),
        :
        : "edx", "ecx"
    );
}

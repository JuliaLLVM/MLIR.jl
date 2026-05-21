# build a local version of mlir-jl-tblgen
using Pkg, Scratch, Preferences, CMake_jll, ArgParse

if haskey(ENV, "GITHUB_ACTIONS")
    println(
        "::warning ::Using a locally-built mlir-jl-tblgen; A bump of mlir_jl_tblgen_jll will be required before releasing MLIR.jl.",
    )
end

MLIR = Base.UUID("bfde9dd4-8f40-4a1e-be09-1475335e1c92")

s = ArgParseSettings()
#! format: off
@add_arg_table s begin
    "--build-dir"
        help = "Build directory."
        arg_type = String
        default = mktempdir()
    "--install-dir"
        help = "Scratch directory for installation."
        arg_type = String
        default = get_scratch!(MLIR, "build")
    "--llvm-version"
        help = "Target LLVM/MLIR version."
        arg_type = String
        default = "$(Base.libllvm_version.major).$(Base.libllvm_version.minor)"
    "--llvm-assertions"
        help = "Build LLVM/MLIR with assertions enabled."
        arg_type = Bool
        default = try
            cglobal((:_ZN4llvm24DisableABIBreakingChecksE, Base.libllvm_path()), Cvoid)
            false
        catch
            true
        end
    "--llvm-dir"
        help = "Path to LLVM installation"
        arg_type = String
    "--debug"
        help = "Build with debug symbols."
        action = :store_true
    "--disable-cleanup"
        help = "Don't delete install directory after building."
        action = :store_true
end
#! format: on

parsed_args = parse_args(ARGS, s)

println("Parsed args:")
for (k, v) in parsed_args
    println("  $k => $v")
end
println()

source_dir = joinpath(@__DIR__, "tblgen")
llvm_version = VersionNumber(parsed_args["llvm-version"])
llvm_assertions = parsed_args["llvm-assertions"]
debug = parsed_args["debug"]

install_dir = parsed_args["install-dir"]
if isdir(install_dir) && !parsed_args["disable-cleanup"]
    println("Removing existing installation at $install_dir")
    rm(install_dir; recursive=true)
end

build_dir = parsed_args["build-dir"]

# download LLVM and MLIR artifacts if required
llvm_dir = if !isnothing(parsed_args["llvm-dir"])
    parsed_args["llvm-dir"]
else
    Pkg.activate(; temp=true)

    LLVM = if llvm_assertions
        Pkg.add(; name="LLVM_full_assert_jll", version=llvm_version)
        Base.require(Core.Module(:LLVM_full_assert_jll), :LLVM_full_assert_jll)
    else
        Pkg.add(; name="LLVM_full_jll", version=llvm_version)
        Base.require(Core.Module(:LLVM_full_jll), :LLVM_full_jll)
    end
    LLVM.artifact_dir
end

LLVM_DIR = joinpath(llvm_dir, "lib", "cmake", "llvm")
MLIR_DIR = joinpath(llvm_dir, "lib", "cmake", "mlir")

# build and install
@info "Building" source_dir install_dir build_dir LLVM_DIR MLIR_DIR
cmake() do cmake_path
    build_type = debug ? "Debug" : "Release"
    config_opts = `-DLLVM_ROOT=$(LLVM_DIR) -DMLIR_ROOT=$(MLIR_DIR) -DCMAKE_INSTALL_PREFIX=$(install_dir)`
    if Sys.iswindows()
        # prevent picking up MSVC
        config_opts = `$config_opts -G "MSYS Makefiles"`
    end
    run(`$cmake_path $config_opts -B$(build_dir) -S$(source_dir)`)
    run(`$cmake_path --build $(build_dir) --target install`)
end

bin_path = joinpath(install_dir, "bin", "mlir-jl-tblgen")
isfile(bin_path) || error("Could not find executable $bin_path in build directory")

# tell LLVM.jl to load our executable instead of the default artifact one
set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "mlir_jl_tblgen_jll",
    "mlir_jl_tblgen" => bin_path;
    force=true,
)

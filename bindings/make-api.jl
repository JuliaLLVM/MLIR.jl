using Pkg
import BinaryBuilderBase:
    PkgSpec, Prefix, temp_prefix, setup_dependencies, cleanup_dependencies, destdir
using Clang.Generators
using Clang: Clang, CLLinkageSpec, children

# Add support for extern "C" blocks - Clang.jl doesn't handle these by default
# We need to recurse into the children of the linkage spec to process the declarations inside
function Generators.collect_top_level_nodes!(
    nodes::Vector{ExprNode}, cursor::CLLinkageSpec, options
)
    for child in children(cursor)
        Generators.collect_top_level_nodes!(nodes, child, options)
    end
    return nodes
end

julia_llvm = [
    (v"1.9", v"14.0.6+4"),
    (v"1.10", v"15.0.7+9"),
    (v"1.11", v"16.0.6+5"),
    (v"1.12", v"17.0.6+5"),
    (v"1.12", v"18.1.7+4"),
    (v"1.12", v"19.1.7+1"),
    (v"1.12", v"20.1.8+0"),
    (v"1.12", v"21.1.8+0"),
]
options = load_options(joinpath(@__DIR__, "wrap.toml"))

@add_def off_t
@add_def MlirTypesCallback

for (julia_version, llvm_version) in julia_llvm
    println("Generating... julia version: $julia_version, llvm version: $llvm_version")

    temp_prefix() do prefix
        platform = Pkg.BinaryPlatforms.HostPlatform()
        platform["llvm_version"] = string(llvm_version.major)
        platform["julia_version"] = string(julia_version)

        # Note: 1.10
        dependencies = PkgSpec[
            PkgSpec(; name="LLVM_full_jll", version=llvm_version),
            PkgSpec(; name="mlir_jl_tblgen_jll"),
        ]

        artifact_paths = setup_dependencies(prefix, dependencies, platform; verbose=true)

        mlir_jl_tblgen = joinpath(destdir(prefix, platform), "bin", "mlir-jl-tblgen")
        include_dir = joinpath(destdir(prefix, platform), "include")

        # generate MLIR API bindings
        mkpath(joinpath(@__DIR__, "..", "src", "API", string(llvm_version.major)))

        let options = deepcopy(options)
            output_file_path = joinpath(
                @__DIR__,
                "..",
                "src",
                "API",
                string(llvm_version.major),
                options["general"]["output_file_path"],
            )
            isdir(dirname(output_file_path)) || mkpath(dirname(output_file_path))
            options["general"]["output_file_path"] = output_file_path

            libmlir_header_dir = joinpath(include_dir, "mlir-c")
            args = Generators.get_default_args(get_triple(); is_cxx=true)
            push!(args, "-I$include_dir")
            push!(args, "-xc++")

            headers = detect_headers(
                libmlir_header_dir, args, Dict(), endswith("Python/Interop.h")
            )
            ctx = create_context(headers, args, options)

            # build without printing so we can do custom rewriting
            build!(ctx, BUILDSTAGE_NO_PRINTING)
            
            # anonymous enums give some problems...
            # specifically when no explicit type is given (thus `int`) and unsigned literals are used
            # check https://github.com/JuliaInterop/Clang.jl/issues/572
            filter!(get_nodes(ctx.dag)) do node
                !(node isa ExprNode{Clang.Generators.EnumAnonymous, Clang.Generators.CLEnumDecl})
            end

            # print
            build!(ctx, BUILDSTAGE_PRINTING_ONLY)
        end
    end
end

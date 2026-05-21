using Pkg
import BinaryBuilderBase:
    PkgSpec, Prefix, temp_prefix, setup_dependencies, cleanup_dependencies, destdir
using ArgParse

s = ArgParseSettings()
#! format: off
@add_arg_table s begin
    "--julia-version"
        help = "Target Julia version."
        arg_type = String
        default = "1.10"
    "--llvm-version"
        help = "Target LLVM version."
        arg_type = String
    "--local"
        help = "Whether to build mlir-jl-tblgen locally for the target."
        action = :store_true
end
#! format: on

parsed_args = parse_args(ARGS, s)

julia_version = VersionNumber(parsed_args["julia-version"])

llvm_versions = if !isnothing(parsed_args["llvm-version"])
    [VersionNumber(parsed_args["llvm-version"])]
else
    [
        v"14.0.6+4",
        v"15.0.7+9",
        v"16.0.6+5",
        v"17.0.6+5",
        v"18.1.7+4",
        v"19.1.7+1",
        v"20.1.8+0",
        v"21.1.8+0",
    ]
end

build_local = parsed_args["local"]

# returns a mapping of dialect name to expected output file name and list of td files to generate from, for a given MLIR version
function mlir_dialects(version::VersionNumber)
    @assert v"14" <= version "Unsupported MLIR version: $version. Supported versions are 14-22."

    # mapping of dialects to output file name and td files, in alphabetical order
    dialects = Dict{String,Any}()

    dialects["acc"] = (; output_file="OpenACC.jl", td_files=["OpenACC/OpenACCOps.td"])
    dialects["affine"] = (; output_file="Affine.jl", td_files=["Affine/IR/AffineOps.td"])

    # dialect added in v15 and files moved to IR subfolder in v17
    if v"15" <= version < v"17"
        dialects["amdgpu"] = (; output_file="AMDGPU.jl", td_files=["AMDGPU/AMDGPU.td"])
    elseif v"17" <= version
        dialects["amdgpu"] = (; output_file="AMDGPU.jl", td_files=["AMDGPU/IR/AMDGPU.td"])
    end

    dialects["amx"] = (; output_file="AMX.jl", td_files=["AMX/AMX.td"])

    # folder renamed from 'Arithmetic' to 'Arith' in v16 (dialect name remains 'arith')
    if version < v"16"
        dialects["arith"] = (;
            output_file="Arith.jl", td_files=["Arithmetic/IR/ArithmeticOps.td"]
        )
    else
        dialects["arith"] = (; output_file="Arith.jl", td_files=["Arith/IR/ArithOps.td"])
    end

    dialects["arm_neon"] = (; output_file="ArmNeon.jl", td_files=["ArmNeon/ArmNeon.td"])

    # files moved to IR subfolder in v18
    if version < v"18"
        dialects["arm_sve"] = (; output_file="ArmSVE.jl", td_files=["ArmSVE/ArmSVE.td"])
    elseif v"18" <= version
        dialects["arm_sve"] = (; output_file="ArmSVE.jl", td_files=["ArmSVE/IR/ArmSVE.td"])
    end

    # dialect added in v17 and extended in v18
    if v"17" <= version < v"18"
        dialects["arm_sme"] = (; output_file="ArmSME.jl", td_files=["ArmSME/IR/ArmSME.td"])
    elseif v"18" <= version
        dialects["arm_sme"] = (;
            output_file="ArmSME.jl",
            td_files=["ArmSME/IR/ArmSMEOps.td", "ArmSME/IR/ArmSMEIntrinsicOps.td"],
        )
    end

    dialects["async"] = (; output_file="Async.jl", td_files=["Async/IR/AsyncOps.td"])
    dialects["bufferization"] = (;
        output_file="Bufferization.jl", td_files=["Bufferization/IR/BufferizationOps.td"]
    )
    dialects["builtin"] = (; output_file="Builtin.jl", td_files=["../IR/BuiltinOps.td"])
    dialects["complex"] = (;
        output_file="Complex.jl", td_files=["Complex/IR/ComplexOps.td"]
    )

    if v"15" <= version
        dialects["cf"] = (;
            output_file="ControlFlow.jl", td_files=["ControlFlow/IR/ControlFlowOps.td"]
        )
    end

    # dialects["dlti"] = (; output_file = "DLTI.jl", td_files = ["DLTI/DLTI.td"]) # TODO crashes
    dialects["emitc"] = (; output_file="EmitC.jl", td_files=["EmitC/IR/EmitC.td"])

    # dialect added in v15
    if v"15" <= version
        dialects["func"] = (; output_file="Func.jl", td_files=["Func/IR/FuncOps.td"])
    end

    # moved to IR subfolder in v15
    if version < v"15"
        dialects["gpu"] = (; output_file="GPU.jl", td_files=["GPU/GPUOps.td"])
    else
        dialects["gpu"] = (; output_file="GPU.jl", td_files=["GPU/IR/GPUOps.td"])
    end

    # dialect added in v17
    if v"16" <= version
        dialects["index"] = (; output_file="Index.jl", td_files=["Index/IR/IndexOps.td"])
    end

    # dialect added in v17
    if v"17" <= version
        dialects["irdl"] = (; output_file="IRDL.jl", td_files=["IRDL/IR/IRDLOps.td"])
    end

    dialects["linalg"] = (;
        output_file="Linalg.jl",
        td_files=["Linalg/IR/LinalgOps.td", "Linalg/IR/LinalgStructuredOps.td"],
    )

    dialects["llvm"] = (;
        output_file="LLVMIR.jl",
        td_files=["LLVMIR/LLVMOps.td", "LLVMIR/NVVMOps.td", "LLVMIR/ROCDLOps.td"],
    )
    if v"16" <= version
        push!(dialects["llvm"].td_files, "LLVMIR/LLVMIntrinsicOps.td")
    end
    if v"19" <= version
        push!(dialects["llvm"].td_files, "LLVMIR/VCIXOps.td")
    end
    if v"21" <= version
        push!(dialects["llvm"].td_files, "LLVMIR/XeVMOps.td")
    end

    dialects["math"] = (; output_file="Math.jl", td_files=["Math/IR/MathOps.td"])
    dialects["memref"] = (; output_file="MemRef.jl", td_files=["MemRef/IR/MemRefOps.td"])

    # dialect added in v18, renamed to `shard` in v22
    if v"18" <= version < v"22"
        dialects["mesh"] = (; output_file="Mesh.jl", td_files=["Mesh/IR/MeshOps.td"])
    end

    # dialect added in v15
    if v"15" <= version
        dialects["ml_program"] = (;
            output_file="MLProgram.jl", td_files=["MLProgram/IR/MLProgramOps.td"]
        )
    end

    # dialect added in v19
    if v"19" <= version
        dialects["mpi"] = (; output_file="MPI.jl", td_files=["MPI/IR/MPIOps.td"])
    end

    # dialect added in v15
    # if v"15" <= version
    #     dialects["nvgpu"] = (; output_file="NVGPU.jl", td_files=["NVGPU/IR/NVGPU.td"])
    # end

    dialects["omp"] = (; output_file="OpenMP.jl", td_files=["OpenMP/OpenMPOps.td"])
    dialects["pdl_interp"] = (;
        output_file="PDLInterp.jl", td_files=["PDLInterp/IR/PDLInterpOps.td"]
    )
    dialects["pdl"] = (; output_file="PDL.jl", td_files=["PDL/IR/PDLOps.td"])

    # dialect added in v19 and removed in v20
    if v"19" <= version < v"20"
        dialects["polynomial"] = (;
            output_file="Polynomial.jl", td_files=["Polynomial/IR/Polynomial.td"]
        )
    end

    # dialect added in v19, but ops added in v21
    if v"21" <= version
        dialects["ptr"] = (; output_file = "Ptr.jl", td_files = ["Ptr/IR/PtrOps.td"])
    end

    if version < v"20"
        dialects["quant"] = (; output_file="Quant.jl", td_files=["Quant/QuantOps.td"])
    else
        dialects["quant"] = (; output_file="Quant.jl", td_files=["Quant/IR/QuantOps.td"])
    end

    # moved to IR subfolder in v15
    if version < v"15"
        dialects["scf"] = (; output_file="SCF.jl", td_files=["SCF/SCFOps.td"])
    else
        dialects["scf"] = (; output_file="SCF.jl", td_files=["SCF/IR/SCFOps.td"])
    end

    dialects["shape"] = (; output_file="Shape.jl", td_files=["Shape/IR/ShapeOps.td"])

    # previously named "mesh" before v22
    if v"22" <= version
        dialects["shard"] = (; output_file="Shard.jl", td_files=["Shard/IR/ShardOps.td"])
    end

    # dialect added in v20
    if v"21" <= version
        dialects["smt"] = (; output_file="SMT.jl", td_files=["SMT/IR/SMT.td"])
    end

    dialects["sparse_tensor"] = (;
        output_file="SparseTensor.jl", td_files=["SparseTensor/IR/SparseTensorOps.td"]
    )

    # dialect name renamed to 'spirv' in v16
    if version < v"16"
        dialects["spv"] = (; output_file="SPIRV.jl", td_files=["SPIRV/IR/SPIRVOps.td"])
    else
        dialects["spirv"] = (; output_file="SPIRV.jl", td_files=["SPIRV/IR/SPIRVOps.td"])
    end

    # deleted in v15
    if version < v"15"
        dialects["std"] = (;
            output_file="StandardOps.jl", td_files=["StandardOps/IR/Ops.td"]
        )
    end

    dialects["tensor"] = (; output_file="Tensor.jl", td_files=["Tensor/IR/TensorOps.td"])
    dialects["tosa"] = (; output_file="Tosa.jl", td_files=["Tosa/IR/TosaOps.td"])

    # dialect added in v15 and files added on multiple versions
    if v"15" <= version < v"16"
        dialects["transform"] = (;
            output_file="Transform.jl",
            td_files=[
                "Bufferization/TransformOps/BufferizationTransformOps.td",
                "Linalg/TransformOps/LinalgTransformOps.td",
                "SCF/TransformOps/SCFTransformOps.td",
                "Transform/IR/TransformOps.td",
            ],
        )
    elseif v"16" <= version < v"17"
        dialects["transform"] = (;
            output_file="Transform.jl",
            td_files=[
                "Affine/TransformOps/AffineTransformOps.td",
                "Bufferization/TransformOps/BufferizationTransformOps.td",
                "GPU/TransformOps/GPUTransformOps.td",
                "Linalg/TransformOps/LinalgTransformOps.td",
                "MemRef/TransformOps/MemRefTransformOps.td",
                "SCF/TransformOps/SCFTransformOps.td",
                "Transform/IR/TransformOps.td",
                "Vector/TransformOps/VectorTransformOps.td",
            ],
        )
    elseif v"17" <= version < v"18"
        dialects["transform"] = (;
            output_file="Transform.jl",
            td_files=[
                "Affine/TransformOps/AffineTransformOps.td",
                "Bufferization/TransformOps/BufferizationTransformOps.td",
                "GPU/TransformOps/GPUTransformOps.td",
                "Linalg/TransformOps/LinalgMatchOps.td",
                "Linalg/TransformOps/LinalgTransformOps.td",
                "MemRef/TransformOps/MemRefTransformOps.td",
                "NVGPU/TransformOps/NVGPUTransformOps.td",
                "SCF/TransformOps/SCFTransformOps.td",
                "Tensor/TransformOps/TensorTransformOps.td",
                "Transform/IR/TransformOps.td",
                "Vector/TransformOps/VectorTransformOps.td",
            ],
        )
    elseif v"18" <= version
        dialects["transform"] = (;
            output_file="Transform.jl",
            td_files=[
                "Affine/TransformOps/AffineTransformOps.td",
                "Bufferization/TransformOps/BufferizationTransformOps.td",
                "Func/TransformOps/FuncTransformOps.td",
                "GPU/TransformOps/GPUTransformOps.td",
                "Linalg/TransformOps/LinalgMatchOps.td",
                "Linalg/TransformOps/LinalgTransformOps.td",
                "MemRef/TransformOps/MemRefTransformOps.td",
                "NVGPU/TransformOps/NVGPUTransformOps.td",
                "SCF/TransformOps/SCFTransformOps.td",
                "SparseTensor/TransformOps/SparseTensorTransformOps.td",
                "Tensor/TransformOps/TensorTransformOps.td",
                "Transform/IR/TransformOps.td",
                "Transform/DebugExtension/DebugExtensionOps.td",
                "Transform/LoopExtension/LoopExtensionOps.td",
                "Transform/PDLExtension/PDLExtensionOps.td",
                "Vector/TransformOps/VectorTransformOps.td",
            ],
        )
    end

    # dialect added in v17
    if v"17" <= version
        dialects["ub"] = (; output_file="UB.jl", td_files=["UB/IR/UBOps.td"])
    end

    dialects["vector"] = (; output_file="Vector.jl", td_files=["Vector/IR/VectorOps.td"])
    dialects["x86vector"] = (;
        output_file="X86Vector.jl", td_files=["X86Vector/X86Vector.td"]
    )

    # dialect added in v19
    if v"19" < version
        dialects["xegpu"] = (; output_file="XeGPU.jl", td_files=["XeGPU/IR/XeGPUOps.td"])
    end

    return dialects
end

for llvm_version in llvm_versions
    println("Generating... Julia $julia_version, LLVM $llvm_version")

    temp_prefix() do prefix
        platform = Pkg.BinaryPlatforms.HostPlatform()
        platform["llvm_version"] = string(llvm_version.major)
        platform["julia_version"] = string(julia_version)

        # Note: 1.10
        dependencies = PkgSpec[
            PkgSpec(; name="LLVM_full_jll", version=llvm_version),
            PkgSpec(; name="mlir_jl_tblgen_jll"),
        ]

        project_prefix = destdir(prefix, platform)
        artifact_paths = setup_dependencies(prefix, dependencies, platform; verbose=true)

        mlir_jl_tblgen = joinpath(project_prefix, "bin", "mlir-jl-tblgen")
        include_dir = joinpath(project_prefix, "include")

        if build_local
            run(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..", "deps")) $(joinpath(@__DIR__, "..", "deps", "build_local.jl")) --llvm-version=$(llvm_version) --install-dir=$project_prefix --disable-cleanup --debug`)
        end

        # generate MLIR dialect bindings
        mkpath(joinpath(@__DIR__, "..", "src", "Dialects", string(llvm_version.major)))

        for (dialect_name, (binding, tds)) in sort(mlir_dialects(llvm_version), by=first)
            tempfiles = map(tds) do td
                tempfile, _ = mktemp()
                tdpath = joinpath(include_dir, "mlir", "Dialect", td)
                flags = [
                    "--generator=emit-op-table-defs",
                    "--disable-module-wrap",
                    "--dialect-name=$(dialect_name)",
                    tdpath,
                    "-I",
                    dirname(tdpath),
                    "-I",
                    include_dir,
                    "-o",
                    tempfile,
                ]
                run(`$mlir_jl_tblgen $flags`)
                return tempfile
            end

            output = joinpath(
                @__DIR__, "..", "src", "Dialects", string(llvm_version.major), binding
            )
            open(output, "w") do io
                println(io, "module $dialect_name\n")
                for tempfile in tempfiles
                    write(io, read(tempfile, String))
                end
                println(io, "end # $dialect_name")
            end

            println("- Generated \"$binding\" from $(join(tds, ",", " and "))")
        end
    end
end

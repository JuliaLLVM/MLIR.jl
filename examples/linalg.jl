using MLIR.IR
using MLIR.API
using MLIR.Dialects: arith, linalg, func

n = 128
a = rand(Float64, n, n)
b = rand(Float64, n, n)

function linalg_matmul(a::IR.Value, b::IR.Value; result::Union{Nothing,IR.MLIRType}=nothing, location=IR.Location())
    IR.create_operation(
        "linalg.matmul", location;
        operands=Value[a, b],
        owned_regions=IR.Region[],
        successors=IR.Block[],
        results=isnothing(result) ? nothing : MLIRType[result],
        attributes=IR.NamedAttribute[],
    )
end

fptr = IR.context!(IR.Context()) do
    IR.enable_multithreading!(false)

    mod = IR.Module(Location())
    body = IR.get_body(mod)

    # Create a function    
    mattype = MLIRType(API.mlirRankedTensorTypeGet(2, [n, n], MLIRType(Float64), API.mlirAttributeGetNull()))

    block = IR.Block()
    a_ir = IR.push_argument!(block, mattype, IR.Location())
    b_ir = IR.push_argument!(block, mattype, IR.Location())
    op = linalg_matmul(a_ir, b_ir; result=mattype) # TODO refactor to `linalg.matmul` when bindings are regenerated
    push!(block, op)

    region = IR.Region()
    push!(region, block)

    ftype = MLIRType(API.mlirFunctionTypeGet(IR.context(), 2, [mattype, mattype], 1, [mattype]))
    f = func.func_(;
        sym_name=IR.Attribute("matmul_demo"),
        function_type=IR.Attribute(ftype),
        body=region,
    )
    push!(body, f)

    pm = IR.PassManager()
    opm = IR.OpPassManager(pm)

    IR.enable_ir_printing!(pm)
    IR.enable_verifier!(pm, true)

    API.mlirRegisterAllPasses()
    API.mlirRegisterAllLLVMTranslations(IR.context())
    IR.add_pipeline!(opm, "convert-func-to-llvm")

    IR.run!(pm, mod)

    # jit = if LLVM.version() >= v"16"
    #     API.mlirExecutionEngineCreate(mod, 0, 0, C_NULL, false)
    # else
    #     API.mlirExecutionEngineCreate(mod, 0, 0, C_NULL)
    # end
    # API.mlirExecutionEngineLookup(jit, "matmul_demo")
end

# @test ccall(fptr, Ptr{Float64}, (Ptr{Float64}, Ptr{Float64}), a, b) ≈ a * b

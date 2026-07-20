using MLIR.Dialects: arith, builtin
using MLIR.IR, LLVM

@testset "operation introspection" begin
    ctx = IR.Context()
    IR.activate(ctx)

    IR.get_or_load_dialect!(IR.DialectHandle(:linalg))
    if LLVM.version() >= v"15"
        IR.load_all_available_dialects()
    end
    op = arith.constant(; value=true, result=IR.Type(Bool))

    @test IR.name(op) == "arith.constant"
    @test IR.dialect(op) === :arith
    @test IR.isbool(IR.getattr(op, "value"))

    IR.deactivate(ctx)
    IR.dispose(ctx)
end

@testset "Module construction from operation" begin
    ctx = IR.Context()
    IR.activate(ctx)

    if LLVM.version() >= v"15"
        op = builtin.module_(; bodyRegion=IR.Region())
    else
        op = builtin.module_(; body=IR.Region())
    end
    mod = IR.Module(op)
    op = IR.Operation(mod)

    @test IR.name(op) == "builtin.module"

    # Only a `module` operation can be used to create a module.
    @test_throws UndefRefError IR.Module(
        arith.constant(; value=true, result=IR.Type(Bool))
    )

    IR.deactivate(ctx)
    IR.dispose(ctx)
end

@testset "Iterators" begin
    ctx = IR.Context()
    IR.activate(ctx)

    mod = if LLVM.version() >= v"15"
        IR.load_all_available_dialects()
        IR.get_or_load_dialect!(IR.DialectHandle(:func))
        parse(
            IR.Module,
            """
            module {
                func.func @f() {
                    return
                }
            }
            """,
        )
    else
        IR.get_or_load_dialect!(IR.DialectHandle(:std))
        parse(
            IR.Module,
            """
            module {
                func @f() {
                    std.return
                }
            }
            """,
        )
    end
    b = IR.body(mod)
    ops = collect(b)
    @test ops isa Vector{Operation}
    @test length(ops) == 1

    op = only(ops)
    regions = collect(op)
    @test regions isa Vector{Region}
    @test length(regions) == 1

    region = only(regions)
    blocks = collect(region)
    @test blocks isa Vector{Block}
    @test length(blocks) == 1

    IR.deactivate(ctx)
    IR.dispose(ctx)
end

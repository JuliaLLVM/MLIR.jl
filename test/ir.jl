using MLIR.Dialects: arith, builtin
using MLIR.IR, LLVM

@testset "operation introspection" begin
    IR.context!(IR.Context()) do
        IR.get_or_load_dialect!(IR.DialectHandle(:linalg))
        if LLVM.version() >= v"15"
            IR.load_all_available_dialects()
        end
        op = arith.constant(; value=true, result=IR.Type(Bool))

        @test IR.name(op) == "arith.constant"
        @test IR.dialect(op) === :arith
        @test IR.isbool(attr(op, "value"))
    end
end

@testset "Module construction from operation" begin
    IR.context!(IR.Context()) do
        if LLVM.version() >= v"15"
            op = builtin.module_(bodyRegion=IR.Region())
        else
            op = builtin.module_(body=IR.Region())
        end
        mod = IR.Module(op)
        op = IR.Operation(mod)

        @test IR.name(op) == "builtin.module"

        # Only a `module` operation can be used to create a module.
        @test_throws AssertionError IR.Module(arith.constant(; value=true, result=IR.Type(Bool)))
    end
end

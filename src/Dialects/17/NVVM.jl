module nvvm

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes

"""
`barrier0`

"""
function barrier0(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.barrier0", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`read_ptx_sreg_ntid_x`

"""
function read_ptx_sreg_ntid_x(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.read.ptx.sreg.ntid.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`read_ptx_sreg_ntid_y`

"""
function read_ptx_sreg_ntid_y(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.read.ptx.sreg.ntid.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`read_ptx_sreg_ntid_z`

"""
function read_ptx_sreg_ntid_z(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.read.ptx.sreg.ntid.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`read_ptx_sreg_ctaid_x`

"""
function read_ptx_sreg_ctaid_x(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.read.ptx.sreg.ctaid.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`read_ptx_sreg_ctaid_y`

"""
function read_ptx_sreg_ctaid_y(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.read.ptx.sreg.ctaid.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`read_ptx_sreg_ctaid_z`

"""
function read_ptx_sreg_ctaid_z(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.read.ptx.sreg.ctaid.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_bulk_tensor_shared_cluster_global`

"""
function cp_async_bulk_tensor_shared_cluster_global(dstMem::Value, tmaDescriptor::Value, mbar::Value, coordinates::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dstMem, tmaDescriptor, mbar, coordinates..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.cp.async.bulk.tensor.shared.cluster.global", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_commit_group`

"""
function cp_async_commit_group(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.cp.async.commit.group", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_shared_global`

"""
function cp_async_shared_global(dst::Value, src::Value, cpSize=nothing::Union{Nothing, Value}; size, modifier, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dst, src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("size", size), NamedAttribute("modifier", modifier), ]
    !isnothing(cpSize) && push!(operands, cpSize)
    
    IR.create_operation(
        "nvvm.cp.async.shared.global", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_wait_group`

"""
function cp_async_wait_group(; n, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("n", n), ]
    
    IR.create_operation(
        "nvvm.cp.async.wait.group", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`read_ptx_sreg_nctaid_x`

"""
function read_ptx_sreg_nctaid_x(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.read.ptx.sreg.nctaid.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`read_ptx_sreg_nctaid_y`

"""
function read_ptx_sreg_nctaid_y(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.read.ptx.sreg.nctaid.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`read_ptx_sreg_nctaid_z`

"""
function read_ptx_sreg_nctaid_z(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.read.ptx.sreg.nctaid.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`read_ptx_sreg_laneid`

"""
function read_ptx_sreg_laneid(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.read.ptx.sreg.laneid", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`ldmatrix`

"""
function ldmatrix(ptr::Value; res::IR.Type, num, layout, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("num", num), NamedAttribute("layout", layout), ]
    
    IR.create_operation(
        "nvvm.ldmatrix", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mbarrier_arrive_expect_tx`

"""
function mbarrier_arrive_expect_tx(addr::Value, txcount::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, txcount, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mbarrier.arrive.expect_tx", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mbarrier_arrive_expect_tx_shared`

"""
function mbarrier_arrive_expect_tx_shared(addr::Value, txcount::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, txcount, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mbarrier.arrive.expect_tx.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mbarrier_arrive_nocomplete`

"""
function mbarrier_arrive_nocomplete(addr::Value, count::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[addr, count, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mbarrier.arrive.nocomplete", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mbarrier_arrive_nocomplete_shared`

"""
function mbarrier_arrive_nocomplete_shared(addr::Value, count::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[addr, count, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mbarrier.arrive.nocomplete.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mbarrier_arrive`

"""
function mbarrier_arrive(addr::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mbarrier.arrive", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mbarrier_arrive_shared`

"""
function mbarrier_arrive_shared(addr::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mbarrier.arrive.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mbarrier_init`

"""
function mbarrier_init(addr::Value, count::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, count, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mbarrier.init", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mbarrier_init_shared`

"""
function mbarrier_init_shared(addr::Value, count::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, count, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mbarrier.init.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mbarrier_inval`

"""
function mbarrier_inval(addr::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mbarrier.inval", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mbarrier_inval_shared`

"""
function mbarrier_inval_shared(addr::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mbarrier.inval.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mbarrier_test_wait`

"""
function mbarrier_test_wait(addr::Value, state::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[addr, state, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mbarrier.test.wait", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mbarrier_test_wait_shared`

"""
function mbarrier_test_wait_shared(addr::Value, state::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[addr, state, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mbarrier.test.wait.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mbarrier_try_wait_parity`

"""
function mbarrier_try_wait_parity(addr::Value, phase::Value, ticks::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, phase, ticks, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mbarrier.try_wait.parity", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mbarrier_try_wait_parity_shared`

"""
function mbarrier_try_wait_parity_shared(addr::Value, phase::Value, ticks::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, phase, ticks, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mbarrier.try_wait.parity.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mma_sync`

The `nvvm.mma.sync` operation collectively performs the operation
`D = matmul(A, B) + C` using all threads in a warp.

All the threads in the warp must execute the same `mma.sync` operation.

For each possible multiplicand PTX data type, there are one or more possible
instruction shapes given as \"mMnNkK\". The below table describes the posssibilities
as well as the types required for the operands. Note that the data type for
C (the accumulator) and D (the result) can vary independently when there are
multiple possibilities in the \"C/D Type\" column.

When an optional attribute cannot be immediately inferred from the types of
the operands and the result during parsing or validation, an error will be
raised.

`b1Op` is only relevant when the binary (b1) type is given to
`multiplicandDataType`. It specifies how the multiply-and-acumulate is
performed and is either `xor_popc` or `and_poc`. The default is `xor_popc`.

`intOverflowBehavior` is only relevant when the `multiplicandType` attribute
is one of `u8, s8, u4, s4`, this attribute describes how overflow is handled
in the accumulator. When the attribute is `satfinite`, the accumulator values
are clamped in the int32 range on overflow. This is the default behavior.
Alternatively, accumulator behavior `wrapped` can also be specified, in
which case overflow wraps from one end of the range to the other.

`layoutA` and `layoutB` are required and should generally be set to
`#nvvm.mma_layout<row>` and `#nvvm.mma_layout<col>` respectively, but other
combinations are possible for certain layouts according to the table below.

```
| A/B Type | Shape     | ALayout | BLayout | A Type   | B Type   | C/D Type          |
|----------|-----------|---------|---------|----------|----------|-------------------|
| f64      | .m8n8k4   | row     | col     | 1x f64   | 1x f64   | 2x f64            |
| f16      | .m8n8k4   | row/col | row/col | 2x f16x2 | 2x f16x2 | 4x f16x2 or 8xf32 |
|          | .m16n8k8  | row     | col     | 2x f16x2 | 1x f16x2 | 2x f16x2 or 4 f32 |
|          | .m16n8k16 | row     | col     | 4x f16x2 | 2x f16x2 | 2x f16x2 or 4 f32 |
| bf16     | .m16n8k8  | row     | col     | 2x f16x2 | 1x f16x2 | 2x f16x2 or 4 f32 |
|          | .m16n8k16 | row     | col     | 4x f16x2 | 2x f16x2 | 2x f16x2 or 4 f32 |
| tf32     | .m16n8k4  | row     | col     | 2x i32   | 1x i32   | 4x f32            |
|          | .m16n8k8  | row     | col     | 4x i32   | 2x i32   | 2x f16x2 or 4 f32 |
| u8/s8    | .m8n8k16  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
|          | .m16n8k16 | row     | col     | 2x i32   | 1x i32   | 4x i32            |
|          | .m16n8k32 | row     | col     | 4x i32   | 2x i32   | 4x i32            |
| u4/s4    | .m8n8k32  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
|          | m16n8k32  | row     | col     | 2x i32   | 1x i32   | 4x i32            |
|          | m16n8k64  | row     | col     | 4x i32   | 2x i32   | 4x i32            |
| b1       | m8n8k128  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
|          | m16n8k128 | row     | col     | 2x i32   | 1x i32   | 4x i32            |
```


# Example
```mlir

%128 = nvvm.mma.sync A[%120, %121, %122, %123]
                     B[%124, %125]
                     C[%126, %127]
                     {layoutA = #nvvm.mma_layout<row>,
                      layoutB = #nvvm.mma_layout<col>,
                      shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}}
    : (vector<2xf16>, vector<2xf16>, vector<2xf16>)
       -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
```
"""
function mma_sync(operandA::Vector{Value}, operandB::Vector{Value}, operandC::Vector{Value}; res::IR.Type, shape, b1Op=nothing, intOverflowBehavior=nothing, layoutA, layoutB, multiplicandAPtxType=nothing, multiplicandBPtxType=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operandA..., operandB..., operandC..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("shape", shape), NamedAttribute("layoutA", layoutA), NamedAttribute("layoutB", layoutB), ]
    push!(attributes, operandsegmentsizes([length(operandA), length(operandB), length(operandC), ]))
    !isnothing(b1Op) && push!(attributes, NamedAttribute("b1Op", b1Op))
    !isnothing(intOverflowBehavior) && push!(attributes, NamedAttribute("intOverflowBehavior", intOverflowBehavior))
    !isnothing(multiplicandAPtxType) && push!(attributes, NamedAttribute("multiplicandAPtxType", multiplicandAPtxType))
    !isnothing(multiplicandBPtxType) && push!(attributes, NamedAttribute("multiplicandBPtxType", multiplicandBPtxType))
    
    IR.create_operation(
        "nvvm.mma.sync", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`rcp_approx_ftz_f`

"""
function rcp_approx_ftz_f(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.rcp.approx.ftz.f", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`redux_sync`

"""
function redux_sync(val::Value, mask_and_clamp::Value; res::IR.Type, kind, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[val, mask_and_clamp, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("kind", kind), ]
    
    IR.create_operation(
        "nvvm.redux.sync", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`shfl_sync`

"""
function shfl_sync(dst::Value, val::Value, offset::Value, mask_and_clamp::Value; res::IR.Type, kind, return_value_and_is_valid=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[dst, val, offset, mask_and_clamp, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("kind", kind), ]
    !isnothing(return_value_and_is_valid) && push!(attributes, NamedAttribute("return_value_and_is_valid", return_value_and_is_valid))
    
    IR.create_operation(
        "nvvm.shfl.sync", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`bar_warp_sync`

"""
function bar_warp_sync(mask::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[mask, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.bar.warp.sync", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`read_ptx_sreg_tid_x`

"""
function read_ptx_sreg_tid_x(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.read.ptx.sreg.tid.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`read_ptx_sreg_tid_y`

"""
function read_ptx_sreg_tid_y(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.read.ptx.sreg.tid.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`read_ptx_sreg_tid_z`

"""
function read_ptx_sreg_tid_z(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.read.ptx.sreg.tid.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`vote_ballot_sync`

"""
function vote_ballot_sync(mask::Value, pred::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[mask, pred, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.vote.ballot.sync", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`wmma_load`

"""
function wmma_load(ptr::Value, stride::Value; res::IR.Type, m, n, k, layout, eltype, frag, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, stride, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("m", m), NamedAttribute("n", n), NamedAttribute("k", k), NamedAttribute("layout", layout), NamedAttribute("eltype", eltype), NamedAttribute("frag", frag), ]
    
    IR.create_operation(
        "nvvm.wmma.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`wmma_mma`

"""
function wmma_mma(args::Vector{Value}; res::IR.Type, m, n, k, layoutA, layoutB, eltypeA, eltypeB, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("m", m), NamedAttribute("n", n), NamedAttribute("k", k), NamedAttribute("layoutA", layoutA), NamedAttribute("layoutB", layoutB), NamedAttribute("eltypeA", eltypeA), NamedAttribute("eltypeB", eltypeB), ]
    
    IR.create_operation(
        "nvvm.wmma.mma", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`wmma_store`

"""
function wmma_store(ptr::Value, args::Vector{Value}, stride::Value; m, n, k, layout, eltype, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, args..., stride, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("m", m), NamedAttribute("n", n), NamedAttribute("k", k), NamedAttribute("layout", layout), NamedAttribute("eltype", eltype), ]
    
    IR.create_operation(
        "nvvm.wmma.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`read_ptx_sreg_warpsize`

"""
function read_ptx_sreg_warpsize(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.read.ptx.sreg.warpsize", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`wgmma_fence_aligned`

Enforce an ordering of register accesses between warpgroup level matrix 
multiplication and other operations. 
See for more information:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence
"""
function wgmma_fence_aligned(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.wgmma.fence.aligned", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`wgmma_commit_group_sync_aligned`

Commits all prior uncommitted warpgroup level matrix multiplication operations.
See for more information:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group
"""
function wgmma_commit_group_sync_aligned(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.wgmma.commit.group.sync.aligned", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`wgmma_wait_group_sync_aligned`

Signal the completion of a preceding warpgroup operation.
See for more information:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group
"""
function wgmma_wait_group_sync_aligned(; group, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("group", group), ]
    
    IR.create_operation(
        "nvvm.wgmma.wait.group.sync.aligned", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end
end # nvvm

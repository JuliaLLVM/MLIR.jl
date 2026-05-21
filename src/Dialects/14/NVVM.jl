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
function cp_async_shared_global(dst::Value, src::Value; size, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dst, src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("size", size), ]
    
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
`mma_sync`

"""
function mma_sync(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "nvvm.mma.sync", location;
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
end # nvvm

module rocdl

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes

"""
`ballot`

Ballot provides a bit mask containing the 1-bit predicate value from each lane.
The nth bit of the result contains the 1 bit contributed by the nth warp lane.
"""
function ballot(pred::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[pred, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.ballot", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function barrier(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.barrier", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function s_barrier_signal(; id, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("id", id), ]
    
    IR.create_operation(
        "rocdl.s.barrier.signal", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function s_barrier_wait(; id, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("id", id), ]
    
    IR.create_operation(
        "rocdl.s.barrier.wait", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workgroup_dim_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    IR.create_operation(
        "rocdl.workgroup.dim.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workgroup_dim_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    IR.create_operation(
        "rocdl.workgroup.dim.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workgroup_dim_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    IR.create_operation(
        "rocdl.workgroup.dim.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workgroup_id_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    IR.create_operation(
        "rocdl.workgroup.id.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workgroup_id_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    IR.create_operation(
        "rocdl.workgroup.id.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workgroup_id_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    IR.create_operation(
        "rocdl.workgroup.id.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_f32_bf8`

Convert 8-bit bf8 value from the `byteSel`th bit of `srcA` to fp32.
"""
function cvt_f32_bf8(srcA::Value, byteSel::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, byteSel, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.f32.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_f32_fp8`

Convert 8-bit fp8 value from the `byteSel`th bit of `srcA` to fp32.
"""
function cvt_f32_fp8(srcA::Value, byteSel::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, byteSel, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.f32.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_pk_bf8_f32`

Convert `srcA` and `srcB` to bf8 and store into the low/high word of
`old`, preserving the other word.
"""
function cvt_pk_bf8_f32(srcA::Value, srcB::Value, old::Value, wordSel::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, srcB, old, wordSel, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.pk.bf8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_pk_fp8_f32`

Convert `srcA` and `srcB` to fp8 and store into the low/high word of
`old`, preserving the other word.
"""
function cvt_pk_fp8_f32(srcA::Value, srcB::Value, old::Value, wordSel::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, srcB, old, wordSel, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.pk.fp8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_pkrtz`

Convert two f32 values into a packed vector<2xf16>.
"""
function cvt_pkrtz(srcA::Value, srcB::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, srcB, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.pkrtz", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_sr_bf8_f32`

Convert `srcA` to bf8, adding the rounding factor from `srcB`,
and store into the `byteSel`th byte of `old`, preserving the others.
"""
function cvt_sr_bf8_f32(srcA::Value, srcB::Value, old::Value, byteSel::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, srcB, old, byteSel, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.sr.bf8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_sr_fp8_f32`

Convert `srcA` to fp8, adding the rounding factor from `srcB`,
and store into the `byteSel`th byte of `old`, preserving the others.
"""
function cvt_sr_fp8_f32(srcA::Value, srcB::Value, old::Value, byteSel::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, srcB, old, byteSel, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.sr.fp8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function update_dpp(old::Value, src::Value; res::IR.Type, dppCtrl, rowMask, bankMask, boundCtrl, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[old, src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dppCtrl", dppCtrl), NamedAttribute("rowMask", rowMask), NamedAttribute("bankMask", bankMask), NamedAttribute("boundCtrl", boundCtrl), ]
    
    IR.create_operation(
        "rocdl.update.dpp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_bpermute(index::Value, src::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[index, src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.ds_bpermute", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_swizzle(src::Value, offset::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, offset, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.ds_swizzle", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function global_load_lds(globalPtr::Value, ldsPtr::Value, size::Value, offset::Value, aux::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[globalPtr, ldsPtr, size, offset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.global.load.lds", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function grid_dim_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    IR.create_operation(
        "rocdl.grid.dim.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function grid_dim_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    IR.create_operation(
        "rocdl.grid.dim.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function grid_dim_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    IR.create_operation(
        "rocdl.grid.dim.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function iglp_opt(; variant, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("variant", variant), ]
    
    IR.create_operation(
        "rocdl.iglp.opt", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function make_buffer_rsrc(base::Value, stride::Value, numRecords::Value, flags::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[base, stride, numRecords, flags, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.make.buffer.rsrc", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbcnt_hi(in0::Value, in1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in0, in1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mbcnt.hi", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbcnt_lo(in0::Value, in1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in0, in1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mbcnt.lo", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_buffer_atomic_cmpswap(src::Value, cmp::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, cmp, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.raw.buffer.atomic.cmpswap", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_buffer_atomic_fadd(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.raw.buffer.atomic.fadd", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_buffer_atomic_fmax(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.raw.buffer.atomic.fmax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_buffer_atomic_smax(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.raw.buffer.atomic.smax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_buffer_atomic_umin(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.raw.buffer.atomic.umin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_buffer_load(rsrc::Value, offset::Value, soffset::Value, aux::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.raw.buffer.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_buffer_store(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.raw.buffer.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_atomic_cmpswap(src::Value, cmp::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; res::IR.Type, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, cmp, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
        "rocdl.raw.ptr.buffer.atomic.cmpswap", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_atomic_fadd(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
        "rocdl.raw.ptr.buffer.atomic.fadd", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_atomic_fmax(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
        "rocdl.raw.ptr.buffer.atomic.fmax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_atomic_smax(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
        "rocdl.raw.ptr.buffer.atomic.smax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_atomic_umin(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
        "rocdl.raw.ptr.buffer.atomic.umin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_load(rsrc::Value, offset::Value, soffset::Value, aux::Value; res::IR.Type, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
        "rocdl.raw.ptr.buffer.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_store(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
        "rocdl.raw.ptr.buffer.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`readlane`

Get the value in lane `src1` from input `src0`.
"""
function readlane(src0::Value, src1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src0, src1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.readlane", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function s_barrier(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.s.barrier", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function sched_barrier(; mask, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mask", mask), ]
    
    IR.create_operation(
        "rocdl.sched.barrier", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function sched_group_barrier(; mask, size, groupId, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mask", mask), NamedAttribute("size", size), NamedAttribute("groupId", groupId), ]
    
    IR.create_operation(
        "rocdl.sched.group.barrier", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function s_setprio(; priority, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("priority", priority), ]
    
    IR.create_operation(
        "rocdl.s.setprio", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workitem_id_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    IR.create_operation(
        "rocdl.workitem.id.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workitem_id_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    IR.create_operation(
        "rocdl.workitem.id.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workitem_id_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    IR.create_operation(
        "rocdl.workitem.id.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function s_wait_dscnt(; id, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("id", id), ]
    
    IR.create_operation(
        "rocdl.s.wait.dscnt", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function waitcnt(; bitfield, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("bitfield", bitfield), ]
    
    IR.create_operation(
        "rocdl.waitcnt", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_read_tr4_b64(ptr::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.ds.read.tr4.b64", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_read_tr6_b96(ptr::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.ds.read.tr6.b96", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_read_tr8_b64(ptr::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.ds.read.tr8.b64", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_read_tr16_b64(ptr::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.ds.read.tr16.b64", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_4x4x1f32(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.4x4x1f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_4x4x2bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.4x4x2bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_4x4x4bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.4x4x4bf16.1k", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_4x4x4f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.4x4x4f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x1f32(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x1f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x2bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x2bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x4bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x4bf16.1k", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x4f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x4f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x4f32(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x4f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x8_xf32(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x8.xf32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x8bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x8bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x16bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x16bf16.1k", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x16f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x16f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x32_bf8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x32.bf8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x32_bf8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x32.bf8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x32_bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x32.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x32_f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x32.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x32_fp8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x32.fp8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x32_fp8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.16x16x32.fp8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x1f32(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x1f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x2bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x2bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x2f32(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x2f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x4_xf32(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x4.xf32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x4bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x4bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x4bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x4bf16.1k", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x4f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x4f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x8bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x8bf16.1k", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x8f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x8f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x16_bf8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x16.bf8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x16_bf8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x16.bf8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x16_bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x16.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x16_f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x16.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x16_fp8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x16.fp8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x16_fp8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f32.32x32x16.fp8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f64_4x4x4f64(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f64.4x4x4f64", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f64_16x16x4f64(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.f64.16x16x4f64", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_4x4x4i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.i32.4x4x4i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_16x16x4i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.i32.16x16x4i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_16x16x16i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.i32.16x16x16i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_16x16x32_i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.i32.16x16x32.i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_16x16x64_i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.i32.16x16x64.i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_32x32x4i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.i32.32x32x4i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_32x32x8i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.i32.32x32x8i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_32x32x16_i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.i32.32x32x16.i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_32x32x32_i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.i32.32x32x32.i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_scale_f32_16x16x128_f8f6f4(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.scale.f32.16x16x128.f8f6f4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_scale_f32_32x32x64_f8f6f4(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.mfma.scale.f32.32x32x64.f8f6f4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_16x16x32_bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.smfmac.f32.16x16x32.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_16x16x32_f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.smfmac.f32.16x16x32.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_16x16x64_bf8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.smfmac.f32.16x16x64.bf8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_16x16x64_bf8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.smfmac.f32.16x16x64.bf8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_16x16x64_fp8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.smfmac.f32.16x16x64.fp8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_16x16x64_fp8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.smfmac.f32.16x16x64.fp8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_32x32x16_bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.smfmac.f32.32x32x16.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_32x32x16_f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.smfmac.f32.32x32x16.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_32x32x32_bf8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.smfmac.f32.32x32x32.bf8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_32x32x32_bf8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.smfmac.f32.32x32x32.bf8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_32x32x32_fp8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.smfmac.f32.32x32x32.fp8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_32x32x32_fp8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.smfmac.f32.32x32x32.fp8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_i32_16x16x64_i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.smfmac.i32.16x16x64.i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_i32_32x32x32_i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.smfmac.i32.32x32x32.i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_bf16_16x16x16_bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.wmma.bf16.16x16x16.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_f16_16x16x16_f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.wmma.f16.16x16x16.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_f32_16x16x16_bf8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.wmma.f32.16x16x16.bf8_bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_f32_16x16x16_bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.wmma.f32.16x16x16.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_f32_16x16x16_f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.wmma.f32.16x16x16.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_f32_16x16x16_fp8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.wmma.f32.16x16x16.fp8_fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_i32_16x16x16_iu4(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.wmma.i32.16x16x16.iu4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_i32_16x16x16_iu8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.wmma.i32.16x16x16.iu8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end
end # rocdl

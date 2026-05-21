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
function cvt_f32_bf8(srcA::Value; res::IR.Type, byteSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("byteSel", byteSel), ]
    
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
function cvt_f32_fp8(srcA::Value; res::IR.Type, byteSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("byteSel", byteSel), ]
    
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
function cvt_pk_bf8_f32(srcA::Value, srcB::Value, old::Value; res::IR.Type, wordSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, srcB, old, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("wordSel", wordSel), ]
    
    IR.create_operation(
        "rocdl.cvt.pk.bf8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_pk_f32_bf8`

Convert `src` based on \$wordSel to packed fp32,
"""
function cvt_pk_f32_bf8(src::Value; res::IR.Type, wordSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("wordSel", wordSel), ]
    
    IR.create_operation(
        "rocdl.cvt.pk.f32.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_pk_f32_fp8`

Convert `src` based on \$wordSel to packed fp32.
"""
function cvt_pk_f32_fp8(src::Value; res::IR.Type, wordSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("wordSel", wordSel), ]
    
    IR.create_operation(
        "rocdl.cvt.pk.f32.fp8", location;
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
function cvt_pk_fp8_f32(srcA::Value, srcB::Value, old::Value; res::IR.Type, wordSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, srcB, old, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("wordSel", wordSel), ]
    
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
`cvt_scalef32_f16_bf8`

Convert a bf8 byte from `src`, selected by
`srcSelIndex`, to f16 while multiplying it by the expontent of `scale`,
and place it into the `dstLoHiSel`th bit
of `oldVdst` preserving the other element of that vector in
the return value.

The bytes are stored as an `i32` and not a `<4 x i8>`.
"""
function cvt_scalef32_f16_bf8(oldVdst::Value, src::Value, scale::Value; res::IR.Type, srcSelIndex, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcSelIndex", srcSelIndex), NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.f16.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_f16_fp8`

Convert a fp8 byte from `src`, selected by
`srcSelIndex`, to f16 while multiplying it by the expontent of `scale`,
and place it into the `dstLoHiSel`th bit
of `oldVdst` preserving the other element of that vector in
the return value.

The bytes are stored as an `i32` and not a `<4 x i8>`.
"""
function cvt_scalef32_f16_fp8(oldVdst::Value, src::Value, scale::Value; res::IR.Type, srcSelIndex, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcSelIndex", srcSelIndex), NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.f16.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_f32_bf8`

Convert a bf8 byte from `src`, selected by
`srcSelIndex`, to f32, multiplying it by the exponent of `scale`.

The bytes are stored in an `i32`, not a `<4 x i8>`.
"""
function cvt_scalef32_f32_bf8(src::Value, scale::Value; res::IR.Type, srcSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcSelIndex", srcSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.f32.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_f32_fp8`

Convert a fp8 byte from `src`, selected by
`srcSelIndex`, to f32, multiplying it by the exponent of `scale`.

The bytes are stored in an `i32`, not a `<4 x i8>`.
"""
function cvt_scalef32_f32_fp8(src::Value, scale::Value; res::IR.Type, srcSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcSelIndex", srcSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.f32.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_bf6_bf16`

Convert 32 packed bf16 values to packed bf6, dividing by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_bf6_bf16(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk32.bf6.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_bf6_f16`

Convert 32 packed f16 values to packed bf6, dividing by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_bf6_f16(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk32.bf6.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_bf16_bf6`

Convert 32 packed bf6 values to packed bf16, multiplying by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_bf16_bf6(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk32.bf16.bf6", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_bf16_fp6`

Convert 32 packed fp6 values to packed bf16, multiplying by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_bf16_fp6(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk32.bf16.fp6", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_f16_bf6`

Convert 32 packed bf6 values to packed f16, multiplying by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_f16_bf6(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk32.f16.bf6", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_f16_fp6`

Convert 32 packed fp6 values to packed f16, multiplying by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_f16_fp6(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk32.f16.fp6", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_f32_bf6`

Convert 32 packed bf6 values to packed f32, multiplying by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_f32_bf6(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk32.f32.bf6", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_f32_fp6`

Convert 32 packed fp6 values to packed f32, multiplying by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_f32_fp6(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk32.f32.fp6", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_fp6_bf16`

Convert 32 packed bf16 values to packed fp6, dividing by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_fp6_bf16(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk32.fp6.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_fp6_f16`

Convert 32 packed f16 values to packed fp6, dividing by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_fp6_f16(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk32.fp6.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_bf8_bf16`

Convert two bf16 values in `src0` to two bf8 bytes, dividing by the exponent in `scale`. The bytes are
packed into a 16-bit value which is inserted into `oldVdst` at the
`dstLoHiSel` position, with the entire updated vector being returned.
"""
function cvt_scalef32_pk_bf8_bf16(oldVdst::Value, src0::Value, scale::Value; res::IR.Type, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.bf8.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_bf8_f16`

Convert two f16 values in `src0` to two bf8 bytes, dividing by the exponent in `scale`. The bytes are
packed into a 16-bit value which is inserted into `oldVdst` at the
`dstLoHiSel` position, with the entire updated vector being returned.
"""
function cvt_scalef32_pk_bf8_f16(oldVdst::Value, src0::Value, scale::Value; res::IR.Type, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.bf8.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_bf8_f32`

Convert two f32 values in `src0` and `src1` to two bf8 bytes,
dividing by the exponent in `scale`. The bytes are packed into
a 16-bit value which is inserted into `oldVdst` at the `dstLoHiSel`
position, with the entire updated vector being returned.
"""
function cvt_scalef32_pk_bf8_f32(oldVdst::Value, src0::Value, src1::Value, scale::Value; res::IR.Type, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, src1, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.bf8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_bf16_bf8`

Convert two packed bf8 values in `src0` to two bf16 values, multiplying by the exponent in `scale`.
The two values to be converted are selected from the low or high half
of `src` (a packed vector represented as an `i32`)
on the basis of `srcLoHiSel`.
"""
function cvt_scalef32_pk_bf16_bf8(src::Value, scale::Value; res::IR.Type, srcLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcLoHiSel", srcLoHiSel), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.bf16.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_bf16_fp4`

Convert two packed fp4 (f4E2M1) values  stored as one byte of a 32-bit integer
to packed bf16, multiplying by the exponent part of `scale`
before doing so.

The byte to convert is chosen by `srcSelIndex`.
"""
function cvt_scalef32_pk_bf16_fp4(src::Value, scale::Value; res::IR.Type, srcSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcSelIndex", srcSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.bf16.fp4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_bf16_fp8`

Convert two packed fp8 values in `src0` to two bf16 values, multiplying by the exponent in `scale`.
The two values to be converted are selected from the low or high half
of `src` (a packed vector represented as an `i32`)
on the basis of `srcLoHiSel`.
"""
function cvt_scalef32_pk_bf16_fp8(src::Value, scale::Value; res::IR.Type, srcLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcLoHiSel", srcLoHiSel), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.bf16.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_f16_bf8`

Convert two packed bf8 values in `src0` to two f16 values, multiplying by the exponent in `scale`.
The two values to be converted are selected from the low or high half
of `src` (a packed vector represented as an `i32`)
on the basis of `srcLoHiSel`.
"""
function cvt_scalef32_pk_f16_bf8(src::Value, scale::Value; res::IR.Type, srcLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcLoHiSel", srcLoHiSel), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.f16.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_f16_fp4`

Convert two packed fp4 (f4E2M1) values  stored as one byte of a 32-bit integer
to packed f16, multiplying by the exponent part of `scale`
before doing so.

The byte to convert is chosen by `srcSelIndex`.
"""
function cvt_scalef32_pk_f16_fp4(src::Value, scale::Value; res::IR.Type, srcSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcSelIndex", srcSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.f16.fp4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_f16_fp8`

Convert two packed fp8 values in `src0` to two f16 values, multiplying by the exponent in `scale`.
The two values to be converted are selected from the low or high half
of `src` (a packed vector represented as an `i32`)
on the basis of `srcLoHiSel`.
"""
function cvt_scalef32_pk_f16_fp8(src::Value, scale::Value; res::IR.Type, srcLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcLoHiSel", srcLoHiSel), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.f16.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_f32_bf8`

Convert two packed bf8 values in `src0` to two f32 values, multiplying by the exponent in `scale`.
The two values to be converted are selected from the low or high half
of `src` (a packed vector represented as an `i32`)
on the basis of `srcLoHiSel`.
"""
function cvt_scalef32_pk_f32_bf8(src::Value, scale::Value; res::IR.Type, srcLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcLoHiSel", srcLoHiSel), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.f32.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_f32_fp4`

Convert two packed fp4 (f4E2M1) values  stored as one byte of a 32-bit integer
to packed f32, multiplying by the exponent part of `scale`
before doing so.

The byte to convert is chosen by `srcSelIndex`.
"""
function cvt_scalef32_pk_f32_fp4(src::Value, scale::Value; res::IR.Type, srcSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcSelIndex", srcSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.f32.fp4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_f32_fp8`

Convert two packed fp8 values in `src0` to two f32 values, multiplying by the exponent in `scale`.
The two values to be converted are selected from the low or high half
of `src` (a packed vector represented as an `i32`)
on the basis of `srcLoHiSel`.
"""
function cvt_scalef32_pk_f32_fp8(src::Value, scale::Value; res::IR.Type, srcLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcLoHiSel", srcLoHiSel), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.f32.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_fp4_bf16`

Convert two packed bf16 values to packed
fp4, dividing by the exponent part of `scale`
before doing so.

The two scaled values are packed  into a byte.
That byte is used to update the `dstSelIndex`th
byte of `oldVdst`, which is returned in its entirity.
"""
function cvt_scalef32_pk_fp4_bf16(oldVdst::Value, src::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.fp4.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_fp4_f16`

Convert two packed f16 values to packed
fp4, dividing by the exponent part of `scale`
before doing so.

The two scaled values are packed  into a byte.
That byte is used to update the `dstSelIndex`th
byte of `oldVdst`, which is returned in its entirity.
"""
function cvt_scalef32_pk_fp4_f16(oldVdst::Value, src::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.fp4.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_fp4_f32`

Convert two single-precision float values, passed in `src0` and `src1`
into two fp4 values, dividing them by the expontent part of `scale`
before doing so.

The two scaled values are packed  into a byte.
That byte is used to update the `dstSelIndex`th
byte of `oldVdst`, which is returned in its entirity.
"""
function cvt_scalef32_pk_fp4_f32(oldVdst::Value, src0::Value, src1::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, src1, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.fp4.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_fp8_bf16`

Convert two bf16 values in `src0` to two fp8 bytes, dividing by the exponent in `scale`. The bytes are
packed into a 16-bit value which is inserted into `oldVdst` at the
`dstLoHiSel` position, with the entire updated vector being returned.
"""
function cvt_scalef32_pk_fp8_bf16(oldVdst::Value, src0::Value, scale::Value; res::IR.Type, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.fp8.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_fp8_f16`

Convert two f16 values in `src0` to two fp8 bytes, dividing by the exponent in `scale`. The bytes are
packed into a 16-bit value which is inserted into `oldVdst` at the
`dstLoHiSel` position, with the entire updated vector being returned.
"""
function cvt_scalef32_pk_fp8_f16(oldVdst::Value, src0::Value, scale::Value; res::IR.Type, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.fp8.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_fp8_f32`

Convert two f32 values in `src0` and `src1` to two fp8 bytes,
dividing by the exponent in `scale`. The bytes are packed into
a 16-bit value which is inserted into `oldVdst` at the `dstLoHiSel`
position, with the entire updated vector being returned.
"""
function cvt_scalef32_pk_fp8_f32(oldVdst::Value, src0::Value, src1::Value, scale::Value; res::IR.Type, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, src1, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.pk.fp8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_bf8_bf16`

Convert a bf16 value in `src0` to a bf8 bytes, dividing by the exponent in `scale` and using `seed`
for stochiastic rounding. Place the resulting byte in the
`dstSelIndex`th bit of `oldVdst` and return the entire packed vector,
which is stored as an `i32`.
"""
function cvt_scalef32_sr_bf8_bf16(oldVdst::Value, src0::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.bf8.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_bf8_f16`

Convert a f16 value in `src0` to a bf8 bytes, dividing by the exponent in `scale` and using `seed`
for stochiastic rounding. Place the resulting byte in the
`dstSelIndex`th bit of `oldVdst` and return the entire packed vector,
which is stored as an `i32`.
"""
function cvt_scalef32_sr_bf8_f16(oldVdst::Value, src0::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.bf8.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_bf8_f32`

Convert a f32 value in `src0` to a bf8 bytes, dividing by the exponent in `scale` and using `seed`
for stochiastic rounding. Place the resulting byte in the
`dstSelIndex`th bit of `oldVdst` and return the entire packed vector,
which is stored as an `i32`.
"""
function cvt_scalef32_sr_bf8_f32(oldVdst::Value, src0::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.bf8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_fp8_bf16`

Convert a bf16 value in `src0` to a fp8 bytes, dividing by the exponent in `scale` and using `seed`
for stochiastic rounding. Place the resulting byte in the
`dstSelIndex`th bit of `oldVdst` and return the entire packed vector,
which is stored as an `i32`.
"""
function cvt_scalef32_sr_fp8_bf16(oldVdst::Value, src0::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.fp8.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_fp8_f16`

Convert a f16 value in `src0` to a fp8 bytes, dividing by the exponent in `scale` and using `seed`
for stochiastic rounding. Place the resulting byte in the
`dstSelIndex`th bit of `oldVdst` and return the entire packed vector,
which is stored as an `i32`.
"""
function cvt_scalef32_sr_fp8_f16(oldVdst::Value, src0::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.fp8.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_fp8_f32`

Convert a f32 value in `src0` to a fp8 bytes, dividing by the exponent in `scale` and using `seed`
for stochiastic rounding. Place the resulting byte in the
`dstSelIndex`th bit of `oldVdst` and return the entire packed vector,
which is stored as an `i32`.
"""
function cvt_scalef32_sr_fp8_f32(oldVdst::Value, src0::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.fp8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk32_bf6_bf16`

Convert 32 packed bf16 values to packed bf6, dividing by the exponent part of `scale`
before doing so and applying random rounding derived from
`seed`.
"""
function cvt_scalef32_sr_pk32_bf6_bf16(src::Value, seed::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.pk32.bf6.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk32_bf6_f16`

Convert 32 packed f16 values to packed bf6, dividing by the exponent part of `scale`
before doing so and applying random rounding derived from
`seed`.
"""
function cvt_scalef32_sr_pk32_bf6_f16(src::Value, seed::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.pk32.bf6.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk32_bf6_f32`

Convert 32 packed f32 values to packed bf6, dividing by the exponent part of `scale`
before doing so and applying random rounding derived from
`seed`.
"""
function cvt_scalef32_sr_pk32_bf6_f32(src::Value, seed::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.pk32.bf6.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk32_fp6_bf16`

Convert 32 packed bf16 values to packed fp6, dividing by the exponent part of `scale`
before doing so and applying random rounding derived from
`seed`.
"""
function cvt_scalef32_sr_pk32_fp6_bf16(src::Value, seed::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.pk32.fp6.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk32_fp6_f16`

Convert 32 packed f16 values to packed fp6, dividing by the exponent part of `scale`
before doing so and applying random rounding derived from
`seed`.
"""
function cvt_scalef32_sr_pk32_fp6_f16(src::Value, seed::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.pk32.fp6.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk32_fp6_f32`

Convert 32 packed f32 values to packed fp6, dividing by the exponent part of `scale`
before doing so and applying random rounding derived from
`seed`.
"""
function cvt_scalef32_sr_pk32_fp6_f32(src::Value, seed::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.pk32.fp6.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk_fp4_bf16`

Convert two packed bf16 values to packed
fp4, dividing by the exponent part of `scale`
before doing so and using `seed` as the random seed for
stochiastic rounding.

The two scaled values are packed (little-endian)
into a byte. That byte is used to update the `dstSelIndex`th
byte of `oldVdst`, which is returned in its entirity.
"""
function cvt_scalef32_sr_pk_fp4_bf16(oldVdst::Value, src::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.pk.fp4.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk_fp4_f16`

Convert two packed f16 values to packed
fp4, dividing by the exponent part of `scale`
before doing so and using `seed` as the random seed for
stochiastic rounding.

The two scaled values are packed (little-endian)
into a byte. That byte is used to update the `dstSelIndex`th
byte of `oldVdst`, which is returned in its entirity.
"""
function cvt_scalef32_sr_pk_fp4_f16(oldVdst::Value, src::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.pk.fp4.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk_fp4_f32`

Convert two packed f32 values to packed
fp4, dividing by the exponent part of `scale`
before doing so and using `seed` as the random seed for
stochiastic rounding.

The two scaled values are packed (little-endian)
into a byte. That byte is used to update the `dstSelIndex`th
byte of `oldVdst`, which is returned in its entirity.
"""
function cvt_scalef32_sr_pk_fp4_f32(oldVdst::Value, src::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.sr.pk.fp4.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_2xpk16_bf6_f32`

Convert 32 single-precision float values, packed into two length-16
vectors that will be logically concanenated, to packed bf6, dividing by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_2xpk16_bf6_f32(src0::Value, src1::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src0, src1, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.2xpk16.bf6.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_2xpk16_fp6_f32`

Convert 32 single-precision float values, packed into two length-16
vectors that will be logically concanenated, to packed fp6, dividing by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_2xpk16_fp6_f32(src0::Value, src1::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src0, src1, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.cvt.scalef32.2xpk16.fp6.f32", location;
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
function cvt_sr_bf8_f32(srcA::Value, srcB::Value, old::Value; res::IR.Type, byteSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, srcB, old, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("byteSel", byteSel), ]
    
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
function cvt_sr_fp8_f32(srcA::Value, srcB::Value, old::Value; res::IR.Type, byteSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, srcB, old, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("byteSel", byteSel), ]
    
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


function global_load_lds(globalPtr::Value, ldsPtr::Value; size, offset, aux, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[globalPtr, ldsPtr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("size", size), NamedAttribute("offset", offset), NamedAttribute("aux", aux), ]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
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


function load_to_lds(globalPtr::Value, ldsPtr::Value; size, offset, aux, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[globalPtr, ldsPtr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("size", size), NamedAttribute("offset", offset), NamedAttribute("aux", aux), ]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
        "rocdl.load.to.lds", location;
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

"""
`permlanex16`

Performs a `permlanex16` operation with the given operands, applying the
permutation specified by \$fi to the provided inputs.
"""
function permlanex16(old::Value, src0::Value, src1::Value, src2::Value; res::IR.Type, fi, boundControl, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[old, src0, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("fi", fi), NamedAttribute("boundControl", boundControl), ]
    
    IR.create_operation(
        "rocdl.permlanex16", location;
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


function raw_ptr_buffer_load_lds(rsrc::Value, ldsPtr::Value, size::Value, voffset::Value, soffset::Value, offset::Value, aux::Value; alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[rsrc, ldsPtr, size, voffset, soffset, offset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
        "rocdl.raw.ptr.buffer.load.lds", location;
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


function s_sleep(; count, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("count", count), ]
    
    IR.create_operation(
        "rocdl.s.sleep", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function s_waitcnt(; bitfield, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("bitfield", bitfield), ]
    
    IR.create_operation(
        "rocdl.s.waitcnt", location;
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


function wavefrontsize(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    IR.create_operation(
        "rocdl.wavefrontsize", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_read_tr4_b64(ptr::Value; res::IR.Type, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
        "rocdl.ds.read.tr4.b64", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_read_tr6_b96(ptr::Value; res::IR.Type, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
        "rocdl.ds.read.tr6.b96", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_read_tr8_b64(ptr::Value; res::IR.Type, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
        "rocdl.ds.read.tr8.b64", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_read_tr16_b64(ptr::Value; res::IR.Type, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
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


function wmma_f32_16x16x16_bf8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.wmma.f32.16x16x16.bf8_fp8", location;
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


function wmma_f32_16x16x16_fp8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.wmma.f32.16x16x16.fp8_bf8", location;
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


function wmma_i32_16x16x32_iu4(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.wmma.i32.16x16x32.iu4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end
end # rocdl

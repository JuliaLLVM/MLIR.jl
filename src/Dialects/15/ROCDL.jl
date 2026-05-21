module rocdl

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes

"""
`barrier`

"""
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

"""
`workgroup_dim_x`

"""
function workgroup_dim_x(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.workgroup.dim.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`workgroup_dim_y`

"""
function workgroup_dim_y(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.workgroup.dim.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`workgroup_dim_z`

"""
function workgroup_dim_z(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.workgroup.dim.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`workgroup_id_x`

"""
function workgroup_id_x(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.workgroup.id.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`workgroup_id_y`

"""
function workgroup_id_y(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.workgroup.id.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`workgroup_id_z`

"""
function workgroup_id_z(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.workgroup.id.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`grid_dim_x`

"""
function grid_dim_x(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.grid.dim.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`grid_dim_y`

"""
function grid_dim_y(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.grid.dim.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`grid_dim_z`

"""
function grid_dim_z(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.grid.dim.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`buffer_load`

"""
function buffer_load(rsrc::Value, vindex::Value, offset::Value, glc::Value, slc::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[rsrc, vindex, offset, glc, slc, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.buffer.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`buffer_store`

"""
function buffer_store(vdata::Value, rsrc::Value, vindex::Value, offset::Value, glc::Value, slc::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, vindex, offset, glc, slc, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.buffer.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`raw_buffer_atomic_fadd`

"""
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

"""
`raw_buffer_load`

"""
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

"""
`raw_buffer_store`

"""
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

"""
`workitem_id_x`

"""
function workitem_id_x(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.workitem.id.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`workitem_id_y`

"""
function workitem_id_y(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.workitem.id.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`workitem_id_z`

"""
function workitem_id_z(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "rocdl.workitem.id.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mfma_f32_16x16x16bf16_1k`

"""
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

"""
`mfma_f32_16x16x16f16`

"""
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

"""
`mfma_f32_16x16x1f32`

"""
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

"""
`mfma_f32_16x16x2bf16`

"""
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

"""
`mfma_f32_16x16x4bf16_1k`

"""
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

"""
`mfma_f32_16x16x4f16`

"""
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

"""
`mfma_f32_16x16x4f32`

"""
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

"""
`mfma_f32_16x16x8_xf32`

"""
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

"""
`mfma_f32_16x16x8bf16`

"""
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

"""
`mfma_f32_32x32x1f32`

"""
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

"""
`mfma_f32_32x32x2bf16`

"""
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

"""
`mfma_f32_32x32x2f32`

"""
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

"""
`mfma_f32_32x32x4_xf32`

"""
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

"""
`mfma_f32_32x32x4bf16`

"""
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

"""
`mfma_f32_32x32x4bf16_1k`

"""
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

"""
`mfma_f32_32x32x4f16`

"""
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

"""
`mfma_f32_32x32x8bf16_1k`

"""
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

"""
`mfma_f32_32x32x8f16`

"""
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

"""
`mfma_f32_4x4x1f32`

"""
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

"""
`mfma_f32_4x4x2bf16`

"""
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

"""
`mfma_f32_4x4x4bf16_1k`

"""
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

"""
`mfma_f32_4x4x4f16`

"""
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

"""
`mfma_f64_16x16x4f64`

"""
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

"""
`mfma_f64_4x4x4f64`

"""
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

"""
`mfma_i32_16x16x16i8`

"""
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

"""
`mfma_i32_16x16x32_i8`

"""
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

"""
`mfma_i32_16x16x4i8`

"""
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

"""
`mfma_i32_32x32x16_i8`

"""
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

"""
`mfma_i32_32x32x4i8`

"""
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

"""
`mfma_i32_32x32x8i8`

"""
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

"""
`mfma_i32_4x4x4i8`

"""
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
end # rocdl

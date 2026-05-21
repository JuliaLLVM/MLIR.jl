module amx

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes

"""
`tile_load`

Loads a tile from memory defined by a base and indices, with the
shape defined by the 2-dim vector type of the result. This is
eventually lowered into the \"tileloadd\" instruction with the
corresponding tile configuration.

# Example

```mlir
  %0 = amx.tile_load %arg0[%c0, %c0] : memref<?x?xi8> into !amx.tile<16x64xi8>
```
"""
function tile_load(base::Value, indices::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[base, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "amx.tile_load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tile_mulf`

Multiplies a \"m x k\" tile with a \"k x n\" tile and accumulates the results
into a \"m x n\" destination tile. Supports \"f32 <- bf16 x bf16\" (with
pairs of \"bf16\"). The operation is eventually lowered into the
\"tdpbf16ps\" instruction with the corresponding tile configuration.

# Example

```mlir
  %0 = amx.tile_mulf %a, %b, %c
    : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
```
"""
function tile_mulf(lhs::Value, rhs::Value, acc::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, acc, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "amx.tile_mulf", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tile_muli`

Multiplies a \"m x k\" tile with a \"k x n\" tile and accumulates the results
into a \"m x n\" destination tile. Supports all \"si32 <- s/ui8 x s/ui8\"
combinations (4 bytes packed into dwords in the columns of both the
source operand tiles; the zero or sign extension is specified with
the attributes and default to sign extended). The operation is eventually
lowered into one of the \"tdpbssd\", \"tdpbsud\", \"tdpbusd\", or \"tdpbuud\"
instructions with the corresponding tile configuration.

# Example

```mlir
  %0 = amx.tile_muli %a zext, %b zext, %c
    : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
```
"""
function tile_muli(lhs::Value, rhs::Value, acc::Value; res::IR.Type, isZextLhs=nothing, isZextRhs=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, acc, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(isZextLhs) && push!(attributes, NamedAttribute("isZextLhs", isZextLhs))
    !isnothing(isZextRhs) && push!(attributes, NamedAttribute("isZextRhs", isZextRhs))
    
    IR.create_operation(
        "amx.tile_muli", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tile_store`

Stores a tile to memory defined by a base and indices, with the
shape defined by the 2-dim vector type of the value. This is
eventually lowered into the \"tilestored\" instruction with the
corresponding tile configuration.

# Example

```mlir
  amx.tile_store %arg1[%c0, %c0], %0 : memref<?x?xi8>, !amx.tile<16x64xi8>
```
"""
function tile_store(base::Value, indices::Vector{Value}, val::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[base, indices..., val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "amx.tile_store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tile_zero`

Zeroes the destination tile, with the shape defined by the 2-dim
vector type of the result. This is eventually lowered into the
\"tilezero\" instruction with the corresponding tile configuration.

# Example

```mlir
  %0 = amx.tile_zero : !amx.tile<16x16xbf16>
```
"""
function tile_zero(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "amx.tile_zero", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end
end # amx

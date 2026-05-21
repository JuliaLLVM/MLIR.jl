module ptr

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes

"""
`from_ptr`

The `from_ptr` operation casts a `ptr` value to a ptr-like object. It\'s
important to note that:
- The ptr-like object cannot be a `!ptr.ptr`.
- The memory-space of both the `ptr` and ptr-like object must match.
- The cast is Pure (no UB and side-effect free).

The optional `metadata` operand exists to provide any ptr-like metadata
that might be required to perform the cast.

# Example

```mlir
%typed_ptr = ptr.from_ptr %ptr : !ptr.ptr<#ptr.generic_space> -> !my.ptr<f32, #ptr.generic_space>
%memref = ptr.from_ptr %ptr metadata %md : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
  
// Cast the `%ptr` to a memref without utilizing metadata.
%memref = ptr.from_ptr %ptr : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
```
"""
function from_ptr(ptr::Value, metadata=nothing::Union{Nothing, Value}; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(metadata) && push!(operands, metadata)
    
    IR.create_operation(
        "ptr.from_ptr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`get_metadata`

The `get_metadata` operation produces an opaque value that encodes the
metadata of the ptr-like type.

# Example

```mlir
%metadata = ptr.get_metadata %memref : memref<?x?xf32>
```
"""
function get_metadata(ptr::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "ptr.get_metadata", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`ptr_add`

The `ptr_add` operation adds an integer offset to a pointer to produce a new
pointer. The input and output pointer types are always the same.

# Example

```mlir
%x_off  = ptr.ptr_add %x, %off : !ptr.ptr<#ptr.generic_space>, i32
%x_off0 = ptr.ptr_add nusw %x, %off : !ptr.ptr<#ptr.generic_space>, i32
```
"""
function ptr_add(base::Value, offset::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[base, offset, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "ptr.ptr_add", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`to_ptr`

The `to_ptr` operation casts a ptr-like object to a `!ptr.ptr`. It\'s
important to note that:
- The ptr-like object cannot be a `!ptr.ptr`.
- The memory-space of both the `ptr` and ptr-like object must match.
- The cast is side-effect free.

# Example

```mlir
%ptr0 = ptr.to_ptr %my_ptr : !my.ptr<f32, #ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
%ptr1 = ptr.to_ptr %memref : memref<f32, #ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
```
"""
function to_ptr(ptr::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "ptr.to_ptr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`type_offset`

The `type_offset` operation produces an int or index-typed SSA value
equal to a target-specific constant representing the offset of a single
element of the given type.

# Example

```mlir
// Return the offset between two f32 stored in memory
%0 = ptr.type_offset f32 : index
// Return the offset between two memref descriptors stored in memory
%1 = ptr.type_offset memref<12 x f64> : i32
```
"""
function type_offset(; result::IR.Type, elementType, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("elementType", elementType), ]
    
    IR.create_operation(
        "ptr.type_offset", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end
end # ptr

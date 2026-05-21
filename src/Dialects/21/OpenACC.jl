module acc

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes

"""
`atomic_capture`

This operation performs an atomic capture.

The region has the following allowed forms:

```
  acc.atomic.capture {
    acc.atomic.update ...
    acc.atomic.read ...
    acc.terminator
  }

  acc.atomic.capture {
    acc.atomic.read ...
    acc.atomic.update ...
    acc.terminator
  }

  acc.atomic.capture {
    acc.atomic.read ...
    acc.atomic.write ...
    acc.terminator
  }
```
"""
function atomic_capture(; region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "acc.atomic.capture", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`atomic_read`

This operation performs an atomic read.

The operand `x` is the address from where the value is atomically read.
The operand `v` is the address where the value is stored after reading.
"""
function atomic_read(x::Value, v::Value; element_type, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[x, v, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("element_type", element_type), ]
    
    IR.create_operation(
        "acc.atomic.read", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`atomic_update`

This operation performs an atomic update.

The operand `x` is exactly the same as the operand `x` in the OpenACC
Standard (OpenACC 3.3, section 2.12). It is the address of the variable
that is being updated. `x` is atomically read/written.

The region describes how to update the value of `x`. It takes the value at
`x` as an input and must yield the updated value. Only the update to `x` is
atomic. Generally the region must have only one instruction, but can
potentially have more than one instructions too. The update is sematically
similar to a compare-exchange loop based atomic update.

The syntax of atomic update operation is different from atomic read and
atomic write operations. This is because only the host dialect knows how to
appropriately update a value. For example, while generating LLVM IR, if
there are no special `atomicrmw` instructions for the operation-type
combination in atomic update, a compare-exchange loop is generated, where
the core update operation is directly translated like regular operations by
the host dialect. The front-end must handle semantic checks for allowed
operations.
"""
function atomic_update(x::Value; region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[x, ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "acc.atomic.update", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`atomic_write`

This operation performs an atomic write.

The operand `x` is the address to where the `expr` is atomically
written w.r.t. multiple threads. The evaluation of `expr` need not be
atomic w.r.t. the write to address. In general, the type(x) must
dereference to type(expr).
"""
function atomic_write(x::Value, expr::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[x, expr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "acc.atomic.write", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`attach`

Description of arguments:
- `var`: The variable to copy. Must be either `MappableType` or
`PointerLikeType`.
- `varType`: The type of the variable that is being copied. When `var` is
a `MappableType`, this matches the type of `var`. When `var` is a
`PointerLikeType`, this type holds information about the target of the
pointer.
- `varPtrPtr`: Specifies the address of the address of `var` - only used
when the variable copied is a field in a struct. This is important for
OpenACC due to implicit attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data entry operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function attach(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.attach", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cache`

Description of arguments:
- `var`: The variable to copy. Must be either `MappableType` or
`PointerLikeType`.
- `varType`: The type of the variable that is being copied. When `var` is
a `MappableType`, this matches the type of `var`. When `var` is a
`PointerLikeType`, this type holds information about the target of the
pointer.
- `varPtrPtr`: Specifies the address of the address of `var` - only used
when the variable copied is a field in a struct. This is important for
OpenACC due to implicit attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data entry operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function cache(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.cache", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`copyin`

Description of arguments:
- `var`: The variable to copy. Must be either `MappableType` or
`PointerLikeType`.
- `varType`: The type of the variable that is being copied. When `var` is
a `MappableType`, this matches the type of `var`. When `var` is a
`PointerLikeType`, this type holds information about the target of the
pointer.
- `varPtrPtr`: Specifies the address of the address of `var` - only used
when the variable copied is a field in a struct. This is important for
OpenACC due to implicit attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data entry operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function copyin(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.copyin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`copyout`
- `varPtr`: The address of variable to copy back to.
    - `accVar`: The acc variable. This is the link from the data-entry
    operation used.
    - `bounds`: Used when copying just slice of array or array\'s bounds are not
    encoded in type. They are in rank order where rank 0 is inner-most dimension.
    - `asyncOperands` and `asyncOperandsDeviceType`:
    pair-wise lists of the async clause values associated with device_type\'s.
    - `asyncOnly`: a list of device_type\'s for which async clause
    does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
    - `dataClause`: Keeps track of the data clause the user used. This is because
    the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
    `acc.copyin` and `acc.copyout` operations, but both have dataClause that
    specifies `acc_copy` in this field.
    - `structured`: Flag to note whether this is associated with structured region
    (parallel, kernels, data) or unstructured (enter data, exit data). This is
    important due to spec specifically calling out structured and dynamic reference
    counters (2.6.7).
    - `implicit`: Whether this is an implicitly generated operation, such as copies
    done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
    - `modifiers`: Keeps track of the data clause modifiers (eg zero, always, etc)
    - `name`: Holds the name of variable as specified in user clause (including bounds).

    The async values attached to the data exit operation imply that the data
    action applies to all device types specified by the device_type clauses
    using the activity queues on these devices as defined by the async values.
"""
function copyout(accVar::Value, var::Value, bounds::Vector{Value}, asyncOperands::Vector{Value}; varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[accVar, var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    push!(attributes, operandsegmentsizes([1, 1, length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.copyout", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`create`

Description of arguments:
- `var`: The variable to copy. Must be either `MappableType` or
`PointerLikeType`.
- `varType`: The type of the variable that is being copied. When `var` is
a `MappableType`, this matches the type of `var`. When `var` is a
`PointerLikeType`, this type holds information about the target of the
pointer.
- `varPtrPtr`: Specifies the address of the address of `var` - only used
when the variable copied is a field in a struct. This is important for
OpenACC due to implicit attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data entry operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function create(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.create", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`bounds`

This operation is used to record bounds used in acc data clause in a
normalized fashion (zero-based). This works well with the `PointerLikeType`
requirement in data clauses - since a `lowerbound` of 0 means looking
at data at the zero offset from pointer.

The operation must have an `upperbound` or `extent` (or both are allowed -
but not checked for consistency). When the source language\'s arrays are
not zero-based, the `startIdx` must specify the zero-position index.

The `stride` represents the distance between consecutive elements. For
multi-dimensional arrays, the `stride` for each outer dimension must account
for the complete size of all inner dimensions.

The `strideInBytes` flag indicates that the `stride` is specified in bytes
rather than the number of elements.

Examples below show copying a slice of 10-element array except first element.
Note that the examples use extent in data clause for C++ and upperbound
for Fortran (as per 2.7.1). To simplify examples, the constants are used
directly in the acc.bounds operands - this is not the syntax of operation.

C++:
```
int array[10];
#pragma acc copy(array[1:9])
```
=>
```mlir
acc.bounds lb(1) ub(9) extent(9) startIdx(0) stride(1)
```

Fortran:
```
integer :: array(1:10)
!\$acc copy(array(2:10))
```
=>
```mlir
acc.bounds lb(1) ub(9) extent(9) startIdx(1) stride(1)
```
"""
function bounds(lowerbound=nothing::Union{Nothing, Value}; upperbound=nothing::Union{Nothing, Value}, extent=nothing::Union{Nothing, Value}, stride=nothing::Union{Nothing, Value}, startIdx=nothing::Union{Nothing, Value}, result::IR.Type, strideInBytes=nothing, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(lowerbound) && push!(operands, lowerbound)
    !isnothing(upperbound) && push!(operands, upperbound)
    !isnothing(extent) && push!(operands, extent)
    !isnothing(stride) && push!(operands, stride)
    !isnothing(startIdx) && push!(operands, startIdx)
    push!(attributes, operandsegmentsizes([Int(!isnothing(lowerbound)), Int(!isnothing(upperbound)), Int(!isnothing(extent)), Int(!isnothing(stride)), Int(!isnothing(startIdx)), ]))
    !isnothing(strideInBytes) && push!(attributes, NamedAttribute("strideInBytes", strideInBytes))
    
    IR.create_operation(
        "acc.bounds", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`data`

The \"acc.data\" operation represents a data construct. It defines vars to
be allocated in the current device memory for the duration of the region,
whether data should be copied from local memory to the current device
memory upon region entry , and copied from device memory to local memory
upon region exit.

# Example

```mlir
acc.data present(%a: memref<10x10xf32>, %b: memref<10x10xf32>,
    %c: memref<10xf32>, %d: memref<10xf32>) {
  // data region
}
```

`async` and `wait` operands are supported with `device_type` information.
They should only be accessed by the extra provided getters. If modified,
the corresponding `device_type` attributes must be modified as well.
"""
function data(ifCond=nothing::Union{Nothing, Value}; asyncOperands::Vector{Value}, waitOperands::Vector{Value}, dataClauseOperands::Vector{Value}, asyncOperandsDeviceType=nothing, asyncOnly=nothing, waitOperandsSegments=nothing, waitOperandsDeviceType=nothing, hasWaitDevnum=nothing, waitOnly=nothing, defaultAttr=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[asyncOperands..., waitOperands..., dataClauseOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([Int(!isnothing(ifCond)), length(asyncOperands), length(waitOperands), length(dataClauseOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(waitOperandsSegments) && push!(attributes, NamedAttribute("waitOperandsSegments", waitOperandsSegments))
    !isnothing(waitOperandsDeviceType) && push!(attributes, NamedAttribute("waitOperandsDeviceType", waitOperandsDeviceType))
    !isnothing(hasWaitDevnum) && push!(attributes, NamedAttribute("hasWaitDevnum", hasWaitDevnum))
    !isnothing(waitOnly) && push!(attributes, NamedAttribute("waitOnly", waitOnly))
    !isnothing(defaultAttr) && push!(attributes, NamedAttribute("defaultAttr", defaultAttr))
    
    IR.create_operation(
        "acc.data", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`declare_device_resident`

Description of arguments:
- `var`: The variable to copy. Must be either `MappableType` or
`PointerLikeType`.
- `varType`: The type of the variable that is being copied. When `var` is
a `MappableType`, this matches the type of `var`. When `var` is a
`PointerLikeType`, this type holds information about the target of the
pointer.
- `varPtrPtr`: Specifies the address of the address of `var` - only used
when the variable copied is a field in a struct. This is important for
OpenACC due to implicit attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data entry operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function declare_device_resident(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.declare_device_resident", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`declare_enter`

The \"acc.declare_enter\" operation represents the OpenACC declare directive
and captures the entry semantics to the implicit data region.
This operation is modeled similarly to \"acc.enter_data\".

Example showing `acc declare create(a)`:

```mlir
%0 = acc.create varPtr(%a : !llvm.ptr) -> !llvm.ptr
acc.declare_enter dataOperands(%0 : !llvm.ptr)
```
"""
function declare_enter(dataClauseOperands::Vector{Value}; token::IR.Type, location=Location())
    op_ty_results = IR.Type[token, ]
    operands = Value[dataClauseOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "acc.declare_enter", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`declare_exit`

The \"acc.declare_exit\" operation represents the OpenACC declare directive
and captures the exit semantics from the implicit data region.
This operation is modeled similarly to \"acc.exit_data\".

Example showing `acc declare device_resident(a)`:

```mlir
%0 = acc.getdeviceptr varPtr(%a : !llvm.ptr) -> !llvm.ptr {dataClause = #acc<data_clause declare_device_resident>}
acc.declare_exit dataOperands(%0 : !llvm.ptr)
acc.delete accPtr(%0 : !llvm.ptr) {dataClause = #acc<data_clause declare_device_resident>}
```
"""
function declare_exit(token=nothing::Union{Nothing, Value}; dataClauseOperands::Vector{Value}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dataClauseOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(token) && push!(operands, token)
    push!(attributes, operandsegmentsizes([Int(!isnothing(token)), length(dataClauseOperands), ]))
    
    IR.create_operation(
        "acc.declare_exit", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`declare_link`

Description of arguments:
- `var`: The variable to copy. Must be either `MappableType` or
`PointerLikeType`.
- `varType`: The type of the variable that is being copied. When `var` is
a `MappableType`, this matches the type of `var`. When `var` is a
`PointerLikeType`, this type holds information about the target of the
pointer.
- `varPtrPtr`: Specifies the address of the address of `var` - only used
when the variable copied is a field in a struct. This is important for
OpenACC due to implicit attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data entry operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function declare_link(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.declare_link", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`declare`

The \"acc.declare\" operation represents an implicit declare region in
function (and subroutine in Fortran).

# Example

```mlir
%pa = acc.present varPtr(%a : memref<10x10xf32>) -> memref<10x10xf32>
acc.declare dataOperands(%pa: memref<10x10xf32>) {
  // implicit region
}
```
"""
function declare(dataClauseOperands::Vector{Value}; region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dataClauseOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "acc.declare", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`delete`

- `accVar`: The acc variable. This is the link from the data-entry
operation used.
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, always, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data exit operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function delete(accVar::Value, bounds::Vector{Value}, asyncOperands::Vector{Value}; asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[accVar, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([1, length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.delete", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`detach`

- `accVar`: The acc variable. This is the link from the data-entry
operation used.
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, always, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data exit operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function detach(accVar::Value, bounds::Vector{Value}, asyncOperands::Vector{Value}; asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[accVar, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([1, length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.detach", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`deviceptr`

Description of arguments:
- `var`: The variable to copy. Must be either `MappableType` or
`PointerLikeType`.
- `varType`: The type of the variable that is being copied. When `var` is
a `MappableType`, this matches the type of `var`. When `var` is a
`PointerLikeType`, this type holds information about the target of the
pointer.
- `varPtrPtr`: Specifies the address of the address of `var` - only used
when the variable copied is a field in a struct. This is important for
OpenACC due to implicit attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data entry operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function deviceptr(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.deviceptr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`enter_data`

The \"acc.enter_data\" operation represents the OpenACC enter data directive.

# Example

```mlir
acc.enter_data create(%d1 : memref<10xf32>) attributes {async}
```
"""
function enter_data(ifCond=nothing::Union{Nothing, Value}; asyncOperand=nothing::Union{Nothing, Value}, waitDevnum=nothing::Union{Nothing, Value}, waitOperands::Vector{Value}, dataClauseOperands::Vector{Value}, async=nothing, wait=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[waitOperands..., dataClauseOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(asyncOperand) && push!(operands, asyncOperand)
    !isnothing(waitDevnum) && push!(operands, waitDevnum)
    push!(attributes, operandsegmentsizes([Int(!isnothing(ifCond)), Int(!isnothing(asyncOperand)), Int(!isnothing(waitDevnum)), length(waitOperands), length(dataClauseOperands), ]))
    !isnothing(async) && push!(attributes, NamedAttribute("async", async))
    !isnothing(wait) && push!(attributes, NamedAttribute("wait", wait))
    
    IR.create_operation(
        "acc.enter_data", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`exit_data`

The \"acc.exit_data\" operation represents the OpenACC exit data directive.

# Example

```mlir
acc.exit_data delete(%d1 : memref<10xf32>) attributes {async}
```
"""
function exit_data(ifCond=nothing::Union{Nothing, Value}; asyncOperand=nothing::Union{Nothing, Value}, waitDevnum=nothing::Union{Nothing, Value}, waitOperands::Vector{Value}, dataClauseOperands::Vector{Value}, async=nothing, wait=nothing, finalize=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[waitOperands..., dataClauseOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(asyncOperand) && push!(operands, asyncOperand)
    !isnothing(waitDevnum) && push!(operands, waitDevnum)
    push!(attributes, operandsegmentsizes([Int(!isnothing(ifCond)), Int(!isnothing(asyncOperand)), Int(!isnothing(waitDevnum)), length(waitOperands), length(dataClauseOperands), ]))
    !isnothing(async) && push!(attributes, NamedAttribute("async", async))
    !isnothing(wait) && push!(attributes, NamedAttribute("wait", wait))
    !isnothing(finalize) && push!(attributes, NamedAttribute("finalize", finalize))
    
    IR.create_operation(
        "acc.exit_data", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`firstprivate`

Description of arguments:
- `var`: The variable to copy. Must be either `MappableType` or
`PointerLikeType`.
- `varType`: The type of the variable that is being copied. When `var` is
a `MappableType`, this matches the type of `var`. When `var` is a
`PointerLikeType`, this type holds information about the target of the
pointer.
- `varPtrPtr`: Specifies the address of the address of `var` - only used
when the variable copied is a field in a struct. This is important for
OpenACC due to implicit attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data entry operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function firstprivate(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.firstprivate", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`firstprivate_recipe`

Declares an OpenACC privatization recipe with copy of the initial value.
The operation requires two mandatory regions and one optional.

  1. The initializer region specifies how to allocate and initialize a new
     private value. For example in Fortran, a derived-type might have a
     default initialization. The region has an argument that contains the
     value that need to be privatized. This is useful if the type is not
     known at compile time and the private value is needed to create its
     copy.
  2. The copy region specifies how to copy the initial value to the newly
     created private value. It takes the initial value and the privatized
     value as arguments.
  3. The destroy region specifies how to destruct the value when it reaches
     its end of life. It takes the privatized value as argument. It is
     optional.

A single privatization recipe can be used for multiple operand if they have
the same type and do not require a specific default initialization.

# Example

```mlir
acc.firstprivate.recipe @privatization_f32 : f32 init {
^bb0(%0: f32):
  // init region contains a sequence of operations to create and
  // initialize the copy if needed. It yields the create copy.
} copy {
^bb0(%0: f32, %1: !llvm.ptr):
  // copy region contains a sequence of operations to copy the initial value
  // of the firstprivate value to the newly created value.
} destroy {
^bb0(%0: f32)
  // destroy region contains a sequences of operations to destruct the
  // created copy.
}

// The privatization symbol is then used in the corresponding operation.
acc.parallel firstprivate(@privatization_f32 -> %a : f32) {
}
```
"""
function firstprivate_recipe(; sym_name, type, initRegion::Region, copyRegion::Region, destroyRegion::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[initRegion, copyRegion, destroyRegion, ]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), NamedAttribute("type", type), ]
    
    IR.create_operation(
        "acc.firstprivate.recipe", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`getdeviceptr`

This operation is used to get the `accPtr` for a variable. This is often
used in conjunction with data exit operations when the data entry
operation is not visible. This operation can have a `dataClause` argument
that is any of the valid `mlir::acc::DataClause` entries.
\\
    
    Description of arguments:
    - `var`: The variable to copy. Must be either `MappableType` or
    `PointerLikeType`.
    - `varType`: The type of the variable that is being copied. When `var` is
    a `MappableType`, this matches the type of `var`. When `var` is a
    `PointerLikeType`, this type holds information about the target of the
    pointer.
    - `varPtrPtr`: Specifies the address of the address of `var` - only used
    when the variable copied is a field in a struct. This is important for
    OpenACC due to implicit attach semantics on data clauses (2.6.4).
    - `bounds`: Used when copying just slice of array or array\'s bounds are not
    encoded in type. They are in rank order where rank 0 is inner-most dimension.
    - `asyncOperands` and `asyncOperandsDeviceType`:
    pair-wise lists of the async clause values associated with device_type\'s.
    - `asyncOnly`: a list of device_type\'s for which async clause
    does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
    - `dataClause`: Keeps track of the data clause the user used. This is because
    the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
    `acc.copyin` and `acc.copyout` operations, but both have dataClause that
    specifies `acc_copy` in this field.
    - `structured`: Flag to note whether this is associated with structured region
    (parallel, kernels, data) or unstructured (enter data, exit data). This is
    important due to spec specifically calling out structured and dynamic reference
    counters (2.6.7).
    - `implicit`: Whether this is an implicitly generated operation, such as copies
    done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
    - `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
    - `name`: Holds the name of variable as specified in user clause (including bounds).

    The async values attached to the data entry operation imply that the data
    action applies to all device types specified by the device_type clauses
    using the activity queues on these devices as defined by the async values.
"""
function getdeviceptr(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.getdeviceptr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`global_ctor`

The \"acc.global_ctor\" operation is used to capture OpenACC actions to apply
on globals (such as `acc declare`) at the entry to the implicit data region.
This operation is isolated and intended to be used in a module.

Example showing `declare create` of global:

```mlir
llvm.mlir.global external @globalvar() : i32 {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}
acc.global_ctor @acc_constructor {
  %0 = llvm.mlir.addressof @globalvar : !llvm.ptr
  %1 = acc.create varPtr(%0 : !llvm.ptr) -> !llvm.ptr
  acc.declare_enter dataOperands(%1 : !llvm.ptr)
}
```
"""
function global_ctor(; sym_name, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), ]
    
    IR.create_operation(
        "acc.global_ctor", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`global_dtor`

The \"acc.global_dtor\" operation is used to capture OpenACC actions to apply
on globals (such as `acc declare`) at the exit from the implicit data
region. This operation is isolated and intended to be used in a module.

Example showing delete associated with `declare create` of global:

```mlir
llvm.mlir.global external @globalvar() : i32 {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}
acc.global_dtor @acc_destructor {
  %0 = llvm.mlir.addressof @globalvar : !llvm.ptr
  %1 = acc.getdeviceptr varPtr(%0 : !llvm.ptr) -> !llvm.ptr {dataClause = #acc<data_clause create>}
  acc.declare_exit dataOperands(%1 : !llvm.ptr)
  acc.delete accPtr(%1 : !llvm.ptr) {dataClause = #acc<data_clause create>}
}
```
"""
function global_dtor(; sym_name, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), ]
    
    IR.create_operation(
        "acc.global_dtor", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`host_data`

The \"acc.host_data\" operation represents the OpenACC host_data construct.

# Example

```mlir
%0 = acc.use_device varPtr(%a : !llvm.ptr) -> !llvm.ptr
acc.host_data dataOperands(%0 : !llvm.ptr) {

}
```
"""
function host_data(ifCond=nothing::Union{Nothing, Value}; dataClauseOperands::Vector{Value}, ifPresent=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dataClauseOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([Int(!isnothing(ifCond)), length(dataClauseOperands), ]))
    !isnothing(ifPresent) && push!(attributes, NamedAttribute("ifPresent", ifPresent))
    
    IR.create_operation(
        "acc.host_data", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`init`

The \"acc.init\" operation represents the OpenACC init executable
directive.

# Example

```mlir
acc.init
acc.init device_num(%dev1 : i32)
```
"""
function init(deviceNum=nothing::Union{Nothing, Value}; ifCond=nothing::Union{Nothing, Value}, device_types=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(deviceNum) && push!(operands, deviceNum)
    !isnothing(ifCond) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([Int(!isnothing(deviceNum)), Int(!isnothing(ifCond)), ]))
    !isnothing(device_types) && push!(attributes, NamedAttribute("device_types", device_types))
    
    IR.create_operation(
        "acc.init", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`kernels`

The \"acc.kernels\" operation represents a kernels construct block. It has
one region to be compiled into a sequence of kernels for execution on the
current device.

# Example

```mlir
acc.kernels num_gangs(%c10) num_workers(%c10)
    private(%c : memref<10xf32>) {
  // kernels region
}
```

`collapse`, `gang`, `worker`, `vector`, `seq`, `independent`, `auto` and
`tile` operands are supported with `device_type` information. They should
only be accessed by the extra provided getters. If modified, the
corresponding `device_type` attributes must be modified as well.
"""
function kernels(asyncOperands::Vector{Value}, waitOperands::Vector{Value}, numGangs::Vector{Value}, numWorkers::Vector{Value}, vectorLength::Vector{Value}, ifCond=nothing::Union{Nothing, Value}; selfCond=nothing::Union{Nothing, Value}, dataClauseOperands::Vector{Value}, asyncOperandsDeviceType=nothing, asyncOnly=nothing, waitOperandsSegments=nothing, waitOperandsDeviceType=nothing, hasWaitDevnum=nothing, waitOnly=nothing, numGangsSegments=nothing, numGangsDeviceType=nothing, numWorkersDeviceType=nothing, vectorLengthDeviceType=nothing, selfAttr=nothing, defaultAttr=nothing, combined=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[asyncOperands..., waitOperands..., numGangs..., numWorkers..., vectorLength..., dataClauseOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(selfCond) && push!(operands, selfCond)
    push!(attributes, operandsegmentsizes([length(asyncOperands), length(waitOperands), length(numGangs), length(numWorkers), length(vectorLength), Int(!isnothing(ifCond)), Int(!isnothing(selfCond)), length(dataClauseOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(waitOperandsSegments) && push!(attributes, NamedAttribute("waitOperandsSegments", waitOperandsSegments))
    !isnothing(waitOperandsDeviceType) && push!(attributes, NamedAttribute("waitOperandsDeviceType", waitOperandsDeviceType))
    !isnothing(hasWaitDevnum) && push!(attributes, NamedAttribute("hasWaitDevnum", hasWaitDevnum))
    !isnothing(waitOnly) && push!(attributes, NamedAttribute("waitOnly", waitOnly))
    !isnothing(numGangsSegments) && push!(attributes, NamedAttribute("numGangsSegments", numGangsSegments))
    !isnothing(numGangsDeviceType) && push!(attributes, NamedAttribute("numGangsDeviceType", numGangsDeviceType))
    !isnothing(numWorkersDeviceType) && push!(attributes, NamedAttribute("numWorkersDeviceType", numWorkersDeviceType))
    !isnothing(vectorLengthDeviceType) && push!(attributes, NamedAttribute("vectorLengthDeviceType", vectorLengthDeviceType))
    !isnothing(selfAttr) && push!(attributes, NamedAttribute("selfAttr", selfAttr))
    !isnothing(defaultAttr) && push!(attributes, NamedAttribute("defaultAttr", defaultAttr))
    !isnothing(combined) && push!(attributes, NamedAttribute("combined", combined))
    
    IR.create_operation(
        "acc.kernels", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`loop`

The \"acc.loop\" operation represents the OpenACC loop construct. The lower
and upper bounds specify a half-open range: the range includes the lower
bound but does not include the upper bound. If the `inclusive` attribute is
set then the upper bound is included.

# Example

```mlir
acc.loop gang() vector() (%arg3 : index, %arg4 : index, %arg5 : index) = 
    (%c0, %c0, %c0 : index, index, index) to 
    (%c10, %c10, %c10 : index, index, index) step 
    (%c1, %c1, %c1 : index, index, index) {
  // Loop body
  acc.yield
} attributes { collapse = [3] }
```

`collapse`, `gang`, `worker`, `vector`, `seq`, `independent`, `auto` and
`tile` operands are supported with `device_type` information. They should
only be accessed by the extra provided getters. If modified, the
corresponding `device_type` attributes must be modified as well.
"""
function loop(lowerbound::Vector{Value}, upperbound::Vector{Value}, step::Vector{Value}, gangOperands::Vector{Value}, workerNumOperands::Vector{Value}, vectorOperands::Vector{Value}, tileOperands::Vector{Value}, cacheOperands::Vector{Value}, privateOperands::Vector{Value}, reductionOperands::Vector{Value}; results::Vector{IR.Type}, inclusiveUpperbound=nothing, collapse=nothing, collapseDeviceType=nothing, gangOperandsArgType=nothing, gangOperandsSegments=nothing, gangOperandsDeviceType=nothing, workerNumOperandsDeviceType=nothing, vectorOperandsDeviceType=nothing, seq=nothing, independent=nothing, auto_=nothing, gang=nothing, worker=nothing, vector=nothing, tileOperandsSegments=nothing, tileOperandsDeviceType=nothing, privatizationRecipes=nothing, reductionRecipes=nothing, combined=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[results..., ]
    operands = Value[lowerbound..., upperbound..., step..., gangOperands..., workerNumOperands..., vectorOperands..., tileOperands..., cacheOperands..., privateOperands..., reductionOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(lowerbound), length(upperbound), length(step), length(gangOperands), length(workerNumOperands), length(vectorOperands), length(tileOperands), length(cacheOperands), length(privateOperands), length(reductionOperands), ]))
    !isnothing(inclusiveUpperbound) && push!(attributes, NamedAttribute("inclusiveUpperbound", inclusiveUpperbound))
    !isnothing(collapse) && push!(attributes, NamedAttribute("collapse", collapse))
    !isnothing(collapseDeviceType) && push!(attributes, NamedAttribute("collapseDeviceType", collapseDeviceType))
    !isnothing(gangOperandsArgType) && push!(attributes, NamedAttribute("gangOperandsArgType", gangOperandsArgType))
    !isnothing(gangOperandsSegments) && push!(attributes, NamedAttribute("gangOperandsSegments", gangOperandsSegments))
    !isnothing(gangOperandsDeviceType) && push!(attributes, NamedAttribute("gangOperandsDeviceType", gangOperandsDeviceType))
    !isnothing(workerNumOperandsDeviceType) && push!(attributes, NamedAttribute("workerNumOperandsDeviceType", workerNumOperandsDeviceType))
    !isnothing(vectorOperandsDeviceType) && push!(attributes, NamedAttribute("vectorOperandsDeviceType", vectorOperandsDeviceType))
    !isnothing(seq) && push!(attributes, NamedAttribute("seq", seq))
    !isnothing(independent) && push!(attributes, NamedAttribute("independent", independent))
    !isnothing(auto_) && push!(attributes, NamedAttribute("auto_", auto_))
    !isnothing(gang) && push!(attributes, NamedAttribute("gang", gang))
    !isnothing(worker) && push!(attributes, NamedAttribute("worker", worker))
    !isnothing(vector) && push!(attributes, NamedAttribute("vector", vector))
    !isnothing(tileOperandsSegments) && push!(attributes, NamedAttribute("tileOperandsSegments", tileOperandsSegments))
    !isnothing(tileOperandsDeviceType) && push!(attributes, NamedAttribute("tileOperandsDeviceType", tileOperandsDeviceType))
    !isnothing(privatizationRecipes) && push!(attributes, NamedAttribute("privatizationRecipes", privatizationRecipes))
    !isnothing(reductionRecipes) && push!(attributes, NamedAttribute("reductionRecipes", reductionRecipes))
    !isnothing(combined) && push!(attributes, NamedAttribute("combined", combined))
    
    IR.create_operation(
        "acc.loop", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`nocreate`

Description of arguments:
- `var`: The variable to copy. Must be either `MappableType` or
`PointerLikeType`.
- `varType`: The type of the variable that is being copied. When `var` is
a `MappableType`, this matches the type of `var`. When `var` is a
`PointerLikeType`, this type holds information about the target of the
pointer.
- `varPtrPtr`: Specifies the address of the address of `var` - only used
when the variable copied is a field in a struct. This is important for
OpenACC due to implicit attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data entry operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function nocreate(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.nocreate", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`parallel`

The \"acc.parallel\" operation represents a parallel construct block. It has
one region to be executed in parallel on the current device.

# Example

```mlir
acc.parallel num_gangs(%c10) num_workers(%c10)
    private(%c : memref<10xf32>) {
  // parallel region
}
```

`async`, `wait`, `num_gangs`, `num_workers` and `vector_length` operands are
supported with `device_type` information. They should only be accessed by
the extra provided getters. If modified, the corresponding `device_type`
attributes must be modified as well.
"""
function parallel(asyncOperands::Vector{Value}, waitOperands::Vector{Value}, numGangs::Vector{Value}, numWorkers::Vector{Value}, vectorLength::Vector{Value}, ifCond=nothing::Union{Nothing, Value}; selfCond=nothing::Union{Nothing, Value}, reductionOperands::Vector{Value}, privateOperands::Vector{Value}, firstprivateOperands::Vector{Value}, dataClauseOperands::Vector{Value}, asyncOperandsDeviceType=nothing, asyncOnly=nothing, waitOperandsSegments=nothing, waitOperandsDeviceType=nothing, hasWaitDevnum=nothing, waitOnly=nothing, numGangsSegments=nothing, numGangsDeviceType=nothing, numWorkersDeviceType=nothing, vectorLengthDeviceType=nothing, selfAttr=nothing, reductionRecipes=nothing, privatizationRecipes=nothing, firstprivatizationRecipes=nothing, defaultAttr=nothing, combined=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[asyncOperands..., waitOperands..., numGangs..., numWorkers..., vectorLength..., reductionOperands..., privateOperands..., firstprivateOperands..., dataClauseOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(selfCond) && push!(operands, selfCond)
    push!(attributes, operandsegmentsizes([length(asyncOperands), length(waitOperands), length(numGangs), length(numWorkers), length(vectorLength), Int(!isnothing(ifCond)), Int(!isnothing(selfCond)), length(reductionOperands), length(privateOperands), length(firstprivateOperands), length(dataClauseOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(waitOperandsSegments) && push!(attributes, NamedAttribute("waitOperandsSegments", waitOperandsSegments))
    !isnothing(waitOperandsDeviceType) && push!(attributes, NamedAttribute("waitOperandsDeviceType", waitOperandsDeviceType))
    !isnothing(hasWaitDevnum) && push!(attributes, NamedAttribute("hasWaitDevnum", hasWaitDevnum))
    !isnothing(waitOnly) && push!(attributes, NamedAttribute("waitOnly", waitOnly))
    !isnothing(numGangsSegments) && push!(attributes, NamedAttribute("numGangsSegments", numGangsSegments))
    !isnothing(numGangsDeviceType) && push!(attributes, NamedAttribute("numGangsDeviceType", numGangsDeviceType))
    !isnothing(numWorkersDeviceType) && push!(attributes, NamedAttribute("numWorkersDeviceType", numWorkersDeviceType))
    !isnothing(vectorLengthDeviceType) && push!(attributes, NamedAttribute("vectorLengthDeviceType", vectorLengthDeviceType))
    !isnothing(selfAttr) && push!(attributes, NamedAttribute("selfAttr", selfAttr))
    !isnothing(reductionRecipes) && push!(attributes, NamedAttribute("reductionRecipes", reductionRecipes))
    !isnothing(privatizationRecipes) && push!(attributes, NamedAttribute("privatizationRecipes", privatizationRecipes))
    !isnothing(firstprivatizationRecipes) && push!(attributes, NamedAttribute("firstprivatizationRecipes", firstprivatizationRecipes))
    !isnothing(defaultAttr) && push!(attributes, NamedAttribute("defaultAttr", defaultAttr))
    !isnothing(combined) && push!(attributes, NamedAttribute("combined", combined))
    
    IR.create_operation(
        "acc.parallel", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`present`

Description of arguments:
- `var`: The variable to copy. Must be either `MappableType` or
`PointerLikeType`.
- `varType`: The type of the variable that is being copied. When `var` is
a `MappableType`, this matches the type of `var`. When `var` is a
`PointerLikeType`, this type holds information about the target of the
pointer.
- `varPtrPtr`: Specifies the address of the address of `var` - only used
when the variable copied is a field in a struct. This is important for
OpenACC due to implicit attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data entry operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function present(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.present", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`private`

Description of arguments:
- `var`: The variable to copy. Must be either `MappableType` or
`PointerLikeType`.
- `varType`: The type of the variable that is being copied. When `var` is
a `MappableType`, this matches the type of `var`. When `var` is a
`PointerLikeType`, this type holds information about the target of the
pointer.
- `varPtrPtr`: Specifies the address of the address of `var` - only used
when the variable copied is a field in a struct. This is important for
OpenACC due to implicit attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data entry operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function private(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.private", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`private_recipe`

Declares an OpenACC privatization recipe. The operation requires one
mandatory and one optional region.

  1. The initializer region specifies how to allocate and initialize a new
     private value. For example in Fortran, a derived-type might have a
     default initialization. The region has an argument that contains the
     value that need to be privatized. This is useful if the type is not
     known at compile time and the private value is needed to create its
     copy.
  2. The destroy region specifies how to destruct the value when it reaches
     its end of life. It takes the privatized value as argument.

A single privatization recipe can be used for multiple operand if they have
the same type and do not require a specific default initialization.

# Example

```mlir
acc.private.recipe @privatization_f32 : f32 init {
^bb0(%0: f32):
  // init region contains a sequence of operations to create and
  // initialize the copy if needed. It yields the create copy.
} destroy {
^bb0(%0: f32)
  // destroy region contains a sequences of operations to destruct the
  // created copy.
}

// The privatization symbol is then used in the corresponding operation.
acc.parallel private(@privatization_f32 -> %a : f32) {
}
```
"""
function private_recipe(; sym_name, type, initRegion::Region, destroyRegion::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[initRegion, destroyRegion, ]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), NamedAttribute("type", type), ]
    
    IR.create_operation(
        "acc.private.recipe", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`reduction`

Description of arguments:
- `var`: The variable to copy. Must be either `MappableType` or
`PointerLikeType`.
- `varType`: The type of the variable that is being copied. When `var` is
a `MappableType`, this matches the type of `var`. When `var` is a
`PointerLikeType`, this type holds information about the target of the
pointer.
- `varPtrPtr`: Specifies the address of the address of `var` - only used
when the variable copied is a field in a struct. This is important for
OpenACC due to implicit attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data entry operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function reduction(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.reduction", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`reduction_recipe`

Declares an OpenACC reduction recipe. The operation requires two
mandatory regions.

  1. The initializer region specifies how to initialize the local reduction
     value. The region has a first argument that contains the value of the
     reduction accumulator at the start of the reduction. It is expected to
     `acc.yield` the new value. Extra arguments can be added to deal with
     dynamic arrays.
  2. The reduction region contains a sequences of operations to combine two
     values of the reduction type into one. It has at least two arguments
     and it is expected to `acc.yield` the combined value. Extra arguments
     can be added to deal with dynamic arrays.

# Example

```mlir
acc.reduction.recipe @reduction_add_i64 : i64 reduction_operator<add> init {
^bb0(%0: i64):
  // init region contains a sequence of operations to initialize the local
  // reduction value as specified in 2.5.15
  %c0 = arith.constant 0 : i64
  acc.yield %c0 : i64
} combiner {
^bb0(%0: i64, %1: i64)
  // combiner region contains a sequence of operations to combine
  // two values into one.
  %2 = arith.addi %0, %1 : i64
  acc.yield %2 : i64
}

// The reduction symbol is then used in the corresponding operation.
acc.parallel reduction(@reduction_add_i64 -> %a : i64) {
}
```

The following table lists the valid operators and the initialization values
according to OpenACC 3.3:

|------------------------------------------------|
|        C/C++          |        Fortran         |
|-----------------------|------------------------|
| operator | init value | operator | init value  |
|     +    |      0     |     +    |      0      |
|     *    |      1     |     *    |      1      |
|    max   |    least   |    max   |    least    |
|    min   |   largest  |    min   |   largest   |
|     &    |     ~0     |   iand   | all bits on |
|     |    |      0     |    ior   |      0      |
|     ^    |      0     |   ieor   |      0      |
|    &&    |      1     |   .and.  |    .true.   |
|    ||    |      0     |    .or.  |   .false.   |
|          |            |   .eqv.  |    .true.   |
|          |            |  .neqv.  |   .false.   |
-------------------------------------------------|
"""
function reduction_recipe(; sym_name, type, reductionOperator, initRegion::Region, combinerRegion::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[initRegion, combinerRegion, ]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), NamedAttribute("type", type), NamedAttribute("reductionOperator", reductionOperator), ]
    
    IR.create_operation(
        "acc.reduction.recipe", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`routine`

The `acc.routine` operation is used to capture the clauses of acc
routine directive, including the associated function name. The associated
function keeps track of its corresponding routine declaration through
the `RoutineInfoAttr`.

# Example

```mlir
func.func @acc_func(%a : i64) -> () attributes 
    {acc.routine_info = #acc.routine_info<[@acc_func_rout1]>} {
  return
}
acc.routine @acc_func_rout1 func(@acc_func) gang
```

`bind`, `gang`, `worker`, `vector` and `seq` operands are supported with
`device_type` information. They should only be accessed by the extra
provided getters. If modified, the corresponding `device_type` attributes
must be modified as well.
"""
function routine(; sym_name, func_name, bindName=nothing, bindNameDeviceType=nothing, worker=nothing, vector=nothing, seq=nothing, nohost=nothing, implicit=nothing, gang=nothing, gangDim=nothing, gangDimDeviceType=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), NamedAttribute("func_name", func_name), ]
    !isnothing(bindName) && push!(attributes, NamedAttribute("bindName", bindName))
    !isnothing(bindNameDeviceType) && push!(attributes, NamedAttribute("bindNameDeviceType", bindNameDeviceType))
    !isnothing(worker) && push!(attributes, NamedAttribute("worker", worker))
    !isnothing(vector) && push!(attributes, NamedAttribute("vector", vector))
    !isnothing(seq) && push!(attributes, NamedAttribute("seq", seq))
    !isnothing(nohost) && push!(attributes, NamedAttribute("nohost", nohost))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(gang) && push!(attributes, NamedAttribute("gang", gang))
    !isnothing(gangDim) && push!(attributes, NamedAttribute("gangDim", gangDim))
    !isnothing(gangDimDeviceType) && push!(attributes, NamedAttribute("gangDimDeviceType", gangDimDeviceType))
    
    IR.create_operation(
        "acc.routine", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`serial`

The \"acc.serial\" operation represents a serial construct block. It has
one region to be executed in serial on the current device.

# Example

```mlir
acc.serial private(%c : memref<10xf32>) {
  // serial region
}
```

`async` and `wait` operands are supported with `device_type` information.
They should only be accessed by the extra provided getters. If modified,
the corresponding `device_type` attributes must be modified as well.
"""
function serial(asyncOperands::Vector{Value}, waitOperands::Vector{Value}, ifCond=nothing::Union{Nothing, Value}; selfCond=nothing::Union{Nothing, Value}, reductionOperands::Vector{Value}, privateOperands::Vector{Value}, firstprivateOperands::Vector{Value}, dataClauseOperands::Vector{Value}, asyncOperandsDeviceType=nothing, asyncOnly=nothing, waitOperandsSegments=nothing, waitOperandsDeviceType=nothing, hasWaitDevnum=nothing, waitOnly=nothing, selfAttr=nothing, reductionRecipes=nothing, privatizationRecipes=nothing, firstprivatizationRecipes=nothing, defaultAttr=nothing, combined=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[asyncOperands..., waitOperands..., reductionOperands..., privateOperands..., firstprivateOperands..., dataClauseOperands..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    !isnothing(selfCond) && push!(operands, selfCond)
    push!(attributes, operandsegmentsizes([length(asyncOperands), length(waitOperands), Int(!isnothing(ifCond)), Int(!isnothing(selfCond)), length(reductionOperands), length(privateOperands), length(firstprivateOperands), length(dataClauseOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(waitOperandsSegments) && push!(attributes, NamedAttribute("waitOperandsSegments", waitOperandsSegments))
    !isnothing(waitOperandsDeviceType) && push!(attributes, NamedAttribute("waitOperandsDeviceType", waitOperandsDeviceType))
    !isnothing(hasWaitDevnum) && push!(attributes, NamedAttribute("hasWaitDevnum", hasWaitDevnum))
    !isnothing(waitOnly) && push!(attributes, NamedAttribute("waitOnly", waitOnly))
    !isnothing(selfAttr) && push!(attributes, NamedAttribute("selfAttr", selfAttr))
    !isnothing(reductionRecipes) && push!(attributes, NamedAttribute("reductionRecipes", reductionRecipes))
    !isnothing(privatizationRecipes) && push!(attributes, NamedAttribute("privatizationRecipes", privatizationRecipes))
    !isnothing(firstprivatizationRecipes) && push!(attributes, NamedAttribute("firstprivatizationRecipes", firstprivatizationRecipes))
    !isnothing(defaultAttr) && push!(attributes, NamedAttribute("defaultAttr", defaultAttr))
    !isnothing(combined) && push!(attributes, NamedAttribute("combined", combined))
    
    IR.create_operation(
        "acc.serial", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`set`

The \"acc.set\" operation represents the OpenACC set directive.

# Example

```mlir
acc.set device_num(%dev1 : i32)
```
"""
function set(defaultAsync=nothing::Union{Nothing, Value}; deviceNum=nothing::Union{Nothing, Value}, ifCond=nothing::Union{Nothing, Value}, device_type=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(defaultAsync) && push!(operands, defaultAsync)
    !isnothing(deviceNum) && push!(operands, deviceNum)
    !isnothing(ifCond) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([Int(!isnothing(defaultAsync)), Int(!isnothing(deviceNum)), Int(!isnothing(ifCond)), ]))
    !isnothing(device_type) && push!(attributes, NamedAttribute("device_type", device_type))
    
    IR.create_operation(
        "acc.set", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`shutdown`

The \"acc.shutdown\" operation represents the OpenACC shutdown executable
directive.

# Example

```mlir
acc.shutdown
acc.shutdown device_num(%dev1 : i32)
```
"""
function shutdown(deviceNum=nothing::Union{Nothing, Value}; ifCond=nothing::Union{Nothing, Value}, device_types=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(deviceNum) && push!(operands, deviceNum)
    !isnothing(ifCond) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([Int(!isnothing(deviceNum)), Int(!isnothing(ifCond)), ]))
    !isnothing(device_types) && push!(attributes, NamedAttribute("device_types", device_types))
    
    IR.create_operation(
        "acc.shutdown", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`terminator`

A terminator operation for regions that appear in the body of OpenACC
operation. Generic OpenACC construct regions are not expected to return any
value so the terminator takes no operands. The terminator op returns control
to the enclosing op.
"""
function terminator(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "acc.terminator", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`update_device`

Description of arguments:
- `var`: The variable to copy. Must be either `MappableType` or
`PointerLikeType`.
- `varType`: The type of the variable that is being copied. When `var` is
a `MappableType`, this matches the type of `var`. When `var` is a
`PointerLikeType`, this type holds information about the target of the
pointer.
- `varPtrPtr`: Specifies the address of the address of `var` - only used
when the variable copied is a field in a struct. This is important for
OpenACC due to implicit attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data entry operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function update_device(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.update_device", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`update_host`
- `varPtr`: The address of variable to copy back to.
    - `accVar`: The acc variable. This is the link from the data-entry
    operation used.
    - `bounds`: Used when copying just slice of array or array\'s bounds are not
    encoded in type. They are in rank order where rank 0 is inner-most dimension.
    - `asyncOperands` and `asyncOperandsDeviceType`:
    pair-wise lists of the async clause values associated with device_type\'s.
    - `asyncOnly`: a list of device_type\'s for which async clause
    does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
    - `dataClause`: Keeps track of the data clause the user used. This is because
    the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
    `acc.copyin` and `acc.copyout` operations, but both have dataClause that
    specifies `acc_copy` in this field.
    - `structured`: Flag to note whether this is associated with structured region
    (parallel, kernels, data) or unstructured (enter data, exit data). This is
    important due to spec specifically calling out structured and dynamic reference
    counters (2.6.7).
    - `implicit`: Whether this is an implicitly generated operation, such as copies
    done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
    - `modifiers`: Keeps track of the data clause modifiers (eg zero, always, etc)
    - `name`: Holds the name of variable as specified in user clause (including bounds).

    The async values attached to the data exit operation imply that the data
    action applies to all device types specified by the device_type clauses
    using the activity queues on these devices as defined by the async values.
"""
function update_host(accVar::Value, var::Value, bounds::Vector{Value}, asyncOperands::Vector{Value}; varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[accVar, var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    push!(attributes, operandsegmentsizes([1, 1, length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.update_host", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`update`

The `acc.update` operation represents the OpenACC update executable
directive.
As host and self clauses are synonyms, any operands for host and self are
add to \$hostOperands.

# Example

```mlir
acc.update device(%d1 : memref<10xf32>) attributes {async}
```

`async` and `wait` operands are supported with `device_type` information.
They should only be accessed by the extra provided getters. If modified,
the corresponding `device_type` attributes must be modified as well.
"""
function update(ifCond=nothing::Union{Nothing, Value}; asyncOperands::Vector{Value}, waitOperands::Vector{Value}, dataClauseOperands::Vector{Value}, asyncOperandsDeviceType=nothing, asyncOnly=nothing, waitOperandsSegments=nothing, waitOperandsDeviceType=nothing, hasWaitDevnum=nothing, waitOnly=nothing, ifPresent=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[asyncOperands..., waitOperands..., dataClauseOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(ifCond) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([Int(!isnothing(ifCond)), length(asyncOperands), length(waitOperands), length(dataClauseOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(waitOperandsSegments) && push!(attributes, NamedAttribute("waitOperandsSegments", waitOperandsSegments))
    !isnothing(waitOperandsDeviceType) && push!(attributes, NamedAttribute("waitOperandsDeviceType", waitOperandsDeviceType))
    !isnothing(hasWaitDevnum) && push!(attributes, NamedAttribute("hasWaitDevnum", hasWaitDevnum))
    !isnothing(waitOnly) && push!(attributes, NamedAttribute("waitOnly", waitOnly))
    !isnothing(ifPresent) && push!(attributes, NamedAttribute("ifPresent", ifPresent))
    
    IR.create_operation(
        "acc.update", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`use_device`

Description of arguments:
- `var`: The variable to copy. Must be either `MappableType` or
`PointerLikeType`.
- `varType`: The type of the variable that is being copied. When `var` is
a `MappableType`, this matches the type of `var`. When `var` is a
`PointerLikeType`, this type holds information about the target of the
pointer.
- `varPtrPtr`: Specifies the address of the address of `var` - only used
when the variable copied is a field in a struct. This is important for
OpenACC due to implicit attach semantics on data clauses (2.6.4).
- `bounds`: Used when copying just slice of array or array\'s bounds are not
encoded in type. They are in rank order where rank 0 is inner-most dimension.
- `asyncOperands` and `asyncOperandsDeviceType`:
pair-wise lists of the async clause values associated with device_type\'s.
- `asyncOnly`: a list of device_type\'s for which async clause
does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
- `dataClause`: Keeps track of the data clause the user used. This is because
the acc operations are decomposed. So a \'copy\' clause is decomposed to both 
`acc.copyin` and `acc.copyout` operations, but both have dataClause that
specifies `acc_copy` in this field.
- `structured`: Flag to note whether this is associated with structured region
(parallel, kernels, data) or unstructured (enter data, exit data). This is
important due to spec specifically calling out structured and dynamic reference
counters (2.6.7).
- `implicit`: Whether this is an implicitly generated operation, such as copies
done to satisfy \"Variables with Implicitly Determined Data Attributes\" in 2.6.2.
- `modifiers`: Keeps track of the data clause modifiers (eg zero, readonly, etc)
- `name`: Holds the name of variable as specified in user clause (including bounds).

The async values attached to the data entry operation imply that the data
action applies to all device types specified by the device_type clauses
using the activity queues on these devices as defined by the async values.
"""
function use_device(var::Value, varPtrPtr=nothing::Union{Nothing, Value}; bounds::Vector{Value}, asyncOperands::Vector{Value}, accVar::IR.Type, varType, asyncOperandsDeviceType=nothing, asyncOnly=nothing, dataClause=nothing, structured=nothing, implicit=nothing, modifiers=nothing, name=nothing, location=Location())
    op_ty_results = IR.Type[accVar, ]
    operands = Value[var, bounds..., asyncOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varType", varType), ]
    !isnothing(varPtrPtr) && push!(operands, varPtrPtr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(varPtrPtr)), length(bounds), length(asyncOperands), ]))
    !isnothing(asyncOperandsDeviceType) && push!(attributes, NamedAttribute("asyncOperandsDeviceType", asyncOperandsDeviceType))
    !isnothing(asyncOnly) && push!(attributes, NamedAttribute("asyncOnly", asyncOnly))
    !isnothing(dataClause) && push!(attributes, NamedAttribute("dataClause", dataClause))
    !isnothing(structured) && push!(attributes, NamedAttribute("structured", structured))
    !isnothing(implicit) && push!(attributes, NamedAttribute("implicit", implicit))
    !isnothing(modifiers) && push!(attributes, NamedAttribute("modifiers", modifiers))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "acc.use_device", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`wait`

The \"acc.wait\" operation represents the OpenACC wait executable
directive.

# Example

```mlir
acc.wait(%value1: index)
acc.wait() async(%async1: i32)
```

acc.wait does not implement MemoryEffects interface,
so it affects all the resources. This is conservatively
correct. More precise modelling of the memory effects
seems to be impossible without the whole program analysis.
"""
function wait(waitOperands::Vector{Value}, asyncOperand=nothing::Union{Nothing, Value}; waitDevnum=nothing::Union{Nothing, Value}, ifCond=nothing::Union{Nothing, Value}, async=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[waitOperands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncOperand) && push!(operands, asyncOperand)
    !isnothing(waitDevnum) && push!(operands, waitDevnum)
    !isnothing(ifCond) && push!(operands, ifCond)
    push!(attributes, operandsegmentsizes([length(waitOperands), Int(!isnothing(asyncOperand)), Int(!isnothing(waitDevnum)), Int(!isnothing(ifCond)), ]))
    !isnothing(async) && push!(attributes, NamedAttribute("async", async))
    
    IR.create_operation(
        "acc.wait", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`yield`

`acc.yield` is a special terminator operation for block inside regions in
various acc ops (including parallel, loop, atomic.update). It returns values
to the immediately enclosing acc op.
"""
function yield(operands::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "acc.yield", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end
end # acc

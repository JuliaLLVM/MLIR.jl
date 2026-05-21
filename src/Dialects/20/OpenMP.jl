module omp

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes

"""
`atomic_capture`

This operation performs an atomic capture.

The region has the following allowed forms:
```
  omp.atomic.capture {
    omp.atomic.update ...
    omp.atomic.read ...
    omp.terminator
  }

  omp.atomic.capture {
    omp.atomic.read ...
    omp.atomic.update ...
    omp.terminator
  }

  omp.atomic.capture {
    omp.atomic.read ...
    omp.atomic.write ...
    omp.terminator
  }
```
  
`hint` is the value of hint (as specified in the hint clause). It is a
compile time constant. As the name suggests, this is just a hint for
optimization.
  
`memory_order` indicates the memory ordering behavior of the construct. It
can be one of `seq_cst`, `acq_rel`, `release`, `acquire` or `relaxed`.
"""
function atomic_capture(; hint=nothing, memory_order=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(hint) && push!(attributes, NamedAttribute("hint", hint))
    !isnothing(memory_order) && push!(attributes, NamedAttribute("memory_order", memory_order))
    
    IR.create_operation(
        "omp.atomic.capture", location;
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
  
`hint` is the value of hint (as specified in the hint clause). It is a
compile time constant. As the name suggests, this is just a hint for
optimization.
  
`memory_order` indicates the memory ordering behavior of the construct. It
can be one of `seq_cst`, `acq_rel`, `release`, `acquire` or `relaxed`.
"""
function atomic_read(x::Value, v::Value; element_type, hint=nothing, memory_order=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[x, v, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("element_type", element_type), ]
    !isnothing(hint) && push!(attributes, NamedAttribute("hint", hint))
    !isnothing(memory_order) && push!(attributes, NamedAttribute("memory_order", memory_order))
    
    IR.create_operation(
        "omp.atomic.read", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`atomic_update`

This operation performs an atomic update.

The operand `x` is exactly the same as the operand `x` in the OpenMP
Standard (OpenMP 5.0, section 2.17.7). It is the address of the variable
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
  
`hint` is the value of hint (as specified in the hint clause). It is a
compile time constant. As the name suggests, this is just a hint for
optimization.
  
`memory_order` indicates the memory ordering behavior of the construct. It
can be one of `seq_cst`, `acq_rel`, `release`, `acquire` or `relaxed`.
"""
function atomic_update(x::Value; hint=nothing, memory_order=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[x, ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(hint) && push!(attributes, NamedAttribute("hint", hint))
    !isnothing(memory_order) && push!(attributes, NamedAttribute("memory_order", memory_order))
    
    IR.create_operation(
        "omp.atomic.update", location;
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
  
`hint` is the value of hint (as specified in the hint clause). It is a
compile time constant. As the name suggests, this is just a hint for
optimization.
  
`memory_order` indicates the memory ordering behavior of the construct. It
can be one of `seq_cst`, `acq_rel`, `release`, `acquire` or `relaxed`.
"""
function atomic_write(x::Value, expr::Value; hint=nothing, memory_order=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[x, expr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(hint) && push!(attributes, NamedAttribute("hint", hint))
    !isnothing(memory_order) && push!(attributes, NamedAttribute("memory_order", memory_order))
    
    IR.create_operation(
        "omp.atomic.write", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`barrier`

The barrier construct specifies an explicit barrier at the point at which
the construct appears.
"""
function barrier(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "omp.barrier", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cancel`

The cancel construct activates cancellation of the innermost enclosing
region of the type specified.
"""
function cancel(if_expr=nothing::Union{Nothing, Value}; cancel_directive, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("cancel_directive", cancel_directive), ]
    !isnothing(if_expr) && push!(operands, if_expr)
    
    IR.create_operation(
        "omp.cancel", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cancellation_point`

The cancellation point construct introduces a user-defined cancellation
point at which implicit or explicit tasks check if cancellation of the
innermost enclosing region of the type specified has been activated.
"""
function cancellation_point(; cancel_directive, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("cancel_directive", cancel_directive), ]
    
    IR.create_operation(
        "omp.cancellation_point", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`critical_declare`

Declares a named critical section.
  
The `sym_name` can be used in `omp.critical` constructs in the dialect.
  
`hint` is the value of hint (as specified in the hint clause). It is a
compile time constant. As the name suggests, this is just a hint for
optimization.
"""
function critical_declare(; sym_name, hint=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), ]
    !isnothing(hint) && push!(attributes, NamedAttribute("hint", hint))
    
    IR.create_operation(
        "omp.critical.declare", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`critical`

The critical construct imposes a restriction on the associated structured
block (region) to be executed by only a single thread at a time.

The optional `name` argument of critical constructs is used to identify
them. Unnamed critical constructs behave as though an identical name was
specified.
"""
function critical(; name=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    
    IR.create_operation(
        "omp.critical", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`declare_reduction`

Declares an OpenMP reduction kind. This requires two mandatory and three
optional regions.

  1. The optional alloc region specifies how to allocate the thread-local
     reduction value. This region should not contain control flow and all
     IR should be suitable for inlining straight into an entry block. In
     the common case this is expected to contain only allocas. It is
     expected to `omp.yield` the allocated value on all control paths.
     If allocation is conditional (e.g. only allocate if the mold is
     allocated), this should be done in the initilizer region and this
     region not included. The alloc region is not used for by-value
     reductions (where allocation is implicit).
  2. The initializer region specifies how to initialize the thread-local
     reduction value. This is usually the neutral element of the reduction.
     For convenience, the region has an argument that contains the value
     of the reduction accumulator at the start of the reduction. If an alloc
     region is specified, there is a second block argument containing the
     address of the allocated memory. The initializer region is expected to
     `omp.yield` the new value on all control flow paths.
  3. The reduction region specifies how to combine two values into one, i.e.
     the reduction operator. It accepts the two values as arguments and is
     expected to `omp.yield` the combined value on all control flow paths.
  4. The atomic reduction region is optional and specifies how two values
     can be combined atomically given local accumulator variables. It is
     expected to store the combined value in the first accumulator variable.
  5. The cleanup region is optional and specifies how to clean up any memory
     allocated by the initializer region. The region has an argument that
     contains the value of the thread-local reduction accumulator. This will
     be executed after the reduction has completed.

Note that the MLIR type system does not allow for type-polymorphic
reductions. Separate reduction declarations should be created for different
element and accumulator types.

For initializer and reduction regions, the operand to `omp.yield` must
match the parent operation\'s results.
"""
function declare_reduction(; sym_name, type, allocRegion::Region, initializerRegion::Region, reductionRegion::Region, atomicReductionRegion::Region, cleanupRegion::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[allocRegion, initializerRegion, reductionRegion, atomicReductionRegion, cleanupRegion, ]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), NamedAttribute("type", type), ]
    
    IR.create_operation(
        "omp.declare_reduction", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`distribute`

The distribute construct specifies that the iterations of one or more loops
(optionally specified using collapse clause) will be executed by the
initial teams in the context of their implicit tasks. The loops that the
distribute op is associated with starts with the outermost loop enclosed by
the distribute op region and going down the loop nest toward the innermost
loop. The iterations are distributed across the initial threads of all
initial teams that execute the teams region to which the distribute region
binds.

The distribute loop construct specifies that the iterations of the loop(s)
will be executed in parallel by threads in the current context. These
iterations are spread across threads that already exist in the enclosing
region.

The body region can only contain a single block which must contain a single
operation. This operation must be another compatible loop wrapper or an
`omp.loop_nest`.

```mlir
omp.distribute <clauses> {
  omp.loop_nest (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
    %a = load %arrA[%i1, %i2] : memref<?x?xf32>
    %b = load %arrB[%i1, %i2] : memref<?x?xf32>
    %sum = arith.addf %a, %b : f32
    store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
    omp.yield
  }
}
```
  
The `allocator_vars` and `allocate_vars` parameters are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.
  
The `dist_schedule_static` attribute specifies the schedule for this loop,
determining how the loop is distributed across the various teams. The
optional `dist_schedule_chunk_size` associated with this determines further
controls this distribution.
  
The optional `order` attribute specifies which order the iterations of the
associated loops are executed in. Currently the only option for this
attribute is \"concurrent\".
"""
function distribute(allocate_vars::Vector{Value}, allocator_vars::Vector{Value}, dist_schedule_chunk_size=nothing::Union{Nothing, Value}; private_vars::Vector{Value}, dist_schedule_static=nothing, order=nothing, order_mod=nothing, private_syms=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[allocate_vars..., allocator_vars..., private_vars..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(dist_schedule_chunk_size) && push!(operands, dist_schedule_chunk_size)
    push!(attributes, operandsegmentsizes([length(allocate_vars), length(allocator_vars), Int(!isnothing(dist_schedule_chunk_size)), length(private_vars), ]))
    !isnothing(dist_schedule_static) && push!(attributes, NamedAttribute("dist_schedule_static", dist_schedule_static))
    !isnothing(order) && push!(attributes, NamedAttribute("order", order))
    !isnothing(order_mod) && push!(attributes, NamedAttribute("order_mod", order_mod))
    !isnothing(private_syms) && push!(attributes, NamedAttribute("private_syms", private_syms))
    
    IR.create_operation(
        "omp.distribute", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`flush`

The flush construct executes the OpenMP flush operation. This operation
makes a thread\'s temporary view of memory consistent with memory and
enforces an order on the memory operations of the variables explicitly
specified or implied.
"""
function flush(varList::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[varList..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "omp.flush", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`loop_nest`

This operation represents a collapsed rectangular loop nest. For each
rectangular loop of the nest represented by an instance of this operation,
lower and upper bounds, as well as a step variable, must be defined.

The lower and upper bounds specify a half-open range: the range includes the
lower bound but does not include the upper bound. If the `loop_inclusive`
attribute is specified then the upper bound is also included.

The body region can contain any number of blocks. The region is terminated
by an `omp.yield` instruction without operands. The induction variables,
represented as entry block arguments to the loop nest operation\'s single
region, match the types of the `loop_lower_bounds`, `loop_upper_bounds` and
`loop_steps` arguments.

```mlir
omp.loop_nest (%i1, %i2) : i32 = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
  %a = load %arrA[%i1, %i2] : memref<?x?xf32>
  %b = load %arrB[%i1, %i2] : memref<?x?xf32>
  %sum = arith.addf %a, %b : f32
  store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
  omp.yield
}
```

This is a temporary simplified definition of a loop based on existing OpenMP
loop operations intended to serve as a stopgap solution until the long-term
representation of canonical loops is defined. Specifically, this operation
is intended to serve as a unique source for loop information during the
transition to making `omp.distribute`, `omp.simd`, `omp.taskloop` and
`omp.wsloop` wrapper operations. It is not intended to help with the
addition of support for loop transformations, non-rectangular loops and
non-perfectly nested loops.
"""
function loop_nest(loop_lower_bounds::Vector{Value}, loop_upper_bounds::Vector{Value}, loop_steps::Vector{Value}; loop_inclusive=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[loop_lower_bounds..., loop_upper_bounds..., loop_steps..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(loop_inclusive) && push!(attributes, NamedAttribute("loop_inclusive", loop_inclusive))
    
    IR.create_operation(
        "omp.loop_nest", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`loop`

A loop construct specifies that the logical iterations of the associated loops
may execute concurrently and permits the encountering threads to execute the
loop accordingly. A loop construct can have 3 different types of binding:
  1. teams: in which case the binding region is the innermost enclosing `teams`
     region.
  2. parallel: in which case the binding region is the innermost enclosing `parallel`
     region.
  3. thread: in which case the binding region is not defined.

The body region can only contain a single block which must contain a single
operation, this operation must be an `omp.loop_nest`.

```
omp.loop <clauses> {
  omp.loop_nest (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
    %a = load %arrA[%i1, %i2] : memref<?x?xf32>
    %b = load %arrB[%i1, %i2] : memref<?x?xf32>
    %sum = arith.addf %a, %b : f32
    store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
    omp.yield
  }
}
```
  
The `bind` clause specifies the binding region of the construct on which it
appears.
  
The optional `order` attribute specifies which order the iterations of the
associated loops are executed in. Currently the only option for this
attribute is \"concurrent\".
  
Reductions can be performed by specifying the reduction modifer
(`default`, `inscan` or `task`) in `reduction_mod`, reduction accumulator
variables in `reduction_vars`, symbols referring to reduction declarations
in the `reduction_syms` attribute, and whether the reduction variable
should be passed into the reduction region by value or by reference in
`reduction_byref`. Each reduction is identified by the accumulator it uses
and accumulators must not be repeated in the same reduction. A private
variable corresponding to the accumulator is used in place of the
accumulator inside the body of the operation. The reduction declaration
specifies how to combine the values from each iteration, section, team,
thread or simd lane defined by the operation\'s region into the final value,
which is available in the accumulator after they all complete.
"""
function loop(private_vars::Vector{Value}, reduction_vars::Vector{Value}; bind_kind=nothing, private_syms=nothing, order=nothing, order_mod=nothing, reduction_mod=nothing, reduction_byref=nothing, reduction_syms=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[private_vars..., reduction_vars..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(private_vars), length(reduction_vars), ]))
    !isnothing(bind_kind) && push!(attributes, NamedAttribute("bind_kind", bind_kind))
    !isnothing(private_syms) && push!(attributes, NamedAttribute("private_syms", private_syms))
    !isnothing(order) && push!(attributes, NamedAttribute("order", order))
    !isnothing(order_mod) && push!(attributes, NamedAttribute("order_mod", order_mod))
    !isnothing(reduction_mod) && push!(attributes, NamedAttribute("reduction_mod", reduction_mod))
    !isnothing(reduction_byref) && push!(attributes, NamedAttribute("reduction_byref", reduction_byref))
    !isnothing(reduction_syms) && push!(attributes, NamedAttribute("reduction_syms", reduction_syms))
    
    IR.create_operation(
        "omp.loop", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`map_bounds`

This operation is a variation on the OpenACC dialects DataBoundsOp. Within
the OpenMP dialect it stores the bounds/range of data to be mapped to a
device specified by map clauses on target directives. Within
the OpenMP dialect, the MapBoundsOp is associated with MapInfoOp,
helping to store bounds information for the mapped variable.

It is used to support OpenMP array sectioning, Fortran pointer and
allocatable mapping and pointer/allocatable member of derived types.
In all cases the MapBoundsOp holds information on the section of
data to be mapped. Such as the upper bound and lower bound of the
section of data to be mapped. This information is currently
utilised by the LLVM-IR lowering to help generate instructions to
copy data to and from the device when processing target operations.

The example below copys a section of a 10-element array; all except the
first element, utilising OpenMP array sectioning syntax where array
subscripts are provided to specify the bounds to be mapped to device.
To simplify the examples, the constants are used directly, in reality
they will be MLIR SSA values.

C++:
```
int array[10];
#pragma target map(array[1:9])
```
=>
```mlir
omp.map.bounds lower_bound(1) upper_bound(9) extent(9) start_idx(0)
```

Fortran:
```
integer :: array(1:10)
!\$target map(array(2:10))
```
=>
```mlir
omp.map.bounds lower_bound(1) upper_bound(9) extent(9) start_idx(1)
```

For Fortran pointers and allocatables (as well as those that are
members of derived types) the bounds information is provided by
the Fortran compiler and runtime through descriptor information.

A basic pointer example can be found below (constants again
provided for simplicity, where in reality SSA values will be
used, in this case that point to data yielded by Fortran\'s
descriptors):

Fortran:
```
integer, pointer :: ptr(:)
allocate(ptr(10))
!\$target map(ptr)
```
=>
```mlir
omp.map.bounds lower_bound(0) upper_bound(9) extent(10) start_idx(1)
```

This operation records the bounds information in a normalized fashion
(zero-based). This works well with the `PointerLikeType`
requirement in data clauses - since a `lower_bound` of 0 means looking
at data at the zero offset from pointer.

This operation must have an `upper_bound` or `extent` (or both are allowed -
but not checked for consistency). When the source language\'s arrays are
not zero-based, the `start_idx` must specify the zero-position index.
"""
function map_bounds(lower_bound=nothing::Union{Nothing, Value}; upper_bound=nothing::Union{Nothing, Value}, extent=nothing::Union{Nothing, Value}, stride=nothing::Union{Nothing, Value}, start_idx=nothing::Union{Nothing, Value}, result::IR.Type, stride_in_bytes=nothing, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(lower_bound) && push!(operands, lower_bound)
    !isnothing(upper_bound) && push!(operands, upper_bound)
    !isnothing(extent) && push!(operands, extent)
    !isnothing(stride) && push!(operands, stride)
    !isnothing(start_idx) && push!(operands, start_idx)
    push!(attributes, operandsegmentsizes([Int(!isnothing(lower_bound)), Int(!isnothing(upper_bound)), Int(!isnothing(extent)), Int(!isnothing(stride)), Int(!isnothing(start_idx)), ]))
    !isnothing(stride_in_bytes) && push!(attributes, NamedAttribute("stride_in_bytes", stride_in_bytes))
    
    IR.create_operation(
        "omp.map.bounds", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`map_info`

The MapInfoOp captures information relating to individual OpenMP map clauses
that are applied to certain OpenMP directives such as Target and Target Data.

For example, the map type modifier; such as from, tofrom and to, the variable
being captured or the bounds of an array section being mapped.

It can be used to capture both implicit and explicit map information, where
explicit is an argument directly specified to an OpenMP map clause or implicit
where a variable is utilised in a target region but is defined externally to
the target region.

This map information is later used to aid the lowering of the target operations
they are attached to providing argument input and output context for kernels
generated or the target data mapping environment.

Example (Fortran):

```
integer :: index
!\$target map(to: index)
```
=>
```mlir
omp.map.info var_ptr(%index_ssa) map_type(to) map_capture_type(ByRef)
  name(index)
```

Description of arguments:
- `var_ptr`: The address of variable to copy.
- `var_type`: The type of the variable to copy.
- `var_ptr_ptr`: Used when the variable copied is a member of a class, structure
  or derived type and refers to the originating struct.
- `members`: Used to indicate mapped child members for the current MapInfoOp,
   represented as other MapInfoOp\'s, utilised in cases where a parent structure
   type and members of the structure type are being mapped at the same time.
   For example: map(to: parent, parent->member, parent->member2[:10])
- `members_index`: Used to indicate the ordering of members within the containing
   parent (generally a record type such as a structure, class or derived type),
   e.g. struct {int x, float y, double z}, x would be 0, y would be 1, and z
   would be 2. This aids the mapping.
- `bounds`: Used when copying slices of array\'s, pointers or pointer members of
   objects (e.g. derived types or classes), indicates the bounds to be copied
   of the variable. When it\'s an array slice it is in rank order where rank 0
   is the inner-most dimension.
- \'map_type\': OpenMP map type for this map capture, for example: from, to and
   always. It\'s a bitfield composed of the OpenMP runtime flags stored in
   OpenMPOffloadMappingFlags.
- \'map_capture_type\': Capture type for the variable e.g. this, byref, byvalue, byvla
   this can affect how the variable is lowered.
- `name`: Holds the name of variable as specified in user clause (including bounds).
- `partial_map`: The record type being mapped will not be mapped in its entirety,
   it may be used however, in a mapping to bind it\'s mapped components together.
"""
function map_info(var_ptr::Value, var_ptr_ptr=nothing::Union{Nothing, Value}; members::Vector{Value}, bounds::Vector{Value}, omp_ptr::IR.Type, var_type, members_index=nothing, map_type=nothing, map_capture_type=nothing, name=nothing, partial_map=nothing, location=Location())
    op_ty_results = IR.Type[omp_ptr, ]
    operands = Value[var_ptr, members..., bounds..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("var_type", var_type), ]
    !isnothing(var_ptr_ptr) && push!(operands, var_ptr_ptr)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(var_ptr_ptr)), length(members), length(bounds), ]))
    !isnothing(members_index) && push!(attributes, NamedAttribute("members_index", members_index))
    !isnothing(map_type) && push!(attributes, NamedAttribute("map_type", map_type))
    !isnothing(map_capture_type) && push!(attributes, NamedAttribute("map_capture_type", map_capture_type))
    !isnothing(name) && push!(attributes, NamedAttribute("name", name))
    !isnothing(partial_map) && push!(attributes, NamedAttribute("partial_map", partial_map))
    
    IR.create_operation(
        "omp.map.info", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`masked`

Masked construct allows to specify a structured block to be executed by a subset of 
threads of the current team.
  
If `filter` is specified, the masked construct masks the execution of
the region to only the thread id filtered. Other threads executing the
parallel region are not expected to execute the region specified within
the `masked` directive. If `filter` is not specified, master thread is
expected to execute the region enclosed within `masked` directive.
"""
function masked(filtered_thread_id=nothing::Union{Nothing, Value}; region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(filtered_thread_id) && push!(operands, filtered_thread_id)
    
    IR.create_operation(
        "omp.masked", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`master`

The master construct specifies a structured block that is executed by
the master thread of the team.
"""
function master(; region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "omp.master", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`ordered`

The ordered construct without region is a stand-alone directive that
specifies cross-iteration dependencies in a doacross loop nest.
  
The `doacross_depend_type` attribute refers to either the DEPEND(SOURCE)
clause or the DEPEND(SINK: vec) clause.

The `doacross_num_loops` attribute specifies the number of loops in the
doacross nest.

The `doacross_depend_vars` is a variadic list of operands that specifies the
index of the loop iterator in the doacross nest for the DEPEND(SOURCE)
clause or the index of the element of \"vec\" for the DEPEND(SINK: vec)
clause. It contains the operands in multiple \"vec\" when multiple
DEPEND(SINK: vec) clauses exist in one ORDERED directive.
"""
function ordered(doacross_depend_vars::Vector{Value}; doacross_depend_type=nothing, doacross_num_loops=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[doacross_depend_vars..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(doacross_depend_type) && push!(attributes, NamedAttribute("doacross_depend_type", doacross_depend_type))
    !isnothing(doacross_num_loops) && push!(attributes, NamedAttribute("doacross_num_loops", doacross_num_loops))
    
    IR.create_operation(
        "omp.ordered", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`ordered_region`

The ordered construct with region specifies a structured block in a
worksharing-loop, SIMD, or worksharing-loop SIMD region that is executed in
the order of the loop iterations.
  
The `par_level_simd` attribute corresponds to the simd clause specified. If
it is not present, it behaves as if the threads clause is specified or no
clause is specified.
"""
function ordered_region(; par_level_simd=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(par_level_simd) && push!(attributes, NamedAttribute("par_level_simd", par_level_simd))
    
    IR.create_operation(
        "omp.ordered.region", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`parallel`

The parallel construct includes a region of code which is to be executed
by a team of threads.

The optional `if_expr` parameter specifies a boolean result of a conditional
check. If this value is 1 or is not provided then the parallel region runs
as normal, if it is 0 then the parallel region is executed with one thread.
  
The `allocator_vars` and `allocate_vars` parameters are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.
  
The optional `num_threads` parameter specifies the number of threads which
should be used to execute the parallel region.
  
The optional `proc_bind_kind` attribute controls the thread affinity for the
execution of the parallel region.
  
Reductions can be performed by specifying the reduction modifer
(`default`, `inscan` or `task`) in `reduction_mod`, reduction accumulator
variables in `reduction_vars`, symbols referring to reduction declarations
in the `reduction_syms` attribute, and whether the reduction variable
should be passed into the reduction region by value or by reference in
`reduction_byref`. Each reduction is identified by the accumulator it uses
and accumulators must not be repeated in the same reduction. A private
variable corresponding to the accumulator is used in place of the
accumulator inside the body of the operation. The reduction declaration
specifies how to combine the values from each iteration, section, team,
thread or simd lane defined by the operation\'s region into the final value,
which is available in the accumulator after they all complete.
"""
function parallel(allocate_vars::Vector{Value}, allocator_vars::Vector{Value}, if_expr=nothing::Union{Nothing, Value}; num_threads=nothing::Union{Nothing, Value}, private_vars::Vector{Value}, reduction_vars::Vector{Value}, private_syms=nothing, proc_bind_kind=nothing, reduction_mod=nothing, reduction_byref=nothing, reduction_syms=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[allocate_vars..., allocator_vars..., private_vars..., reduction_vars..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(operands, if_expr)
    !isnothing(num_threads) && push!(operands, num_threads)
    push!(attributes, operandsegmentsizes([length(allocate_vars), length(allocator_vars), Int(!isnothing(if_expr)), Int(!isnothing(num_threads)), length(private_vars), length(reduction_vars), ]))
    !isnothing(private_syms) && push!(attributes, NamedAttribute("private_syms", private_syms))
    !isnothing(proc_bind_kind) && push!(attributes, NamedAttribute("proc_bind_kind", proc_bind_kind))
    !isnothing(reduction_mod) && push!(attributes, NamedAttribute("reduction_mod", reduction_mod))
    !isnothing(reduction_byref) && push!(attributes, NamedAttribute("reduction_byref", reduction_byref))
    !isnothing(reduction_syms) && push!(attributes, NamedAttribute("reduction_syms", reduction_syms))
    
    IR.create_operation(
        "omp.parallel", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`private`

This operation provides a declaration of how to implement the
[first]privatization of a variable. The dialect users should provide
information about how to create an instance of the type in the alloc region,
how to initialize the copy from the original item in the copy region, and if
needed, how to deallocate allocated memory in the dealloc region.

Examples:

* `private(x)` would be emitted as:
```mlir
omp.private {type = private} @x.privatizer : !fir.ref<i32> alloc {
^bb0(%arg0: !fir.ref<i32>):
%0 = ... allocate proper memory for the private clone ...
omp.yield(%0 : !fir.ref<i32>)
}
```

* `firstprivate(x)` would be emitted as:
```mlir
omp.private {type = firstprivate} @x.privatizer : !fir.ref<i32> alloc {
^bb0(%arg0: !fir.ref<i32>):
%0 = ... allocate proper memory for the private clone ...
omp.yield(%0 : !fir.ref<i32>)
} copy {
^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
// %arg0 is the original host variable. Same as for `alloc`.
// %arg1 represents the memory allocated in `alloc`.
... copy from host to the privatized clone ....
omp.yield(%arg1 : !fir.ref<i32>)
}
```

* `private(x)` for \"allocatables\" would be emitted as:
```mlir
omp.private {type = private} @x.privatizer : !some.type alloc {
^bb0(%arg0: !some.type):
%0 = ... allocate proper memory for the private clone ...
omp.yield(%0 : !fir.ref<i32>)
} dealloc {
^bb0(%arg0: !some.type):
... deallocate allocated memory ...
omp.yield
}
```

There are no restrictions on the body except for:
- The `alloc` & `dealloc` regions have a single argument.
- The `copy` region has 2 arguments.
- All three regions are terminated by `omp.yield` ops.
The above restrictions and other obvious restrictions (e.g. verifying the
type of yielded values) are verified by the custom op verifier. The actual
contents of the blocks inside all regions are not verified.

Instances of this op would then be used by ops that model directives that
accept data-sharing attribute clauses.

The \$sym_name attribute provides a symbol by which the privatizer op can be
referenced by other dialect ops.

The \$type attribute is the type of the value being privatized.

The \$data_sharing_type attribute specifies whether privatizer corresponds
to a `private` or a `firstprivate` clause.
"""
function private(; sym_name, type, data_sharing_type, alloc_region::Region, copy_region::Region, dealloc_region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[alloc_region, copy_region, dealloc_region, ]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), NamedAttribute("type", type), NamedAttribute("data_sharing_type", data_sharing_type), ]
    
    IR.create_operation(
        "omp.private", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`scan`

The scan directive allows to specify scan reductions. It should be
enclosed within a parent directive along with which a reduction clause
with `inscan` modifier must be specified. The scan directive allows to
split code blocks into input phase and scan phase in the region
enclosed by the parent.
  
The inclusive clause is used on a separating directive that separates a
structured block into two structured block sequences. If it is specified,
the input phase includes the preceding structured block sequence and the
scan phase includes the following structured block sequence.

The `inclusive_vars` is a variadic list of operands that specifies the
scan-reduction accumulator symbols.
  
The exclusive clause is used on a separating directive that separates a
structured block into two structured block sequences. If it
is specified, the input phase excludes the preceding structured block 
sequence and instead includes the following structured block sequence, 
while the scan phase includes the preceding structured block sequence.

The `exclusive_vars` is a variadic list of operands that specifies the
scan-reduction accumulator symbols.
"""
function scan(inclusive_vars::Vector{Value}, exclusive_vars::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[inclusive_vars..., exclusive_vars..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(inclusive_vars), length(exclusive_vars), ]))
    
    IR.create_operation(
        "omp.scan", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`section`

A section operation encloses a region which represents one section in a
sections construct. A section op should always be surrounded by an
`omp.sections` operation. The section operation may have block args
which corespond to the block arguments of the surrounding `omp.sections`
operation. This is done to reflect situations where these block arguments
represent variables private to each section.
"""
function section(; region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "omp.section", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`sections`

The sections construct is a non-iterative worksharing construct that
contains `omp.section` operations. The `omp.section` operations are to be
distributed among and executed by the threads in a team. Each `omp.section`
is executed once by one of the threads in the team in the context of its
implicit task.
Block arguments for reduction variables should be mirrored in enclosed
`omp.section` operations.
  
The `allocator_vars` and `allocate_vars` parameters are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.
  
The optional `nowait` attribute, when present, eliminates the implicit
barrier at the end of the construct, so the parent operation can make
progress even if the child operation has not completed yet.
  
Reductions can be performed by specifying the reduction modifer
(`default`, `inscan` or `task`) in `reduction_mod`, reduction accumulator
variables in `reduction_vars`, symbols referring to reduction declarations
in the `reduction_syms` attribute, and whether the reduction variable
should be passed into the reduction region by value or by reference in
`reduction_byref`. Each reduction is identified by the accumulator it uses
and accumulators must not be repeated in the same reduction. A private
variable corresponding to the accumulator is used in place of the
accumulator inside the body of the operation. The reduction declaration
specifies how to combine the values from each iteration, section, team,
thread or simd lane defined by the operation\'s region into the final value,
which is available in the accumulator after they all complete.
"""
function sections(allocate_vars::Vector{Value}, allocator_vars::Vector{Value}, private_vars::Vector{Value}, reduction_vars::Vector{Value}; nowait=nothing, private_syms=nothing, reduction_mod=nothing, reduction_byref=nothing, reduction_syms=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[allocate_vars..., allocator_vars..., private_vars..., reduction_vars..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(allocate_vars), length(allocator_vars), length(private_vars), length(reduction_vars), ]))
    !isnothing(nowait) && push!(attributes, NamedAttribute("nowait", nowait))
    !isnothing(private_syms) && push!(attributes, NamedAttribute("private_syms", private_syms))
    !isnothing(reduction_mod) && push!(attributes, NamedAttribute("reduction_mod", reduction_mod))
    !isnothing(reduction_byref) && push!(attributes, NamedAttribute("reduction_byref", reduction_byref))
    !isnothing(reduction_syms) && push!(attributes, NamedAttribute("reduction_syms", reduction_syms))
    
    IR.create_operation(
        "omp.sections", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`simd`

The simd construct can be applied to a loop to indicate that the loop can be
transformed into a SIMD loop (that is, multiple iterations of the loop can
be executed concurrently using SIMD instructions).

The body region can only contain a single block which must contain a single
operation. This operation must be another compatible loop wrapper or an
`omp.loop_nest`.

```
omp.simd <clauses> {
  omp.loop_nest (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
    %a = load %arrA[%i1, %i2] : memref<?x?xf32>
    %b = load %arrB[%i1, %i2] : memref<?x?xf32>
    %sum = arith.addf %a, %b : f32
    store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
    omp.yield
  }
}
```

When an if clause is present and evaluates to false, the preferred number of
iterations to be executed concurrently is one, regardless of whether
a simdlen clause is specified.
  
The `alignments` attribute additionally specifies alignment of each
corresponding aligned operand. Note that `aligned_vars` and `alignments`
must contain the same number of elements.
  
The `linear_step_vars` operand additionally specifies the step for each
associated linear operand. Note that the `linear_vars` and
`linear_step_vars` variadic lists should contain the same number of
elements.
  
The optional `nontemporal` attribute specifies variables which have low
temporal locality across the iterations where they are accessed.
  
The optional `order` attribute specifies which order the iterations of the
associated loops are executed in. Currently the only option for this
attribute is \"concurrent\".
  
Reductions can be performed by specifying the reduction modifer
(`default`, `inscan` or `task`) in `reduction_mod`, reduction accumulator
variables in `reduction_vars`, symbols referring to reduction declarations
in the `reduction_syms` attribute, and whether the reduction variable
should be passed into the reduction region by value or by reference in
`reduction_byref`. Each reduction is identified by the accumulator it uses
and accumulators must not be repeated in the same reduction. A private
variable corresponding to the accumulator is used in place of the
accumulator inside the body of the operation. The reduction declaration
specifies how to combine the values from each iteration, section, team,
thread or simd lane defined by the operation\'s region into the final value,
which is available in the accumulator after they all complete.
  
The `safelen` clause specifies that no two concurrent iterations within a
SIMD chunk can have a distance in the logical iteration space that is
greater than or equal to the value given in the clause.
  
When a `simdlen` clause is present, the preferred number of iterations to be
executed concurrently is the value provided to the `simdlen` clause.
"""
function simd(aligned_vars::Vector{Value}, if_expr=nothing::Union{Nothing, Value}; linear_vars::Vector{Value}, linear_step_vars::Vector{Value}, nontemporal_vars::Vector{Value}, private_vars::Vector{Value}, reduction_vars::Vector{Value}, alignments=nothing, order=nothing, order_mod=nothing, private_syms=nothing, reduction_mod=nothing, reduction_byref=nothing, reduction_syms=nothing, safelen=nothing, simdlen=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[aligned_vars..., linear_vars..., linear_step_vars..., nontemporal_vars..., private_vars..., reduction_vars..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(operands, if_expr)
    push!(attributes, operandsegmentsizes([length(aligned_vars), Int(!isnothing(if_expr)), length(linear_vars), length(linear_step_vars), length(nontemporal_vars), length(private_vars), length(reduction_vars), ]))
    !isnothing(alignments) && push!(attributes, NamedAttribute("alignments", alignments))
    !isnothing(order) && push!(attributes, NamedAttribute("order", order))
    !isnothing(order_mod) && push!(attributes, NamedAttribute("order_mod", order_mod))
    !isnothing(private_syms) && push!(attributes, NamedAttribute("private_syms", private_syms))
    !isnothing(reduction_mod) && push!(attributes, NamedAttribute("reduction_mod", reduction_mod))
    !isnothing(reduction_byref) && push!(attributes, NamedAttribute("reduction_byref", reduction_byref))
    !isnothing(reduction_syms) && push!(attributes, NamedAttribute("reduction_syms", reduction_syms))
    !isnothing(safelen) && push!(attributes, NamedAttribute("safelen", safelen))
    !isnothing(simdlen) && push!(attributes, NamedAttribute("simdlen", simdlen))
    
    IR.create_operation(
        "omp.simd", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`single`

The single construct specifies that the associated structured block is
executed by only one of the threads in the team (not necessarily the
master thread), in the context of its implicit task. The other threads
in the team, which do not execute the block, wait at an implicit barrier
at the end of the single construct.
  
The `allocator_vars` and `allocate_vars` parameters are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.
  
If `copyprivate` variables and functions are specified, then each thread
variable is updated with the variable value of the thread that executed
the single region, using the specified copy functions.
  
The optional `nowait` attribute, when present, eliminates the implicit
barrier at the end of the construct, so the parent operation can make
progress even if the child operation has not completed yet.
"""
function single(allocate_vars::Vector{Value}, allocator_vars::Vector{Value}, copyprivate_vars::Vector{Value}, private_vars::Vector{Value}; copyprivate_syms=nothing, nowait=nothing, private_syms=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[allocate_vars..., allocator_vars..., copyprivate_vars..., private_vars..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(allocate_vars), length(allocator_vars), length(copyprivate_vars), length(private_vars), ]))
    !isnothing(copyprivate_syms) && push!(attributes, NamedAttribute("copyprivate_syms", copyprivate_syms))
    !isnothing(nowait) && push!(attributes, NamedAttribute("nowait", nowait))
    !isnothing(private_syms) && push!(attributes, NamedAttribute("private_syms", private_syms))
    
    IR.create_operation(
        "omp.single", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`target_data`

Map variables to a device data environment for the extent of the region.

The omp target data directive maps variables to a device data
environment, and defines the lexical scope of the data environment
that is created. The omp target data directive can reduce data copies
to and from the offloading device when multiple target regions are using
the same data.

The optional `if_expr` parameter specifies a boolean result of a conditional
check. If this value is 1 or is not provided then the target region runs on
a device, if it is 0 then the target region is executed on the host device.
  
The optional `device` parameter specifies the device number for the target
region.
  
The optional `map_vars` maps data from the current task\'s data environment
to the device data environment.
  
The optional `use_device_addr_vars` specifies the address of the objects in
the device data environment.
  
The optional `use_device_ptr_vars` specifies the device pointers to the
corresponding list items in the device data environment.
"""
function target_data(device=nothing::Union{Nothing, Value}; if_expr=nothing::Union{Nothing, Value}, map_vars::Vector{Value}, use_device_addr_vars::Vector{Value}, use_device_ptr_vars::Vector{Value}, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[map_vars..., use_device_addr_vars..., use_device_ptr_vars..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(device) && push!(operands, device)
    !isnothing(if_expr) && push!(operands, if_expr)
    push!(attributes, operandsegmentsizes([Int(!isnothing(device)), Int(!isnothing(if_expr)), length(map_vars), length(use_device_addr_vars), length(use_device_ptr_vars), ]))
    
    IR.create_operation(
        "omp.target_data", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`target_enter_data`

The target enter data directive specifies that variables are mapped to
a device data environment. The target enter data directive is a
stand-alone directive.

The optional `if_expr` parameter specifies a boolean result of a conditional
check. If this value is 1 or is not provided then the target region runs on
a device, if it is 0 then the target region is executed on the host device.
  
The `depend_kinds` and `depend_vars` arguments are variadic lists of values
that specify the dependencies of this particular task in relation to other
tasks.
  
The optional `device` parameter specifies the device number for the target
region.
  
The optional `map_vars` maps data from the current task\'s data environment
to the device data environment.
  
The optional `nowait` attribute, when present, eliminates the implicit
barrier at the end of the construct, so the parent operation can make
progress even if the child operation has not completed yet.
"""
function target_enter_data(depend_vars::Vector{Value}, device=nothing::Union{Nothing, Value}; if_expr=nothing::Union{Nothing, Value}, map_vars::Vector{Value}, depend_kinds=nothing, nowait=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[depend_vars..., map_vars..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(device) && push!(operands, device)
    !isnothing(if_expr) && push!(operands, if_expr)
    push!(attributes, operandsegmentsizes([length(depend_vars), Int(!isnothing(device)), Int(!isnothing(if_expr)), length(map_vars), ]))
    !isnothing(depend_kinds) && push!(attributes, NamedAttribute("depend_kinds", depend_kinds))
    !isnothing(nowait) && push!(attributes, NamedAttribute("nowait", nowait))
    
    IR.create_operation(
        "omp.target_enter_data", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`target_exit_data`

The target exit data directive specifies that variables are mapped to a
device data environment. The target exit data directive is
a stand-alone directive.

The optional `if_expr` parameter specifies a boolean result of a conditional
check. If this value is 1 or is not provided then the target region runs on
a device, if it is 0 then the target region is executed on the host device.
  
The `depend_kinds` and `depend_vars` arguments are variadic lists of values
that specify the dependencies of this particular task in relation to other
tasks.
  
The optional `device` parameter specifies the device number for the target
region.
  
The optional `map_vars` maps data from the current task\'s data environment
to the device data environment.
  
The optional `nowait` attribute, when present, eliminates the implicit
barrier at the end of the construct, so the parent operation can make
progress even if the child operation has not completed yet.
"""
function target_exit_data(depend_vars::Vector{Value}, device=nothing::Union{Nothing, Value}; if_expr=nothing::Union{Nothing, Value}, map_vars::Vector{Value}, depend_kinds=nothing, nowait=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[depend_vars..., map_vars..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(device) && push!(operands, device)
    !isnothing(if_expr) && push!(operands, if_expr)
    push!(attributes, operandsegmentsizes([length(depend_vars), Int(!isnothing(device)), Int(!isnothing(if_expr)), length(map_vars), ]))
    !isnothing(depend_kinds) && push!(attributes, NamedAttribute("depend_kinds", depend_kinds))
    !isnothing(nowait) && push!(attributes, NamedAttribute("nowait", nowait))
    
    IR.create_operation(
        "omp.target_exit_data", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`target`

The target construct includes a region of code which is to be executed
on a device.

The optional `if_expr` parameter specifies a boolean result of a conditional
check. If this value is 1 or is not provided then the target region runs on
a device, if it is 0 then the target region is executed on the host device.

The `private_maps` attribute connects `private` operands to their corresponding
`map` operands. For `private` operands that require a map, the value of the
corresponding element in the attribute is the index of the `map` operand
(relative to other `map` operands not the whole operands of the operation). For
`private` opernads that do not require a map, this value is -1 (which is omitted
from the assembly foramt printing).
  
The `allocator_vars` and `allocate_vars` parameters are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.
  
`ompx_bare` allows `omp target teams` to be executed on a GPU with an
explicit number of teams and threads. This clause also allows the teams and
threads sizes to have up to 3 dimensions.
  
The `depend_kinds` and `depend_vars` arguments are variadic lists of values
that specify the dependencies of this particular task in relation to other
tasks.
  
The optional `device` parameter specifies the device number for the target
region.
  
The optional `has_device_addr_vars` indicates that list items already have
device addresses, so they may be directly accessed from the target device.
This includes array sections.
  
The optional `host_eval_vars` holds values defined outside of the region of
the `IsolatedFromAbove` operation for which a corresponding entry block
argument is defined. The only legal uses for these captured values are the
following:
  - `num_teams` or `thread_limit` clause of an immediately nested
  `omp.teams` operation.
  - If the operation is the top-level `omp.target` of a target SPMD kernel:
    - `num_threads` clause of the nested `omp.parallel` operation.
    - Bounds and steps of the nested `omp.loop_nest` operation.
  
The optional `is_device_ptr_vars` indicates list items are device pointers.
  
The optional `map_vars` maps data from the current task\'s data environment
to the device data environment.
  
The optional `nowait` attribute, when present, eliminates the implicit
barrier at the end of the construct, so the parent operation can make
progress even if the child operation has not completed yet.
  
The optional `thread_limit` specifies the limit on the number of threads.
"""
function target(allocate_vars::Vector{Value}, allocator_vars::Vector{Value}, depend_vars::Vector{Value}, device=nothing::Union{Nothing, Value}; has_device_addr_vars::Vector{Value}, host_eval_vars::Vector{Value}, if_expr=nothing::Union{Nothing, Value}, in_reduction_vars::Vector{Value}, is_device_ptr_vars::Vector{Value}, map_vars::Vector{Value}, private_vars::Vector{Value}, thread_limit=nothing::Union{Nothing, Value}, bare=nothing, depend_kinds=nothing, in_reduction_byref=nothing, in_reduction_syms=nothing, nowait=nothing, private_syms=nothing, private_maps=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[allocate_vars..., allocator_vars..., depend_vars..., has_device_addr_vars..., host_eval_vars..., in_reduction_vars..., is_device_ptr_vars..., map_vars..., private_vars..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(device) && push!(operands, device)
    !isnothing(if_expr) && push!(operands, if_expr)
    !isnothing(thread_limit) && push!(operands, thread_limit)
    push!(attributes, operandsegmentsizes([length(allocate_vars), length(allocator_vars), length(depend_vars), Int(!isnothing(device)), length(has_device_addr_vars), length(host_eval_vars), Int(!isnothing(if_expr)), length(in_reduction_vars), length(is_device_ptr_vars), length(map_vars), length(private_vars), Int(!isnothing(thread_limit)), ]))
    !isnothing(bare) && push!(attributes, NamedAttribute("bare", bare))
    !isnothing(depend_kinds) && push!(attributes, NamedAttribute("depend_kinds", depend_kinds))
    !isnothing(in_reduction_byref) && push!(attributes, NamedAttribute("in_reduction_byref", in_reduction_byref))
    !isnothing(in_reduction_syms) && push!(attributes, NamedAttribute("in_reduction_syms", in_reduction_syms))
    !isnothing(nowait) && push!(attributes, NamedAttribute("nowait", nowait))
    !isnothing(private_syms) && push!(attributes, NamedAttribute("private_syms", private_syms))
    !isnothing(private_maps) && push!(attributes, NamedAttribute("private_maps", private_maps))
    
    IR.create_operation(
        "omp.target", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`target_update`

The target update directive makes the corresponding list items in the device
data environment consistent with their original list items, according to the
specified motion clauses. The target update construct is a stand-alone
directive.

The optional `if_expr` parameter specifies a boolean result of a conditional
check. If this value is 1 or is not provided then the target region runs on
a device, if it is 0 then the target region is executed on the host device.

We use `MapInfoOp` to model the motion clauses and their modifiers. Even
though the spec differentiates between map-types & map-type-modifiers vs.
motion-clauses & motion-modifiers, the motion clauses and their modifiers
are a subset of map types and their modifiers. The subset relation is
handled in during verification to make sure the restrictions for target
update are respected.
  
The `depend_kinds` and `depend_vars` arguments are variadic lists of values
that specify the dependencies of this particular task in relation to other
tasks.
  
The optional `device` parameter specifies the device number for the target
region.
  
The optional `map_vars` maps data from the current task\'s data environment
to the device data environment.
  
The optional `nowait` attribute, when present, eliminates the implicit
barrier at the end of the construct, so the parent operation can make
progress even if the child operation has not completed yet.
"""
function target_update(depend_vars::Vector{Value}, device=nothing::Union{Nothing, Value}; if_expr=nothing::Union{Nothing, Value}, map_vars::Vector{Value}, depend_kinds=nothing, nowait=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[depend_vars..., map_vars..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(device) && push!(operands, device)
    !isnothing(if_expr) && push!(operands, if_expr)
    push!(attributes, operandsegmentsizes([length(depend_vars), Int(!isnothing(device)), Int(!isnothing(if_expr)), length(map_vars), ]))
    !isnothing(depend_kinds) && push!(attributes, NamedAttribute("depend_kinds", depend_kinds))
    !isnothing(nowait) && push!(attributes, NamedAttribute("nowait", nowait))
    
    IR.create_operation(
        "omp.target_update", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`task`

The task construct defines an explicit task.

For definitions of \"undeferred task\", \"included task\", \"final task\" and
\"mergeable task\", please check OpenMP Specification.

When an `if` clause is present on a task construct, and the value of
`if_expr` evaluates to `false`, an \"undeferred task\" is generated, and the
encountering thread must suspend the current task region, for which
execution cannot be resumed until execution of the structured block that is
associated with the generated task is completed.

The `in_reduction` clause specifies that this particular task (among all the
tasks in current taskgroup, if any) participates in a reduction.
`in_reduction_byref` indicates whether each reduction variable should
be passed by value or by reference.
  
The `allocator_vars` and `allocate_vars` parameters are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.
  
The `depend_kinds` and `depend_vars` arguments are variadic lists of values
that specify the dependencies of this particular task in relation to other
tasks.
  
When a `final` clause is present and the `final` clause expression evaluates
to `true`, the generated tasks will be final tasks. All task constructs
encountered during execution of a final task will generate final and
included tasks. The use of a variable in a `final` clause expression causes
an implicit reference to the variable in all enclosing constructs.
  
When the `mergeable` clause is present, the tasks generated by the construct
are \"mergeable tasks\".
  
The `priority` clause is a hint for the priority of the generated tasks.
The `priority` is a non-negative integer expression that provides a hint for
task execution order. Among all tasks ready to be executed, higher priority
tasks (those with a higher numerical value in the priority clause
expression) are recommended to execute before lower priority ones. The
default priority-value when no priority clause is specified should be
assumed to be zero (the lowest priority).
  
If the `untied` clause is present on a task construct, any thread in the
team can resume the task region after a suspension. The `untied` clause is
ignored if a `final` clause is present on the same task construct and the
`final` expression evaluates to `true`, or if a task is an included task.
  
		The detach clause specifies that the task generated by the construct on which it appears is a
	detachable task. A new allow-completion event is created and connected to the completion of the
	associated task region. The original event-handle is updated to represent that allow-completion
	event before the task data environment is created.
"""
function task(allocate_vars::Vector{Value}, allocator_vars::Vector{Value}, depend_vars::Vector{Value}, final=nothing::Union{Nothing, Value}; if_expr=nothing::Union{Nothing, Value}, in_reduction_vars::Vector{Value}, priority=nothing::Union{Nothing, Value}, private_vars::Vector{Value}, event_handle=nothing::Union{Nothing, Value}, depend_kinds=nothing, in_reduction_byref=nothing, in_reduction_syms=nothing, mergeable=nothing, private_syms=nothing, untied=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[allocate_vars..., allocator_vars..., depend_vars..., in_reduction_vars..., private_vars..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(final) && push!(operands, final)
    !isnothing(if_expr) && push!(operands, if_expr)
    !isnothing(priority) && push!(operands, priority)
    !isnothing(event_handle) && push!(operands, event_handle)
    push!(attributes, operandsegmentsizes([length(allocate_vars), length(allocator_vars), length(depend_vars), Int(!isnothing(final)), Int(!isnothing(if_expr)), length(in_reduction_vars), Int(!isnothing(priority)), length(private_vars), Int(!isnothing(event_handle)), ]))
    !isnothing(depend_kinds) && push!(attributes, NamedAttribute("depend_kinds", depend_kinds))
    !isnothing(in_reduction_byref) && push!(attributes, NamedAttribute("in_reduction_byref", in_reduction_byref))
    !isnothing(in_reduction_syms) && push!(attributes, NamedAttribute("in_reduction_syms", in_reduction_syms))
    !isnothing(mergeable) && push!(attributes, NamedAttribute("mergeable", mergeable))
    !isnothing(private_syms) && push!(attributes, NamedAttribute("private_syms", private_syms))
    !isnothing(untied) && push!(attributes, NamedAttribute("untied", untied))
    
    IR.create_operation(
        "omp.task", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`taskgroup`

The taskgroup construct specifies a wait on completion of child tasks of the
current task and their descendent tasks.

When a thread encounters a taskgroup construct, it starts executing the
region. All child tasks generated in the taskgroup region and all of their
descendants that bind to the same parallel region as the taskgroup region
are part of the taskgroup set associated with the taskgroup region. There is
an implicit task scheduling point at the end of the taskgroup region. The
current task is suspended at the task scheduling point until all tasks in
the taskgroup set complete execution.
  
The `allocator_vars` and `allocate_vars` parameters are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.
  
The `task_reduction` clause specifies a reduction among tasks. For each list
item, the number of copies is unspecified. Any copies associated with the
reduction are initialized before they are accessed by the tasks
participating in the reduction. After the end of the region, the original
list item contains the result of the reduction. Similarly to the `reduction`
clause, accumulator variables must be passed in `task_reduction_vars`,
symbols referring to reduction declarations in the `task_reduction_syms`
attribute, and whether the reduction variable should be passed into the
reduction region by value or by reference in `task_reduction_byref`.
"""
function taskgroup(allocate_vars::Vector{Value}, allocator_vars::Vector{Value}, task_reduction_vars::Vector{Value}; task_reduction_byref=nothing, task_reduction_syms=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[allocate_vars..., allocator_vars..., task_reduction_vars..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(allocate_vars), length(allocator_vars), length(task_reduction_vars), ]))
    !isnothing(task_reduction_byref) && push!(attributes, NamedAttribute("task_reduction_byref", task_reduction_byref))
    !isnothing(task_reduction_syms) && push!(attributes, NamedAttribute("task_reduction_syms", task_reduction_syms))
    
    IR.create_operation(
        "omp.taskgroup", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`taskloop`

The taskloop construct specifies that the iterations of one or more
associated loops will be executed in parallel using explicit tasks. The
iterations are distributed across tasks generated by the construct and
scheduled to be executed.

The body region can only contain a single block which must contain a single
operation. This operation must be another compatible loop wrapper or an
`omp.loop_nest`.

```
omp.taskloop <clauses> {
  omp.loop_nest (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
    %a = load %arrA[%i1, %i2] : memref<?x?xf32>
    %b = load %arrB[%i1, %i2] : memref<?x?xf32>
    %sum = arith.addf %a, %b : f32
    store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
    omp.yield
  }
}
```

For definitions of \"undeferred task\", \"included task\", \"final task\" and
\"mergeable task\", please check OpenMP Specification.

When an `if` clause is present on a taskloop construct, and if the `if`
clause expression evaluates to `false`, undeferred tasks are generated. The
use of a variable in an `if` clause expression of a taskloop construct
causes an implicit reference to the variable in all enclosing constructs.
  
The `allocator_vars` and `allocate_vars` parameters are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.
  
When a `final` clause is present and the `final` clause expression evaluates
to `true`, the generated tasks will be final tasks. All task constructs
encountered during execution of a final task will generate final and
included tasks. The use of a variable in a `final` clause expression causes
an implicit reference to the variable in all enclosing constructs.
  
If a `grainsize` clause is present, the number of logical loop iterations
assigned to each generated task is greater than or equal to the minimum of
the value of the grain-size expression and the number of logical loop
iterations, but less than two times the value of the grain-size expression.
  
When the `mergeable` clause is present, the tasks generated by the construct
are \"mergeable tasks\".
  
By default, the taskloop construct executes as if it was enclosed in a
taskgroup construct with no statements or directives outside of the taskloop
construct. Thus, the taskloop construct creates an implicit taskgroup
region. If the `nogroup` clause is present, no implicit taskgroup region is
created.
  
If `num_tasks` is specified, the taskloop construct creates as many tasks as
the minimum of the num-tasks expression and the number of logical loop
iterations. Each task must have at least one logical loop iteration.
  
The `priority` clause is a hint for the priority of the generated tasks.
The `priority` is a non-negative integer expression that provides a hint for
task execution order. Among all tasks ready to be executed, higher priority
tasks (those with a higher numerical value in the priority clause
expression) are recommended to execute before lower priority ones. The
default priority-value when no priority clause is specified should be
assumed to be zero (the lowest priority).
  
Reductions can be performed by specifying the reduction modifer
(`default`, `inscan` or `task`) in `reduction_mod`, reduction accumulator
variables in `reduction_vars`, symbols referring to reduction declarations
in the `reduction_syms` attribute, and whether the reduction variable
should be passed into the reduction region by value or by reference in
`reduction_byref`. Each reduction is identified by the accumulator it uses
and accumulators must not be repeated in the same reduction. A private
variable corresponding to the accumulator is used in place of the
accumulator inside the body of the operation. The reduction declaration
specifies how to combine the values from each iteration, section, team,
thread or simd lane defined by the operation\'s region into the final value,
which is available in the accumulator after they all complete.
  
If the `untied` clause is present on a task construct, any thread in the
team can resume the task region after a suspension. The `untied` clause is
ignored if a `final` clause is present on the same task construct and the
`final` expression evaluates to `true`, or if a task is an included task.
  
If an `in_reduction` clause is present on the taskloop construct, the
behavior is as if each generated task was defined by a task construct on
which an `in_reduction` clause with the same reduction operator and list
items is present. Thus, the generated tasks are participants of a reduction
previously defined by a reduction scoping clause. In this case, accumulator
variables are specified in `in_reduction_vars`, symbols referring to
reduction declarations in `in_reduction_syms` and `in_reduction_byref`
indicate for each reduction variable whether it should be passed by value or
by reference.

If a `reduction` clause is present on the taskloop construct, the behavior
is as if a `task_reduction` clause with the same reduction operator and list
items was applied to the implicit taskgroup construct enclosing the taskloop
construct. The taskloop construct executes as if each generated task was
defined by a task construct on which an `in_reduction` clause with the same
reduction operator and list items is present. Thus, the generated tasks are
participants of the reduction defined by the `task_reduction` clause that
was applied to the implicit taskgroup construct.
"""
function taskloop(allocate_vars::Vector{Value}, allocator_vars::Vector{Value}, final=nothing::Union{Nothing, Value}; grainsize=nothing::Union{Nothing, Value}, if_expr=nothing::Union{Nothing, Value}, in_reduction_vars::Vector{Value}, num_tasks=nothing::Union{Nothing, Value}, priority=nothing::Union{Nothing, Value}, private_vars::Vector{Value}, reduction_vars::Vector{Value}, in_reduction_byref=nothing, in_reduction_syms=nothing, mergeable=nothing, nogroup=nothing, private_syms=nothing, reduction_mod=nothing, reduction_byref=nothing, reduction_syms=nothing, untied=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[allocate_vars..., allocator_vars..., in_reduction_vars..., private_vars..., reduction_vars..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(final) && push!(operands, final)
    !isnothing(grainsize) && push!(operands, grainsize)
    !isnothing(if_expr) && push!(operands, if_expr)
    !isnothing(num_tasks) && push!(operands, num_tasks)
    !isnothing(priority) && push!(operands, priority)
    push!(attributes, operandsegmentsizes([length(allocate_vars), length(allocator_vars), Int(!isnothing(final)), Int(!isnothing(grainsize)), Int(!isnothing(if_expr)), length(in_reduction_vars), Int(!isnothing(num_tasks)), Int(!isnothing(priority)), length(private_vars), length(reduction_vars), ]))
    !isnothing(in_reduction_byref) && push!(attributes, NamedAttribute("in_reduction_byref", in_reduction_byref))
    !isnothing(in_reduction_syms) && push!(attributes, NamedAttribute("in_reduction_syms", in_reduction_syms))
    !isnothing(mergeable) && push!(attributes, NamedAttribute("mergeable", mergeable))
    !isnothing(nogroup) && push!(attributes, NamedAttribute("nogroup", nogroup))
    !isnothing(private_syms) && push!(attributes, NamedAttribute("private_syms", private_syms))
    !isnothing(reduction_mod) && push!(attributes, NamedAttribute("reduction_mod", reduction_mod))
    !isnothing(reduction_byref) && push!(attributes, NamedAttribute("reduction_byref", reduction_byref))
    !isnothing(reduction_syms) && push!(attributes, NamedAttribute("reduction_syms", reduction_syms))
    !isnothing(untied) && push!(attributes, NamedAttribute("untied", untied))
    
    IR.create_operation(
        "omp.taskloop", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`taskwait`

The taskwait construct specifies a wait on the completion of child tasks
of the current task.
  
The `depend_kinds` and `depend_vars` arguments are variadic lists of values
that specify the dependencies of this particular task in relation to other
tasks.
  
The optional `nowait` attribute, when present, eliminates the implicit
barrier at the end of the construct, so the parent operation can make
progress even if the child operation has not completed yet.
"""
function taskwait(depend_vars::Vector{Value}; depend_kinds=nothing, nowait=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[depend_vars..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(depend_kinds) && push!(attributes, NamedAttribute("depend_kinds", depend_kinds))
    !isnothing(nowait) && push!(attributes, NamedAttribute("nowait", nowait))
    
    IR.create_operation(
        "omp.taskwait", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`taskyield`

The taskyield construct specifies that the current task can be suspended
in favor of execution of a different task.
"""
function taskyield(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "omp.taskyield", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`teams`

The teams construct defines a region of code that triggers the creation of a
league of teams. Once created, the number of teams remains constant for the
duration of its code region.

If the `if_expr` is present and it evaluates to `false`, the number of teams
created is one.
  
The `allocator_vars` and `allocate_vars` parameters are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.
  
The optional `num_teams_upper` and `num_teams_lower` arguments specify the
limit on the number of teams to be created. If only the upper bound is
specified, it acts as if the lower bound was set to the same value. It is
not allowed to set `num_teams_lower` if `num_teams_upper` is not specified.
They define a closed range, where both the lower and upper bounds are
included.
  
Reductions can be performed by specifying the reduction modifer
(`default`, `inscan` or `task`) in `reduction_mod`, reduction accumulator
variables in `reduction_vars`, symbols referring to reduction declarations
in the `reduction_syms` attribute, and whether the reduction variable
should be passed into the reduction region by value or by reference in
`reduction_byref`. Each reduction is identified by the accumulator it uses
and accumulators must not be repeated in the same reduction. A private
variable corresponding to the accumulator is used in place of the
accumulator inside the body of the operation. The reduction declaration
specifies how to combine the values from each iteration, section, team,
thread or simd lane defined by the operation\'s region into the final value,
which is available in the accumulator after they all complete.
  
The optional `thread_limit` specifies the limit on the number of threads.
"""
function teams(allocate_vars::Vector{Value}, allocator_vars::Vector{Value}, if_expr=nothing::Union{Nothing, Value}; num_teams_lower=nothing::Union{Nothing, Value}, num_teams_upper=nothing::Union{Nothing, Value}, private_vars::Vector{Value}, reduction_vars::Vector{Value}, thread_limit=nothing::Union{Nothing, Value}, private_syms=nothing, reduction_mod=nothing, reduction_byref=nothing, reduction_syms=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[allocate_vars..., allocator_vars..., private_vars..., reduction_vars..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(if_expr) && push!(operands, if_expr)
    !isnothing(num_teams_lower) && push!(operands, num_teams_lower)
    !isnothing(num_teams_upper) && push!(operands, num_teams_upper)
    !isnothing(thread_limit) && push!(operands, thread_limit)
    push!(attributes, operandsegmentsizes([length(allocate_vars), length(allocator_vars), Int(!isnothing(if_expr)), Int(!isnothing(num_teams_lower)), Int(!isnothing(num_teams_upper)), length(private_vars), length(reduction_vars), Int(!isnothing(thread_limit)), ]))
    !isnothing(private_syms) && push!(attributes, NamedAttribute("private_syms", private_syms))
    !isnothing(reduction_mod) && push!(attributes, NamedAttribute("reduction_mod", reduction_mod))
    !isnothing(reduction_byref) && push!(attributes, NamedAttribute("reduction_byref", reduction_byref))
    !isnothing(reduction_syms) && push!(attributes, NamedAttribute("reduction_syms", reduction_syms))
    
    IR.create_operation(
        "omp.teams", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`terminator`

A terminator operation for regions that appear in the body of OpenMP
operation.  These regions are not expected to return any value so the
terminator takes no operands. The terminator op returns control to the
enclosing op.
"""
function terminator(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "omp.terminator", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`threadprivate`

The threadprivate directive specifies that variables are replicated, with
each thread having its own copy.

The current implementation uses the OpenMP runtime to provide thread-local
storage (TLS). Using the TLS feature of the LLVM IR will be supported in
future.

This operation takes in the address of a symbol that represents the original
variable and returns the address of its TLS. All occurrences of
threadprivate variables in a parallel region should use the TLS returned by
this operation.

The `sym_addr` refers to the address of the symbol, which is a pointer to
the original variable.
"""
function threadprivate(sym_addr::Value; tls_addr::IR.Type, location=Location())
    op_ty_results = IR.Type[tls_addr, ]
    operands = Value[sym_addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "omp.threadprivate", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`workshare_loop_wrapper`

This operation wraps a loop nest that is marked for dividing into units of
work by an encompassing omp.workshare operation.
"""
function workshare_loop_wrapper(; region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "omp.workshare.loop_wrapper", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`workshare`

The workshare construct divides the execution of the enclosed structured
block into separate units of work, and causes the threads of the team to
share the work such that each unit is executed only once by one thread, in
the context of its implicit task

This operation is used for the intermediate representation of the workshare
block before the work gets divided between the threads. See the flang
LowerWorkshare pass for details.
  
The optional `nowait` attribute, when present, eliminates the implicit
barrier at the end of the construct, so the parent operation can make
progress even if the child operation has not completed yet.
"""
function workshare(; nowait=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(nowait) && push!(attributes, NamedAttribute("nowait", nowait))
    
    IR.create_operation(
        "omp.workshare", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`wsloop`

The worksharing-loop construct specifies that the iterations of the loop(s)
will be executed in parallel by threads in the current context. These
iterations are spread across threads that already exist in the enclosing
parallel region.

The body region can only contain a single block which must contain a single
operation. This operation must be another compatible loop wrapper or an
`omp.loop_nest`.

```
omp.wsloop <clauses> {
  omp.loop_nest (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
    %a = load %arrA[%i1, %i2] : memref<?x?xf32>
    %b = load %arrB[%i1, %i2] : memref<?x?xf32>
    %sum = arith.addf %a, %b : f32
    store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
    omp.yield
  }
}
```
  
The `allocator_vars` and `allocate_vars` parameters are a variadic list of
values that specify the memory allocator to be used to obtain storage for
private values.
  
The `linear_step_vars` operand additionally specifies the step for each
associated linear operand. Note that the `linear_vars` and
`linear_step_vars` variadic lists should contain the same number of
elements.
  
The optional `nowait` attribute, when present, eliminates the implicit
barrier at the end of the construct, so the parent operation can make
progress even if the child operation has not completed yet.
  
The optional `order` attribute specifies which order the iterations of the
associated loops are executed in. Currently the only option for this
attribute is \"concurrent\".
  
The optional `ordered` attribute specifies how many loops are associated
with the worksharing-loop construct. The value of zero refers to the ordered
clause specified without parameter.
  
Reductions can be performed by specifying the reduction modifer
(`default`, `inscan` or `task`) in `reduction_mod`, reduction accumulator
variables in `reduction_vars`, symbols referring to reduction declarations
in the `reduction_syms` attribute, and whether the reduction variable
should be passed into the reduction region by value or by reference in
`reduction_byref`. Each reduction is identified by the accumulator it uses
and accumulators must not be repeated in the same reduction. A private
variable corresponding to the accumulator is used in place of the
accumulator inside the body of the operation. The reduction declaration
specifies how to combine the values from each iteration, section, team,
thread or simd lane defined by the operation\'s region into the final value,
which is available in the accumulator after they all complete.
  
The optional `schedule_kind` attribute specifies the loop schedule for this
loop, determining how the loop is distributed across the parallel threads.
The optional `schedule_chunk` associated with this determines further
controls this distribution.
"""
function wsloop(allocate_vars::Vector{Value}, allocator_vars::Vector{Value}, linear_vars::Vector{Value}, linear_step_vars::Vector{Value}, private_vars::Vector{Value}, reduction_vars::Vector{Value}, schedule_chunk=nothing::Union{Nothing, Value}; nowait=nothing, order=nothing, order_mod=nothing, ordered=nothing, private_syms=nothing, reduction_mod=nothing, reduction_byref=nothing, reduction_syms=nothing, schedule_kind=nothing, schedule_mod=nothing, schedule_simd=nothing, region::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[allocate_vars..., allocator_vars..., linear_vars..., linear_step_vars..., private_vars..., reduction_vars..., ]
    owned_regions = Region[region, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(schedule_chunk) && push!(operands, schedule_chunk)
    push!(attributes, operandsegmentsizes([length(allocate_vars), length(allocator_vars), length(linear_vars), length(linear_step_vars), length(private_vars), length(reduction_vars), Int(!isnothing(schedule_chunk)), ]))
    !isnothing(nowait) && push!(attributes, NamedAttribute("nowait", nowait))
    !isnothing(order) && push!(attributes, NamedAttribute("order", order))
    !isnothing(order_mod) && push!(attributes, NamedAttribute("order_mod", order_mod))
    !isnothing(ordered) && push!(attributes, NamedAttribute("ordered", ordered))
    !isnothing(private_syms) && push!(attributes, NamedAttribute("private_syms", private_syms))
    !isnothing(reduction_mod) && push!(attributes, NamedAttribute("reduction_mod", reduction_mod))
    !isnothing(reduction_byref) && push!(attributes, NamedAttribute("reduction_byref", reduction_byref))
    !isnothing(reduction_syms) && push!(attributes, NamedAttribute("reduction_syms", reduction_syms))
    !isnothing(schedule_kind) && push!(attributes, NamedAttribute("schedule_kind", schedule_kind))
    !isnothing(schedule_mod) && push!(attributes, NamedAttribute("schedule_mod", schedule_mod))
    !isnothing(schedule_simd) && push!(attributes, NamedAttribute("schedule_simd", schedule_simd))
    
    IR.create_operation(
        "omp.wsloop", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`yield`

\"omp.yield\" yields SSA values from the OpenMP dialect op region and
terminates the region. The semantics of how the values are yielded is
defined by the parent operation.
"""
function yield(results::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[results..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "omp.yield", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end
end # omp

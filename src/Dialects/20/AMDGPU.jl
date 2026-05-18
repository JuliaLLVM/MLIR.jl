module amdgpu

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes
import ...API


"""
`dpp`

This operation represents DPP functionality in a GPU program.
 DPP provides the following operations:
- Full crossbar in a group of four (`quad_perm`)
- Wavefront shift left by one lane (`wave_shl`)
- Wavefront shift right by one lane (`wave_shr`)
- Wavefront rotate right by one lane (`wave_ror`)
- Wavefront rotate left by one lane (`wave_rol`)
- Row shift left by 1–15 lanes (`row_shl`)
- Row shift right by 1–15 lanes (`row_shr`)
- Row rotate right by 1–15 lanes (`row_ror`)
- Reverse within a row (`row_mirror`)
- Reverse within a half-row (`row_half_mirror`)
- Broadcast the 15th lane of each row to the next row (`row_bcast`)
- Broadcast lane 31 to rows 2 and 3 (`row_bcast`)
"""
function dpp(old::Value, src::Value; result::IR.Type, kind, permArgument=nothing, row_mask=nothing, bank_mask=nothing, bound_ctrl=nothing, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[old, src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("kind", kind), ]
    !isnothing(permArgument) && push!(attributes, NamedAttribute("permArgument", permArgument))
    !isnothing(row_mask) && push!(attributes, NamedAttribute("row_mask", row_mask))
    !isnothing(bank_mask) && push!(attributes, NamedAttribute("bank_mask", bank_mask))
    !isnothing(bound_ctrl) && push!(attributes, NamedAttribute("bound_ctrl", bound_ctrl))
    
    create_operation(
        "amdgpu.dpp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`ext_packed_fp8`

Extend the value `source[index]` to a 32-bit float and return it.

This rather unusual signature arises from the fact that AMD GPUs cannot
easily work with sub 32-bit quantities, so the compiler intrinsics for
extending 8-bit floats (which are, currently, the only way to work with
this operation) take packed vectors of 4 such floats.

If the passed-in vector has fewer than four elements, or the input is scalar,
the remaining values in the <4 x i8> will be filled with with
undefined values as needed.
"""
function ext_packed_fp8(source::Value; res::IR.Type, index, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[source, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("index", index), ]
    
    create_operation(
        "amdgpu.ext_packed_fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`lds_barrier`

`amdgpu.lds_barrier` is both a barrier (all workitems in a workgroup must reach
the barrier before any of them may proceed past it) and a wait for all
operations that affect the Local Data Store (LDS) issued from that wrokgroup
to complete before the workgroup may continue. Since the LDS is per-workgroup
memory, this barrier may be used, for example, to ensure all workitems have
written data to LDS before any workitem attempts to read from it.

Note that `lds_barrier` does **not** force reads to or from global memory
to complete before execution continues. Therefore, it should be used when
operations on global memory can be issued far in advance of when their results
are used (for example, by writing them to LDS).

WARNING: On architectures that do not support the BackOffBarrier feature,
(those which will implement this barrier by emitting inline assembly),
use of this operation will impede the usabiliity of memory watches (including
breakpoints set on variables) when debugging.
"""
function lds_barrier(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "amdgpu.lds_barrier", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mfma`

The `amdgpu.mfma` op is an MLIR wrapper around intrinsics
for various `mfma` instructions in the CDNA architecture, which perform
multiple outer products in order to allow fast matrix multiplication.

The wrapper will select an appropriate `mfma` instruction, if one is available,
based on the provided `m`, `k`, `n`, and `nBlks` attributes, along with the
types of the source and destination arguments.

For information on the layouts of the input and output matrces (which are stored
in `sourceA`, `sourceB`, `destC`, and `destD`), see the CDNA ISA documentation.

The `cbsz`, `abid`, and `blgp` parameters control how the lanes of the wave
are permuted when matrix data is being loaded: `blgp` can be any number of
fixed permutations, `cbsz` specifies the log_2 of the number of chunks the lanes
holding sourceA are split into, and `abid` selects one of those chunks.

Note, this wrapper allows specifying `vector<4Kxi8>` arguments to MFMA
intrinsics that take an integer type of width `4K`. For example,
one can provide a vector<4xi8> as an argument to an MFMA instruction that
logically takes 4 i8s but whose intrinsics are specified to take an i32.
In these cases, the bytes in the vector will be concatenated in little-endian
order (that is, v[0] will go to arg[7:0], v[1] to arg[15:8] and so on).

The negateA, negateB, and negateC flags are only supported for double-precision
operations on gfx940+.
"""
function mfma(sourceA::Value, sourceB::Value, destC::Value; destD::IR.Type, m, n, k, blocks, cbsz=nothing, abid=nothing, blgp=nothing, reducePrecision=nothing, negateA=nothing, negateB=nothing, negateC=nothing, location=Location())
    op_ty_results = IR.Type[destD, ]
    operands = Value[sourceA, sourceB, destC, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("m", m), NamedAttribute("n", n), NamedAttribute("k", k), NamedAttribute("blocks", blocks), ]
    !isnothing(cbsz) && push!(attributes, NamedAttribute("cbsz", cbsz))
    !isnothing(abid) && push!(attributes, NamedAttribute("abid", abid))
    !isnothing(blgp) && push!(attributes, NamedAttribute("blgp", blgp))
    !isnothing(reducePrecision) && push!(attributes, NamedAttribute("reducePrecision", reducePrecision))
    !isnothing(negateA) && push!(attributes, NamedAttribute("negateA", negateA))
    !isnothing(negateB) && push!(attributes, NamedAttribute("negateB", negateB))
    !isnothing(negateC) && push!(attributes, NamedAttribute("negateC", negateC))
    
    create_operation(
        "amdgpu.mfma", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`packed_stoch_round_fp8`

Round the input `source`, adding in `stochiasticParam`, and place it into
the `storeIndex`th element of `res`.

If `existing` is passed in, elements of `res` other than the one at `storeIndex`
are copied from `existing`.

The reason for this odd signature is that AMD GPUs cannot easily work with
sub-registers, and so the conversion intrinsics (which are currently the
only way to work with 8-bit float types) take packed vectors of 4 8-bit
values.
"""
function packed_stoch_round_fp8(source::Value, stochiasticParam::Value, existing=nothing::Union{Nothing, Value}; res::IR.Type, storeIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[source, stochiasticParam, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("storeIndex", storeIndex), ]
    !isnothing(existing) && push!(operands, existing)
    
    create_operation(
        "amdgpu.packed_stoch_round_fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`packed_trunc_2xfp8`

Round the inputs `sourceA` and `sourceB` (which is undefined if not
specified) into the low or high word (bottom two or top two) elements
of the returned vector, keeping the other two elements of `existing`
unchanged if present (or undefined if it was not passed in).

The reason for this odd signature is that AMD GPUs cannot easily work with
sub-registers, and so the conversion intrinsics (which are currently the
only way to work with 8-bit float types) take packed vectors of 4 8-bit
values.
"""
function packed_trunc_2xfp8(sourceA::Value, sourceB=nothing::Union{Nothing, Value}; existing=nothing::Union{Nothing, Value}, res::IR.Type, wordIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[sourceA, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("wordIndex", wordIndex), ]
    !isnothing(sourceB) && push!(operands, sourceB)
    !isnothing(existing) && push!(operands, existing)
    push!(attributes, operandsegmentsizes([1, Int(!isnothing(sourceB)), Int(!isnothing(existing)), ]))
    
    create_operation(
        "amdgpu.packed_trunc_2xfp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`raw_buffer_atomic_cmpswap`

The `amdgpu.raw_buffer_atomic_cmpswap` op is a wrapper around the
buffer-based atomic compare-and-swap min available on AMD GPUs.

The index into the buffer is computed as for `memref.store` with the addition
of `indexOffset` (which is used to aid in emitting vectorized code) and,
if present `sgprOffset` (which is added after bounds checks and includes
any non-zero offset on the memref type).

All indexing components are given in terms of the memref\'s element size, not
the byte lengths required by the intrinsic.

Out of bounds atomic operations are ignored in hardware.

See `amdgpu.raw_buffer_load` for a description of how the underlying
instruction is constructed.
"""
function raw_buffer_atomic_cmpswap(src::Value, cmp::Value, memref::Value, indices::Vector{Value}, sgprOffset=nothing::Union{Nothing, Value}; value::IR.Type, boundsCheck=nothing, indexOffset=nothing, location=Location())
    op_ty_results = IR.Type[value, ]
    operands = Value[src, cmp, memref, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(sgprOffset) && push!(operands, sgprOffset)
    push!(attributes, operandsegmentsizes([1, 1, 1, length(indices), Int(!isnothing(sgprOffset)), ]))
    !isnothing(boundsCheck) && push!(attributes, NamedAttribute("boundsCheck", boundsCheck))
    !isnothing(indexOffset) && push!(attributes, NamedAttribute("indexOffset", indexOffset))
    
    create_operation(
        "amdgpu.raw_buffer_atomic_cmpswap", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`raw_buffer_atomic_fadd`

The `amdgpu.raw_buffer_atomic_fadd` op is a wrapper around the
buffer-based atomic floating point addition available on the MI-* series
of AMD GPUs.

The index into the buffer is computed as for `memref.store` with the addition
of `indexOffset` (which is used to aid in emitting vectorized code) and,
if present `sgprOffset` (which is added after bounds checks and includes
any non-zero offset on the memref type).

All indexing components are given in terms of the memref\'s element size, not
the byte lengths required by the intrinsic.

Out of bounds atomic operations are ignored in hardware.

See `amdgpu.raw_buffer_load` for a description of how the underlying
instruction is constructed.
"""
function raw_buffer_atomic_fadd(value::Value, memref::Value, indices::Vector{Value}, sgprOffset=nothing::Union{Nothing, Value}; boundsCheck=nothing, indexOffset=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, memref, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(sgprOffset) && push!(operands, sgprOffset)
    push!(attributes, operandsegmentsizes([1, 1, length(indices), Int(!isnothing(sgprOffset)), ]))
    !isnothing(boundsCheck) && push!(attributes, NamedAttribute("boundsCheck", boundsCheck))
    !isnothing(indexOffset) && push!(attributes, NamedAttribute("indexOffset", indexOffset))
    
    create_operation(
        "amdgpu.raw_buffer_atomic_fadd", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`raw_buffer_atomic_fmax`

The `amdgpu.raw_buffer_atomic_fmax` op is a wrapper around the
buffer-based atomic floating point max available on AMD GPUs (except GFX9).

The index into the buffer is computed as for `memref.store` with the addition
of `indexOffset` (which is used to aid in emitting vectorized code) and,
if present `sgprOffset` (which is added after bounds checks and includes
any non-zero offset on the memref type).

All indexing components are given in terms of the memref\'s element size, not
the byte lengths required by the intrinsic.

Out of bounds atomic operations are ignored in hardware.

See `amdgpu.raw_buffer_load` for a description of how the underlying
instruction is constructed.
"""
function raw_buffer_atomic_fmax(value::Value, memref::Value, indices::Vector{Value}, sgprOffset=nothing::Union{Nothing, Value}; boundsCheck=nothing, indexOffset=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, memref, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(sgprOffset) && push!(operands, sgprOffset)
    push!(attributes, operandsegmentsizes([1, 1, length(indices), Int(!isnothing(sgprOffset)), ]))
    !isnothing(boundsCheck) && push!(attributes, NamedAttribute("boundsCheck", boundsCheck))
    !isnothing(indexOffset) && push!(attributes, NamedAttribute("indexOffset", indexOffset))
    
    create_operation(
        "amdgpu.raw_buffer_atomic_fmax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`raw_buffer_atomic_smax`

The `amdgpu.raw_buffer_atomic_smax` op is a wrapper around the
buffer-based atomic signed integer max available on AMD GPUs.

The index into the buffer is computed as for `memref.store` with the addition
of `indexOffset` (which is used to aid in emitting vectorized code) and,
if present `sgprOffset` (which is added after bounds checks and includes
any non-zero offset on the memref type).

All indexing components are given in terms of the memref\'s element size, not
the byte lengths required by the intrinsic.

Out of bounds atomic operations are ignored in hardware.

See `amdgpu.raw_buffer_load` for a description of how the underlying
instruction is constructed.
"""
function raw_buffer_atomic_smax(value::Value, memref::Value, indices::Vector{Value}, sgprOffset=nothing::Union{Nothing, Value}; boundsCheck=nothing, indexOffset=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, memref, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(sgprOffset) && push!(operands, sgprOffset)
    push!(attributes, operandsegmentsizes([1, 1, length(indices), Int(!isnothing(sgprOffset)), ]))
    !isnothing(boundsCheck) && push!(attributes, NamedAttribute("boundsCheck", boundsCheck))
    !isnothing(indexOffset) && push!(attributes, NamedAttribute("indexOffset", indexOffset))
    
    create_operation(
        "amdgpu.raw_buffer_atomic_smax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`raw_buffer_atomic_umin`

The `amdgpu.raw_buffer_atomic_umin` op is a wrapper around the
buffer-based atomic signed integer min available on AMD GPUs.

The index into the buffer is computed as for `memref.store` with the addition
of `indexOffset` (which is used to aid in emitting vectorized code) and,
if present `sgprOffset` (which is added after bounds checks and includes
any non-zero offset on the memref type).

All indexing components are given in terms of the memref\'s element size, not
the byte lengths required by the intrinsic.

Out of bounds atomic operations are ignored in hardware.

See `amdgpu.raw_buffer_load` for a description of how the underlying
instruction is constructed.
"""
function raw_buffer_atomic_umin(value::Value, memref::Value, indices::Vector{Value}, sgprOffset=nothing::Union{Nothing, Value}; boundsCheck=nothing, indexOffset=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, memref, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(sgprOffset) && push!(operands, sgprOffset)
    push!(attributes, operandsegmentsizes([1, 1, length(indices), Int(!isnothing(sgprOffset)), ]))
    !isnothing(boundsCheck) && push!(attributes, NamedAttribute("boundsCheck", boundsCheck))
    !isnothing(indexOffset) && push!(attributes, NamedAttribute("indexOffset", indexOffset))
    
    create_operation(
        "amdgpu.raw_buffer_atomic_umin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`raw_buffer_load`

The `amdgpu.raw_buffer_load` op is a wrapper around the buffer load intrinsics
available on AMD GPUs, including extensions in newer GPUs.

The index into the buffer is computed as for `memref.load` with the additon
of `indexOffset` and `sgprOffset` (which **may or may not** be considered
in bounds checks and includes any offset present on the memref type if it\'s
non-zero).

All indices and offsets are in units of the memref\'s data type and are
converted to bytes during lowering.

When a load is out of bounds, the instruction returns zero.
Partially-out of bounds have chipset-dependent behavior: whether reading
2 elements starting at index 7 of a `memref<8xf32>` returns the last element
in the first vector component depends on the architecture.

The memref struct is converted into a buffer resource (a V#) and the arguments
are translated to intrinsic arguments as follows:
- The base address of the buffer is the base address of the memref
- The stride is 0 to enable raw mode
- The number of records is the size of the memref, in bytes
  In the case of dynamically-shaped memrefs, this is computed at runtime
  as max_d (size(d) * stride(d)) * sizeof(elementType(memref))
- The offset enable bit is 1, the index enable bit is 0.
- The thread ID addition bit is off
- If `boundsCheck` is false and the target chipset is RDNA, OOB_SELECT is set
  to 2 to disable bounds checks, otherwise it is 3
- The cache coherency bits are off
"""
function raw_buffer_load(memref::Value, indices::Vector{Value}, sgprOffset=nothing::Union{Nothing, Value}; value::IR.Type, boundsCheck=nothing, indexOffset=nothing, location=Location())
    op_ty_results = IR.Type[value, ]
    operands = Value[memref, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(sgprOffset) && push!(operands, sgprOffset)
    push!(attributes, operandsegmentsizes([1, length(indices), Int(!isnothing(sgprOffset)), ]))
    !isnothing(boundsCheck) && push!(attributes, NamedAttribute("boundsCheck", boundsCheck))
    !isnothing(indexOffset) && push!(attributes, NamedAttribute("indexOffset", indexOffset))
    
    create_operation(
        "amdgpu.raw_buffer_load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`raw_buffer_store`

The `amdgpu.raw_buffer_store` op is a wrapper around the buffer store
intrinsics available on AMD GPUs, including extensions in newer GPUs.

The store index is computed as in `memref.store` with the addition of
`indexOffset` (which is included for uniformity with atomics and may be useful
when writing vectorized code) and `sgprOffset` (which is added after bounds
checks and implicitly includes the offset of the memref type if non-zero).
All index components are in terms of the elements of the memref, not bytes,
and are scaled up appropriately.

Out of bounds stores are ignored in hardware.
Wthether a vector write that includes some in-bounds and soeme out-of-bounds
components is partically completed is chipset-dependent.

See `amdgpu.raw_buffer_load` for a description of how the underlying
instruction is constructed.
"""
function raw_buffer_store(value::Value, memref::Value, indices::Vector{Value}, sgprOffset=nothing::Union{Nothing, Value}; boundsCheck=nothing, indexOffset=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, memref, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(sgprOffset) && push!(operands, sgprOffset)
    push!(attributes, operandsegmentsizes([1, 1, length(indices), Int(!isnothing(sgprOffset)), ]))
    !isnothing(boundsCheck) && push!(attributes, NamedAttribute("boundsCheck", boundsCheck))
    !isnothing(indexOffset) && push!(attributes, NamedAttribute("indexOffset", indexOffset))
    
    create_operation(
        "amdgpu.raw_buffer_store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`sched_barrier`

`amdgpu.sched_barrier` serves as a barrier that could be
configured to restrict movements of instructions through it as
defined by sched_barrier_opts.
"""
function sched_barrier(; opts, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("opts", opts), ]
    
    create_operation(
        "amdgpu.sched_barrier", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`wmma`

The `amdgpu.wmma` op is an MLIR wrapper around intrinsics
for various `wmma` instructions in the RDNA3 architecture, which perform
a 16x16 matrix multiplication for different data types.

When emitting f16->f16 (or bf16->bf16) wmma the output is a 16xf16 (or 16xbf16) vector
containing only 8 valid values:
  - If `subwordOffset` is 0, then the output is stored at indices 0, 2, 4, ..., 14.
  - If `subwordOffset` is 1, then the output is stored at indices 1, 3, 5, ..., 15.

`unsignedA` and `unsignedB` flag that the `int8` LLVM inputs are unsigned.

The `clamp` flag is used to saturate the output of type T to numeric_limits<T>::max()
in case of overflow.
"""
function wmma(sourceA::Value, sourceB::Value, destC::Value; destD::IR.Type, subwordOffset=nothing, unsignedA=nothing, unsignedB=nothing, clamp=nothing, location=Location())
    op_ty_results = IR.Type[destD, ]
    operands = Value[sourceA, sourceB, destC, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(subwordOffset) && push!(attributes, NamedAttribute("subwordOffset", subwordOffset))
    !isnothing(unsignedA) && push!(attributes, NamedAttribute("unsignedA", unsignedA))
    !isnothing(unsignedB) && push!(attributes, NamedAttribute("unsignedB", unsignedB))
    !isnothing(clamp) && push!(attributes, NamedAttribute("clamp", clamp))
    
    create_operation(
        "amdgpu.wmma", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

end # amdgpu

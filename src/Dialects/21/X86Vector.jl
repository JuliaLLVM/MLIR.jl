module x86vector

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes

"""
`avx_bcst_to_f32_packed`

#### From the Intel Intrinsics Guide:

Convert scalar BF16 or F16 (16-bit) floating-point element stored at memory locations
starting at location `__A` to a single-precision (32-bit) floating-point,
broadcast it to packed single-precision (32-bit) floating-point elements,
and store the results in `dst`.

# Example
```mlir
%dst = x86vector.avx.bcst_to_f32.packed %a : memref<1xbf16> -> vector<8xf32>
%dst = x86vector.avx.bcst_to_f32.packed %a : memref<1xf16> -> vector<8xf32>
```
"""
function avx_bcst_to_f32_packed(a::Value; dst::IR.Type, location=Location())
    op_ty_results = IR.Type[dst, ]
    operands = Value[a, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "x86vector.avx.bcst_to_f32.packed", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`avx_cvt_packed_even_indexed_to_f32`

#### From the Intel Intrinsics Guide:

Convert packed BF16 or F16 (16-bit) floating-point even-indexed elements stored at
memory locations starting at location `__A` to packed single-precision
(32-bit) floating-point elements, and store the results in `dst`.

# Example
```mlir
%dst = x86vector.avx.cvt.packed.even.indexed_to_f32 %a : memref<16xbf16> -> vector<8xf32>
%dst = x86vector.avx.cvt.packed.even.indexed_to_f32 %a : memref<16xf16> -> vector<8xf32>
```
"""
function avx_cvt_packed_even_indexed_to_f32(a::Value; dst::IR.Type, location=Location())
    op_ty_results = IR.Type[dst, ]
    operands = Value[a, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "x86vector.avx.cvt.packed.even.indexed_to_f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`avx512_cvt_packed_f32_to_bf16`

The `convert_f32_to_bf16` op is an AVX512-BF16 specific op that can lower
to the proper LLVMAVX512BF16 operation `llvm.cvtneps2bf16` depending on
the width of MLIR vectors it is applied to.

#### From the Intel Intrinsics Guide:

Convert packed single-precision (32-bit) floating-point elements in `a` to
packed BF16 (16-bit) floating-point elements, and store the results in `dst`.

# Example
```mlir
%dst = x86vector.avx512.cvt.packed.f32_to_bf16 %a : vector<8xf32> -> vector<8xbf16>
```
"""
function avx512_cvt_packed_f32_to_bf16(a::Value; dst::IR.Type, location=Location())
    op_ty_results = IR.Type[dst, ]
    operands = Value[a, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "x86vector.avx512.cvt.packed.f32_to_bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`avx_cvt_packed_odd_indexed_to_f32`

#### From the Intel Intrinsics Guide:

Convert packed BF16 or F16 (16-bit) floating-point odd-indexed elements stored at
memory locations starting at location `__A` to packed single-precision
(32-bit) floating-point elements, and store the results in `dst`.

# Example
```mlir
%dst = x86vector.avx.cvt.packed.odd.indexed_to_f32 %a : memref<16xbf16> -> vector<8xf32>
%dst = x86vector.avx.cvt.packed.odd.indexed_to_f32 %a : memref<16xf16> -> vector<8xf32>
```
"""
function avx_cvt_packed_odd_indexed_to_f32(a::Value; dst::IR.Type, location=Location())
    op_ty_results = IR.Type[dst, ]
    operands = Value[a, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "x86vector.avx.cvt.packed.odd.indexed_to_f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`avx512_dot`

The `dot` op is an AVX512-BF16 specific op that can lower to the proper
LLVMAVX512BF16 operation `llvm.dpbf16ps` depending on the width of MLIR
vectors it is applied to.

#### From the Intel Intrinsics Guide:

Compute dot-product of BF16 (16-bit) floating-point pairs in `a` and `b`,
accumulating the intermediate single-precision (32-bit) floating-point
elements with elements in `src`, and store the results in `dst`.

# Example
```mlir
%dst = x86vector.avx512.dot %src, %a, %b : vector<32xbf16> -> vector<16xf32>
```
"""
function avx512_dot(src::Value, a::Value, b::Value; dst=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[src, a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(dst) && push!(op_ty_results, dst)
    
    IR.create_operation(
        "x86vector.avx512.dot", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`avx_dot_i8`

The `dot` op is an AVX2-Int8 specific op that can lower to the proper
LLVMAVX2-INT8 operation `llvm.vpdpbssd` depending on the width of MLIR
vectors it is applied to.

#### From the Intel Intrinsics Guide:

Multiply groups of 4 adjacent pairs of signed 8-bit integers in `a` with 
corresponding signed 8-bit integers in `b`, producing 4 intermediate signed 16-bit 
results. Sum these 4 results with the corresponding 32-bit integer in `w`, and 
store the packed 32-bit results in `dst`.

# Example
```mlir
%dst = x86vector.avx.dot.i8 %w, %a, %b : vector<32xi8> -> vector<8xi32>
```
"""
function avx_dot_i8(w::Value, a::Value, b::Value; dst=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[w, a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(dst) && push!(op_ty_results, dst)
    
    IR.create_operation(
        "x86vector.avx.dot.i8", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`avx_intr_dot`

Computes the 4-way dot products of the lower and higher parts of the source
vectors and broadcasts the two results to the lower and higher elements of
the destination vector, respectively. Adding one element of the lower part
to one element of the higher part in the destination vector yields the full
dot product of the two source vectors.

# Example

```mlir
%0 = x86vector.avx.intr.dot %a, %b : vector<8xf32>
%1 = vector.extractelement %0[%i0 : i32]: vector<8xf32>
%2 = vector.extractelement %0[%i4 : i32]: vector<8xf32>
%d = arith.addf %1, %2 : f32
```
"""
function avx_intr_dot(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "x86vector.avx.intr.dot", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`avx512_mask_compress`

The mask.compress op is an AVX512 specific op that can lower to the
`llvm.mask.compress` instruction. Instead of `src`, a constant vector
vector attribute `constant_src` may be specified. If neither `src` nor
`constant_src` is specified, the remaining elements in the result vector are
set to zero.

#### From the Intel Intrinsics Guide:

Contiguously store the active integer/floating-point elements in `a` (those
with their respective bit set in writemask `k`) to `dst`, and pass through the
remaining elements from `src`.
"""
function avx512_mask_compress(k::Value, a::Value, src=nothing::Union{Nothing, Value}; dst=nothing::Union{Nothing, IR.Type}, constant_src=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[k, a, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(src) && push!(operands, src)
    !isnothing(dst) && push!(op_ty_results, dst)
    !isnothing(constant_src) && push!(attributes, NamedAttribute("constant_src", constant_src))
    
    IR.create_operation(
        "x86vector.avx512.mask.compress", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`avx512_mask_rndscale`

The mask.rndscale op is an AVX512 specific op that can lower to the proper
LLVMAVX512 operation: `llvm.mask.rndscale.ps.512` or
`llvm.mask.rndscale.pd.512` instruction depending on the type of vectors it
is applied to.

#### From the Intel Intrinsics Guide:

Round packed floating-point elements in `a` to the number of fraction bits
specified by `imm`, and store the results in `dst` using writemask `k`
(elements are copied from src when the corresponding mask bit is not set).
"""
function avx512_mask_rndscale(src::Value, k::Value, a::Value, imm::Value, rounding::Value; dst=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[src, k, a, imm, rounding, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(dst) && push!(op_ty_results, dst)
    
    IR.create_operation(
        "x86vector.avx512.mask.rndscale", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`avx512_mask_scalef`

The `mask.scalef` op is an AVX512 specific op that can lower to the proper
LLVMAVX512 operation: `llvm.mask.scalef.ps.512` or
`llvm.mask.scalef.pd.512` depending on the type of MLIR vectors it is
applied to.

#### From the Intel Intrinsics Guide:

Scale the packed floating-point elements in `a` using values from `b`, and
store the results in `dst` using writemask `k` (elements are copied from src
when the corresponding mask bit is not set).
"""
function avx512_mask_scalef(src::Value, a::Value, b::Value, k::Value, rounding::Value; dst=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[src, a, b, k, rounding, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(dst) && push!(op_ty_results, dst)
    
    IR.create_operation(
        "x86vector.avx512.mask.scalef", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function avx_rsqrt(a::Value; b=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(b) && push!(op_ty_results, b)
    
    IR.create_operation(
        "x86vector.avx.rsqrt", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`avx512_vp2intersect`

The `vp2intersect` op is an AVX512 specific op that can lower to the proper
LLVMAVX512 operation: `llvm.vp2intersect.d.512` or
`llvm.vp2intersect.q.512` depending on the type of MLIR vectors it is
applied to.

#### From the Intel Intrinsics Guide:

Compute intersection of packed integer vectors `a` and `b`, and store
indication of match in the corresponding bit of two mask registers
specified by `k1` and `k2`. A match in corresponding elements of `a` and
`b` is indicated by a set bit in the corresponding bit of the mask
registers.
"""
function avx512_vp2intersect(a::Value, b::Value; k1=nothing::Union{Nothing, IR.Type}, k2=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(k1) && push!(op_ty_results, k1)
    !isnothing(k2) && push!(op_ty_results, k2)
    
    IR.create_operation(
        "x86vector.avx512.vp2intersect", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end
end # x86vector

module xevm

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes

"""
`blockload2d`

The `xevm.blockload2d` operation loads a two dimensional matrix tile
from a base matrix residing in global memory. The parameters are:
  * `ptr` - the base address of the base matrix containing the tile to load
  * `base_width` - the width of the base matrix in number of bytes.
  * `base_height` - the number of rows in the base matrix
  * `base_pitch` - the physical stride between the first columns of the current
    row and the subsequent row in number of bytes.
  * `x`, `y`, `tile_width`, `tile_height` - the starting offsets and shape of
    the tile to load in number of elements.
  * `elem_size_in_bits` - the size in bits of the matrix element type
    - 32 for f32, tf32
    - 16 for f16, int16, bf16
    - 8 for int8
  * `v_blocks` - number of consecutive tiles in innermost dimension direction to load
  * `transpose` - transpose the tile in registers (useful for 32 bit element type)
  * `pack_register` - pack element types narrower than register bit width.
    [M, N] => [M/factor, N, factor] where factor is register_size_in_bits / elem_size_in_bits
  * `cache_control` - an enumerator that sets the cache behaviour

Notes:
  - the `transpose` and `pack_register` parameters are mutual exclusive
  - transposing the tile loaded is used for A matrix in backward path or used for the B matrix operand
    (D = C + A * B), where A has row-major layout and B should have column-major layout in memory.
  - if the tile loaded contains out of bound elements of the matrix, they are filled with 0.

# Example
```mlir
  %base_width_a = arith.constant 32 : i32
  %base_height_a = arith.constant 8 : i32
  %base_pitch_a = arith.constant 32 : i32
  %x = arith.constant 0 : i32
  %y = arith.constant 0 : i32
  %loaded_a = xevm.blockload2d %src, %base_width_a, %base_height_a, %base_pitch_a, %x, %y
                <{elem_size_in_bits=16 : i32, tile_width=16 : i32, tile_height=8 : i32,
                  v_blocks=1 : i32, transpose=false : i32, pack_register=false,
                  cache_control=#xevm.load_cache_control<Default>}>
                : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
```
"""
function blockload2d(ptr::Value, base_width::Value, base_height::Value, base_pitch::Value, x::Value, y::Value; res::IR.Type, elem_size_in_bits, tile_width, tile_height, v_blocks, transpose, pack_register, cache_control=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, base_width, base_height, base_pitch, x, y, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("elem_size_in_bits", elem_size_in_bits), NamedAttribute("tile_width", tile_width), NamedAttribute("tile_height", tile_height), NamedAttribute("v_blocks", v_blocks), NamedAttribute("transpose", transpose), NamedAttribute("pack_register", pack_register), ]
    !isnothing(cache_control) && push!(attributes, NamedAttribute("cache_control", cache_control))
    
    IR.create_operation(
        "xevm.blockload2d", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`blockprefetch2d`

The `xevm.blockprefetch2d` operation prefetches a two dimensional tile
from a larger base matrix residing in global memory. The parameters are:
  * `ptr` - the base address of the base matrix containing the tile to prefetch
  * `base_width` - the width of the base matrix in number of bytes.
  * `base_height` - the number of rows in the base matrix
  * `base_pitch` - the physical stride between the first columns of the current
    row and the subsequent row in number of bytes.
  * `x`, `y`, `tile_width`, `tile_height` - the starting offsets and shape of tile
    to prefetch in number of elements.
  * `elem_size_in_bits` - the size in bits of the matrix element
    - 32 for f32, bf32
    - 16 for f16, int16, bf16
    - 8 for int8, int4, int2
  * `v_blocks` - number of tiles in innermost dimension direction to prefetch
  * `cache_control` - an enumerator that sets the cache behaviour

# Example
```mlir
  xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y
    <{elem_size_in_bits=8 : i32, tile_width=32 : i32, tile_height=8 : i32,
      v_blocks=1 : i32, cache_control=#xevm.load_cache_control<L1uc_L2uc_L3uc>}>
    : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
```
"""
function blockprefetch2d(ptr::Value, base_width::Value, base_height::Value, base_pitch::Value, x::Value, y::Value; elem_size_in_bits, tile_width, tile_height, v_blocks, cache_control=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, base_width, base_height, base_pitch, x, y, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("elem_size_in_bits", elem_size_in_bits), NamedAttribute("tile_width", tile_width), NamedAttribute("tile_height", tile_height), NamedAttribute("v_blocks", v_blocks), ]
    !isnothing(cache_control) && push!(attributes, NamedAttribute("cache_control", cache_control))
    
    IR.create_operation(
        "xevm.blockprefetch2d", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`blockstore2d`

The `xevm.blockstore2d` operation stores a two dimensional tile into a
larger matrix residing in global memory. The parameters are:
  * `ptr` - the base address of the target matrix where to store the tile
  * `base_width` - the width of the base matrix in number of bytes.
  * `base_height` - the number of rows in the base matrix
  * `base_pitch` - the physical stride between the first columns of the current
    row and the subsequent row in number of bytes.
  * `x`, `y`, `tile_width`, `tile_height` - the starting offsets and shape of the tile to store
  in number of elements.
  * `elem_size_in_bits` - the size in bits of the matrix element
    - 32 for f32, tf32
    - 16 for f16, int16, bf16
    - 8 for int8
  * `cache_control` - an enumerator that sets the cache behaviour
  * `stored_val` - the tile to store

# Example
```mlir
  %base_width_c = arith.constant 64 : i32
  %base_height_c = arith.constant 8 : i32
  %base_pitch_c = arith.constant 64 : i32
  %x = arith.constant 0 : i32
  %y = arith.constant 0 : i32
  xevm.blockstore2d %dst, %base_width_c, %base_height_c, %base_pitch_c, %x, %y, %src
    <{elem_size_in_bits=32 : i32, tile_width=16 : i32, tile_height=8 : i32,
      cache_control=#xevm.load_cache_control<Default>}>
    : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
```
"""
function blockstore2d(ptr::Value, base_width::Value, base_height::Value, base_pitch::Value, x::Value, y::Value, stored_val::Value; elem_size_in_bits, tile_width, tile_height, cache_control=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, base_width, base_height, base_pitch, x, y, stored_val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("elem_size_in_bits", elem_size_in_bits), NamedAttribute("tile_width", tile_width), NamedAttribute("tile_height", tile_height), ]
    !isnothing(cache_control) && push!(attributes, NamedAttribute("cache_control", cache_control))
    
    IR.create_operation(
        "xevm.blockstore2d", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mma`

The `xevm.mma` is a cooperative operation where all threads/lanes in
a subgroup participates and carries out matrix multiplication plus accumulation:

  D = C + A x B

  where the A, B, C input matrices and the result D have shapes:
    - D : MxN
    - C : MxN
    - A : MxK
    - B : KxN

Parameters:
  * `a` - vector of matrix A elements.
  * `b` - vector of matrix B elements.
  * `c` - (optional) vector of matrix C elements.
  * `shape` - the shape of the matrices, specified as `M`, `N`, and `K` values.
  * `types` - the data types of the matrices, specified as `D`, `A`, `B`, and optionally `C`.

# Example
```mlir
  %d = xevm.mma %a, %b, %c { shape=<m=8, n=16, k=16>, types=<d=f32, a=f16, b=f16, c=f32> }
         : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
```
"""
function mma(a::Value, b::Value, c=nothing::Union{Nothing, Value}; d::IR.Type, shape, types, location=Location())
    op_ty_results = IR.Type[d, ]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("shape", shape), NamedAttribute("types", types), ]
    !isnothing(c) && push!(operands, c)
    
    IR.create_operation(
        "xevm.mma", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`memfence`

This operation ensures that all prior memory accesses of this
work-item to `addrspace` are visible to all other work-items in `scope`.
Parameters description:
  * `scope` - specify the memory scope at which all other work-items should observe
    memory operations prior to the fence.
  * `addrspace` - specify the address space of work-item\'s memory accesses
    to be affected by the fence.
"""
function memfence(; scope, addrspace=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("scope", scope), ]
    !isnothing(addrspace) && push!(attributes, NamedAttribute("addrspace", addrspace))
    
    IR.create_operation(
        "xevm.memfence", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`prefetch`

Work-item issues a prefetch from global memory to cache:
  * `ptr` - LLVM pointer with address space. Address space must be 1 (global)
    or 4 (generic)
  * `cache_control` - specify caching options
"""
function prefetch(ptr::Value; cache_control=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(cache_control) && push!(attributes, NamedAttribute("cache_control", cache_control))
    
    IR.create_operation(
        "xevm.prefetch", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end
end # xevm

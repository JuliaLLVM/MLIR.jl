module arm_neon

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`intr_smull`

Signed Multiply Long (vector). This instruction multiplies corresponding
signed integer values in the lower or upper half of the vectors of the two
source SIMD&FP registers, places the results in a vector, and writes the
vector to the destination SIMD&FP register.

Source:
https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics
"""
function intr_smull(a::Value, b::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_neon.intr.smull", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`2d_sdot`

The two input vectors `b` and `c` have a 2D shape, consisting of either 2
or 4 rows, each row having length 4. This operation computes the pair-wise
dot-products of the rows of `b` and `c` and accumulates them with the
corresponding entry of `a`:

```
res[i] := a[i] + dot_product(b[i, ...], c[i, ...])
```
"""
# function 2d_sdot(a::Value, b::Value, c::Value; res::MLIRType, location=Location())
#     results = MLIRType[res, ]
#     operands = Value[a, b, c, ]
#     owned_regions = Region[]
#     successors = Block[]
#     attributes = NamedAttribute[]
    
#     create_operation(
#         "arm_neon.2d.sdot", location;
#         operands, owned_regions, successors, attributes,
#         results=results,
#         result_inference=false
#     )
# end

"""
`intr_sdot`

Signed integer addition of dot product (vector). This instruction performs
the following operation on signed integer vectors: res = dot(b, c) + a,
where vector operands are partitioned into groups of four elements.

Source:
https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics
"""
function intr_sdot(a::Value, b::Value, c::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[a, b, c, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_neon.intr.sdot", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # arm_neon

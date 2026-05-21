module smt

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes

"""
`and`
This operation performs a boolean conjunction.
    The semantics are equivalent to the \'and\' operator in the
    [Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2).
    of the SMT-LIB Standard 2.7.

    It supports a variadic number of operands, but requires at least two.
    This is because the operator is annotated with the `:left-assoc` attribute
    which means that `op a b c` is equivalent to `(op (op a b) c)`.
"""
function and(inputs::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.and", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`apply_func`

This operation performs a function application as described in the
[SMT-LIB 2.7 standard](https://smt-lib.org/papers/smt-lib-reference-v2.7-r2025-02-05.pdf).
It is part of the language itself rather than a theory or logic.
"""
function apply_func(func::Value, args::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[func, args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.apply_func", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`array_broadcast`

This operation represents a broadcast of the \'value\' operand to all indices
of the array. It is equivalent to
```
%0 = smt.declare_fun \"array\" : !smt.array<[!smt.int -> !smt.bool]>
%1 = smt.forall [\"idx\"] {
^bb0(%idx: !smt.int):
  %2 = smt.array.select %0[%idx] : !smt.array<[!smt.int -> !smt.bool]>
  %3 = smt.eq %value, %2 : !smt.bool
  smt.yield %3 : !smt.bool
}
smt.assert %1
// return %0
```

In SMT-LIB, this is frequently written as
`((as const (Array Int Bool)) value)`.
"""
function array_broadcast(value::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "smt.array.broadcast", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`array_select`

This operation is retuns the value stored in the given array at the given
index. The semantics are equivalent to the `select` operator defined in the
[SMT ArrayEx theory](https://smtlib.cs.uiowa.edu/Theories/ArraysEx.smt2) of
the SMT-LIB standard 2.7.
"""
function array_select(array::Value, index::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[array, index, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.array.select", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`array_store`

This operation returns a new array which is the same as the \'array\' operand
except that the value at the given \'index\' is changed to the given \'value\'.
The semantics are equivalent to the \'store\' operator described in the
[SMT ArrayEx theory](https://smtlib.cs.uiowa.edu/Theories/ArraysEx.smt2) of
the SMT-LIB standard 2.7.
"""
function array_store(array::Value, index::Value, value::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[array, index, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.array.store", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function assert(input::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "smt.assert", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`bv2int`

Create an integer from the bit-vector argument `input`. If `is_signed` is
present, the bit-vector is treated as two\'s complement signed.  Otherwise,
it is treated as an unsigned integer in the range [0..2^N-1], where N is
the number of bits in `input`.
"""
function bv2int(input::Value; result=nothing::Union{Nothing, IR.Type}, is_signed=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(is_signed) && push!(attributes, NamedAttribute("is_signed", is_signed))
    
    IR.create_operation(
        "smt.bv2int", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_ashr`
This operation performs arithmetic shift right. The semantics are
    equivalent to the `bvashr` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_ashr(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.ashr", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_add`
This operation performs addition. The semantics are
    equivalent to the `bvadd` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_add(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.add", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_and`
This operation performs bitwise AND. The semantics are
    equivalent to the `bvand` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_and(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.and", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_cmp`

This operation compares bit-vector values, interpreting them as signed or
unsigned values depending on the predicate. The semantics are equivalent to
the `bvslt`, `bvsle`, `bvsgt`, `bvsge`, `bvult`, `bvule`, `bvugt`, or
`bvuge` operator defined in the SMT-LIB 2.7 standard depending on the
specified predicate. More precisely in the
[theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
describing closed quantifier-free formulas over the theory of fixed-size
bit-vectors.
"""
function bv_cmp(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, pred, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("pred", pred), ]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.cmp", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_constant`

This operation produces an SSA value equal to the bit-vector constant
specified by the \'value\' attribute.
Refer to the `BitVectorAttr` documentation for more information about
the semantics of bit-vector constants, their format, and associated sort.
The result type always matches the attribute\'s type.

Examples:
```mlir
%c92_bv8 = smt.bv.constant #smt.bv<92> : !smt.bv<8>
%c5_bv4 = smt.bv.constant #smt.bv<5> : !smt.bv<4>
```
"""
function bv_constant(; result=nothing::Union{Nothing, IR.Type}, value, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("value", value), ]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.constant", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_lshr`
This operation performs logical shift right. The semantics are
    equivalent to the `bvlshr` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_lshr(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.lshr", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_mul`
This operation performs multiplication. The semantics are
    equivalent to the `bvmul` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_mul(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.mul", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_neg`
This operation performs two\'s complement unary minus. The semantics are
    equivalent to the `bvneg` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_neg(input::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.neg", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_not`
This operation performs bitwise negation. The semantics are
    equivalent to the `bvnot` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_not(input::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.not", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_or`
This operation performs bitwise OR. The semantics are
    equivalent to the `bvor` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_or(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.or", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_sdiv`
This operation performs two\'s complement signed division. The semantics are
    equivalent to the `bvsdiv` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_sdiv(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.sdiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_smod`
This operation performs two\'s complement signed remainder (sign follows divisor). The semantics are
    equivalent to the `bvsmod` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_smod(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.smod", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_srem`
This operation performs two\'s complement signed remainder (sign follows dividend). The semantics are
    equivalent to the `bvsrem` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_srem(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.srem", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_shl`
This operation performs shift left. The semantics are
    equivalent to the `bvshl` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_shl(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.shl", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_udiv`
This operation performs unsigned division (rounded towards zero). The semantics are
    equivalent to the `bvudiv` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_udiv(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.udiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_urem`
This operation performs unsigned remainder. The semantics are
    equivalent to the `bvurem` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_urem(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.urem", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bv_xor`
This operation performs bitwise exclusive OR. The semantics are
    equivalent to the `bvxor` operator defined in the SMT-LIB 2.7
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
"""
function bv_xor(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.xor", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`constant`

Produces the constant expressions \'true\' and \'false\' as described in the
[Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2) of the SMT-LIB
Standard 2.7.
"""
function constant(; result=nothing::Union{Nothing, IR.Type}, value, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("value", value), ]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.constant", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`check`

This operation checks if all the assertions in the solver defined by the
nearest ancestor operation of type `smt.solver` are consistent. The outcome
an be \'satisfiable\', \'unknown\', or \'unsatisfiable\' and the corresponding
region will be executed. It is the corresponding construct to the
`check-sat` in SMT-LIB.

# Example
```mlir
%0 = smt.check sat {
  %c1_i32 = arith.constant 1 : i32
  smt.yield %c1_i32 : i32
} unknown {
  %c0_i32 = arith.constant 0 : i32
  smt.yield %c0_i32 : i32
} unsat {
  %c-1_i32 = arith.constant -1 : i32
  smt.yield %c-1_i32 : i32
} -> i32
```
"""
function check(; results::Vector{IR.Type}, satRegion::Region, unknownRegion::Region, unsatRegion::Region, location=Location())
    op_ty_results = IR.Type[results..., ]
    operands = Value[]
    owned_regions = Region[satRegion, unknownRegion, unsatRegion, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "smt.check", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`bv_concat`

This operation concatenates bit-vector values with semantics equivalent to
the `concat` operator defined in the SMT-LIB 2.7 standard. More precisely in
the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
describing closed quantifier-free formulas over the theory of fixed-size
bit-vectors.

Note that the following equivalences hold:
* `smt.bv.concat %a, %b : !smt.bv<4>, !smt.bv<4>` is equivalent to
  `(concat a b)` in SMT-LIB
* `(= (concat #xf #x0) #xf0)`
"""
function bv_concat(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.bv.concat", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`declare_fun`

This operation declares a symbolic value just as the `declare-const` and
`declare-fun` statements in SMT-LIB 2.7. The result type determines the SMT
sort of the symbolic value. The returned value can then be used to refer to
the symbolic value instead of using the identifier like in SMT-LIB.

The optionally provided string will be used as a prefix for the newly
generated identifier (useful for easier readability when exporting to
SMT-LIB). Each `declare` will always provide a unique new symbolic value
even if the identifier strings are the same.

Note that there does not exist a separate operation equivalent to
SMT-LIBs `define-fun` since
```
(define-fun f (a Int) Int (-a))
```
is only syntactic sugar for
```
%f = smt.declare_fun : !smt.func<(!smt.int) !smt.int>
%0 = smt.forall {
^bb0(%arg0: !smt.int):
  %1 = smt.apply_func %f(%arg0) : !smt.func<(!smt.int) !smt.int>
  %2 = smt.int.neg %arg0
  %3 = smt.eq %1, %2 : !smt.int
  smt.yield %3 : !smt.bool
}
smt.assert %0
```

Note that this operation cannot be marked as Pure since two operations (even
with the same identifier string) could then be CSEd, leading to incorrect
behavior.
"""
function declare_fun(; result::IR.Type, namePrefix=nothing, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(namePrefix) && push!(attributes, NamedAttribute("namePrefix", namePrefix))
    
    IR.create_operation(
        "smt.declare_fun", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`distinct`

This operation compares the operands and returns true iff all operands are
not identical to any of the other operands. The semantics are equivalent to
the `distinct` operator defined in the SMT-LIB Standard 2.7 in the
[Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2).

Any SMT sort/type is allowed for the operands and it supports a variadic
number of operands, but requires at least two. This is because the
`distinct` operator is annotated with `:pairwise` which means that
`distinct a b c d` is equivalent to
```
and (distinct a b) (distinct a c) (distinct a d)
    (distinct b c) (distinct b d)
    (distinct c d)
```
where `and` is annotated `:left-assoc`, i.e., it can be further rewritten to
```
(and (and (and (and (and (distinct a b)
                         (distinct a c))
                    (distinct a d))
               (distinct b c))
          (distinct b d))
     (distinct c d)
```
"""
function distinct(inputs::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.distinct", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`eq`

This operation compares the operands and returns true iff all operands are
identical. The semantics are equivalent to the `=` operator defined in the
SMT-LIB Standard 2.7 in the
[Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2).

Any SMT sort/type is allowed for the operands and it supports a variadic
number of operands, but requires at least two. This is because the `=`
operator is annotated with `:chainable` which means that `= a b c d` is
equivalent to `and (= a b) (= b c) (= c d)` where `and` is annotated
`:left-assoc`, i.e., it can be further rewritten to
`and (and (= a b) (= b c)) (= c d)`.
"""
function eq(inputs::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.eq", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`exists`

This operation represents the exists quantifier as described in the
[SMT-LIB 2.7 standard](https://smt-lib.org/papers/smt-lib-reference-v2.7-r2025-02-05.pdf).
It is part of the language itself rather than a theory or logic.

The operation specifies the name prefixes (as an optional attribute) and
types (as the types of the block arguments of the regions) of bound
variables that may be used in the \'body\' of the operation. If a \'patterns\'
region is specified, the block arguments must match the ones of the \'body\'
region and (other than there) must be used at least once in the \'patterns\'
region. It may also not contain any operations that bind variables, such as
quantifiers. While the \'body\' region must always yield exactly one
`!smt.bool`-typed value, the \'patterns\' region can yield an arbitrary number
(but at least one) of SMT values.

The bound variables can be any SMT type except of functions, since SMT only
supports first-order logic.

The \'no_patterns\' attribute is only allowed when no \'patterns\' region is
specified and forbids the solver to generate and use patterns for this
quantifier.

The \'weight\' attribute indicates the importance of this quantifier being
instantiated compared to other quantifiers that may be present. The default
value is zero.

Both the \'no_patterns\' and \'weight\' attributes are annotations to the
quantifiers body term. Annotations and attributes are described in the
standard in sections 3.4, and 3.6 (specifically 3.6.5). SMT-LIB allows
adding custom attributes to provide solvers with additional metadata, e.g.,
hints such as above mentioned attributes. They are not part of the standard
themselves, but supported by common SMT solvers (e.g., Z3).
"""
function exists(; result::IR.Type, weight=nothing, noPattern=nothing, boundVarNames=nothing, body::Region, patterns::Vector{Region}, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[]
    owned_regions = Region[body, patterns..., ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(weight) && push!(attributes, NamedAttribute("weight", weight))
    !isnothing(noPattern) && push!(attributes, NamedAttribute("noPattern", noPattern))
    !isnothing(boundVarNames) && push!(attributes, NamedAttribute("boundVarNames", boundVarNames))
    
    IR.create_operation(
        "smt.exists", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`bv_extract`

This operation extracts the range of bits starting at the \'lowBit\' index
(inclusive) up to the \'lowBit\' + result-width index (exclusive). The
semantics are equivalent to the `extract` operator defined in the SMT-LIB
2.7 standard. More precisely in the
[theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
describing closed quantifier-free formulas over the theory of fixed-size
bit-vectors.

Note that `smt.bv.extract %bv from 2 : (!smt.bv<32>) -> !smt.bv<16>` is
equivalent to `((_ extract 17 2) bv)`, i.e., the SMT-LIB operator takes the
low and high indices where both are inclusive. The following equivalence
holds: `(= ((_ extract 3 0) #x0f) #xf)`
"""
function bv_extract(input::Value; result::IR.Type, lowBit, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("lowBit", lowBit), ]
    
    IR.create_operation(
        "smt.bv.extract", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`forall`

This operation represents the forall quantifier as described in the
[SMT-LIB 2.7 standard](https://smt-lib.org/papers/smt-lib-reference-v2.7-r2025-02-05.pdf).
It is part of the language itself rather than a theory or logic.

The operation specifies the name prefixes (as an optional attribute) and
types (as the types of the block arguments of the regions) of bound
variables that may be used in the \'body\' of the operation. If a \'patterns\'
region is specified, the block arguments must match the ones of the \'body\'
region and (other than there) must be used at least once in the \'patterns\'
region. It may also not contain any operations that bind variables, such as
quantifiers. While the \'body\' region must always yield exactly one
`!smt.bool`-typed value, the \'patterns\' region can yield an arbitrary number
(but at least one) of SMT values.

The bound variables can be any SMT type except of functions, since SMT only
supports first-order logic.

The \'no_patterns\' attribute is only allowed when no \'patterns\' region is
specified and forbids the solver to generate and use patterns for this
quantifier.

The \'weight\' attribute indicates the importance of this quantifier being
instantiated compared to other quantifiers that may be present. The default
value is zero.

Both the \'no_patterns\' and \'weight\' attributes are annotations to the
quantifiers body term. Annotations and attributes are described in the
standard in sections 3.4, and 3.6 (specifically 3.6.5). SMT-LIB allows
adding custom attributes to provide solvers with additional metadata, e.g.,
hints such as above mentioned attributes. They are not part of the standard
themselves, but supported by common SMT solvers (e.g., Z3).
"""
function forall(; result::IR.Type, weight=nothing, noPattern=nothing, boundVarNames=nothing, body::Region, patterns::Vector{Region}, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[]
    owned_regions = Region[body, patterns..., ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(weight) && push!(attributes, NamedAttribute("weight", weight))
    !isnothing(noPattern) && push!(attributes, NamedAttribute("noPattern", noPattern))
    !isnothing(boundVarNames) && push!(attributes, NamedAttribute("boundVarNames", boundVarNames))
    
    IR.create_operation(
        "smt.forall", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`implies`

This operation performs a boolean implication. The semantics are equivalent
to the \'=>\' operator in the
[Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2) of the SMT-LIB
Standard 2.7.
"""
function implies(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.implies", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`int2bv`

Designed to lower directly to an operation of the same name in Z3. The Z3
C API describes the semantics as follows:
Create an n bit bit-vector from the integer argument t1.
The resulting bit-vector has n bits, where the i\'th bit (counting from 0
to n-1) is 1 if (t1 div 2^i) mod 2 is 1.
The node t1 must have integer sort.
"""
function int2bv(input::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "smt.int2bv", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`int_abs`

This operation represents the absolute value function for the `Int` sort.
The semantics are equivalent to the `abs` operator as described in the
[SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
SMT-LIB 2.7 standard.
"""
function int_abs(input::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.int.abs", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`int_add`

This operation represents (infinite-precision) integer addition.
The semantics are equivalent to the corresponding operator described in
the [SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
SMT-LIB 2.7 standard.
"""
function int_add(inputs::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.int.add", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`int_cmp`

This operation represents the comparison of (infinite-precision) integers.
The semantics are equivalent to the `<= (le)`, `< (lt)`, `>= (ge)`, or
`> (gt)` operator depending on the predicate (indicated in parentheses) as
described in the
[SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
SMT-LIB 2.7 standard.
"""
function int_cmp(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, pred, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("pred", pred), ]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.int.cmp", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`int_constant`

This operation represents (infinite-precision) integer literals of the `Int`
sort. The set of values for the sort `Int` consists of all numerals and
all terms of the form `-n`where n is a numeral other than 0. For more
information refer to the 
[SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
SMT-LIB 2.7 standard.
"""
function int_constant(; result=nothing::Union{Nothing, IR.Type}, value, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("value", value), ]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.int.constant", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`int_div`

This operation represents (infinite-precision) integer division.
The semantics are equivalent to the corresponding operator described in
the [SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
SMT-LIB 2.7 standard.
"""
function int_div(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.int.div", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`int_mod`

This operation represents (infinite-precision) integer remainder.
The semantics are equivalent to the corresponding operator described in
the [SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
SMT-LIB 2.7 standard.
"""
function int_mod(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.int.mod", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`int_mul`

This operation represents (infinite-precision) integer multiplication.
The semantics are equivalent to the corresponding operator described in
the [SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
SMT-LIB 2.7 standard.
"""
function int_mul(inputs::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.int.mul", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`int_sub`

This operation represents (infinite-precision) integer subtraction.
The semantics are equivalent to the corresponding operator described in
the [SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
SMT-LIB 2.7 standard.
"""
function int_sub(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.int.sub", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`ite`

This operation returns its second operand or its third operand depending on
whether its first operand is true or not. The semantics are equivalent to
the `ite` operator defined in the
[Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2) of the SMT-LIB
2.7 standard.
"""
function ite(cond::Value, thenValue::Value, elseValue::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[cond, thenValue, elseValue, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.ite", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`not`

This operation performs a boolean negation. The semantics are equivalent to
the \'not\' operator in the
[Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2) of the SMT-LIB
Standard 2.7.
"""
function not(input::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.not", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`or`
This operation performs a boolean disjunction.
    The semantics are equivalent to the \'or\' operator in the
    [Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2).
    of the SMT-LIB Standard 2.7.

    It supports a variadic number of operands, but requires at least two.
    This is because the operator is annotated with the `:left-assoc` attribute
    which means that `op a b c` is equivalent to `(op (op a b) c)`.
"""
function or(inputs::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.or", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function pop(; count, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("count", count), ]
    
    IR.create_operation(
        "smt.pop", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function push(; count, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("count", count), ]
    
    IR.create_operation(
        "smt.push", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`bv_repeat`

This operation is a shorthand for repeated concatenation of the same
bit-vector value, i.e.,
```mlir
smt.bv.repeat 5 times %a : !smt.bv<4>
// is the same as
%0 = smt.bv.repeat 4 times %a : !smt.bv<4>
smt.bv.concat %a, %0 : !smt.bv<4>, !smt.bv<16>
// or also 
%0 = smt.bv.repeat 4 times %a : !smt.bv<4>
smt.bv.concat %0, %a : !smt.bv<16>, !smt.bv<4>
```

The semantics are equivalent to the `repeat` operator defined in the SMT-LIB
2.7 standard. More precisely in the
[theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
describing closed quantifier-free formulas over the theory of fixed-size
bit-vectors.
"""
function bv_repeat(input::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "smt.bv.repeat", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function reset(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "smt.reset", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function set_logic(; logic, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("logic", logic), ]
    
    IR.create_operation(
        "smt.set_logic", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`solver`

This operation defines an SMT context with a solver instance. SMT operations
are only valid when being executed between the start and end of the region
of this operation. Any invocation outside is undefined. However, they do not
have to be direct children of this operation. For example, it is allowed to
have SMT operations in a `func.func` which is only called from within this
region. No SMT value may enter or exit the lifespan of this region (such
that no value created from another SMT context can be used in this scope and
the solver can deallocate all state required to keep track of SMT values at
the end).

As a result, the region is comparable to an entire SMT-LIB script, but
allows for concrete operations and control-flow. Concrete values may be
passed in and returned to influence the computations after the `smt.solver`
operation.

# Example
```mlir
%0:2 = smt.solver (%in) {smt.some_attr} : (i8) -> (i8, i32) {
^bb0(%arg0: i8):
  %c = smt.declare_fun \"c\" : !smt.bool
  smt.assert %c
  %1 = smt.check sat {
    %c1_i32 = arith.constant 1 : i32
    smt.yield %c1_i32 : i32
  } unknown {
    %c0_i32 = arith.constant 0 : i32
    smt.yield %c0_i32 : i32
  } unsat {
    %c-1_i32 = arith.constant -1 : i32
    smt.yield %c-1_i32 : i32
  } -> i32
  smt.yield %arg0, %1 : i8, i32
}
```
"""
function solver(inputs::Vector{Value}; results::Vector{IR.Type}, bodyRegion::Region, location=Location())
    op_ty_results = IR.Type[results..., ]
    operands = Value[inputs..., ]
    owned_regions = Region[bodyRegion, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "smt.solver", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`xor`
This operation performs a boolean exclusive OR.
    The semantics are equivalent to the \'xor\' operator in the
    [Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2).
    of the SMT-LIB Standard 2.7.

    It supports a variadic number of operands, but requires at least two.
    This is because the operator is annotated with the `:left-assoc` attribute
    which means that `op a b c` is equivalent to `(op (op a b) c)`.
"""
function xor(inputs::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "smt.xor", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function yield(values::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[values..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "smt.yield", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end
end # smt

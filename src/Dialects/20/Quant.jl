module quant

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes
import ...API


"""
`dcast`

Convert an input quantized value into its expressed floating-point value.
The dequantization process consists of the following steps:

```
def dequantize(quantizedValue: quantizedType) -> expressedType:
    storedValue = reinterpretCast(quantizedValue, storageType)
    storedValueFloat = convertIntToFloat(storedValue, expressedType)
    zeroPointFloat = convertIntToFloat(zeroPoint, expressedType)
    expressedValue = (storedValueFloat - zeroPointFloat) * scale
    return expressedValue
```

Here, `storageType`, `expressedType`, `scale`, and `zeroPoint` are obtained
from the corresponding parameters encoded in `quantizedType`. For
per-channel quantization, the appropriate `scale` and `zeroPoint` values
are used for each tensor element computation according to the channel the
element belongs to.

The numerical results produced by the algorithm above may vary depending on
the rounding methods used by `convertIntToFloat()`, subtraction (`-`), and
multiplication (`*`). This operation does not define specific rounding
methods; instead, it is the responsibility of a transform pipeline to
determine which rounding method to apply when this operation is broken down
into lower-level dialects.

The operation must satisfy the following syntactic constraints:

- Operand `input` must be a scalar or tensor of type `!quant.uniform`.

- The result type must be a floating-point scalar or tensor.

- The `expressedType` parameter of the `!quant.uniform` type of the input
  must match the floating-point type of the result.

- The operand and result types must be both scalars or both tensors. If
  tensors, they must be both ranked or both unranked. If ranked, both must
  have the same shape, including matching static and dynamic dimensions.

- If the operand uses per-channel quantization, its `!quant.uniform` type
  must adhere to the [Per-axis quantization
  integrity](#per-axis-quantization-integrity) guidelines.

Examples:

```
// Dequantize a scalar quantized value
%result = quant.dcast %input : !quant.uniform<i8:f32, 2.0> to f32

// Dequantize a dynamically shaped tensor of quantized values
%result = quant.dcast %input : tensor<?x!quant.uniform<i8:f32, 2.0>> to tensor<?xf32>

// Dequantize an unranked tensor using per-axis quantization information
%result = quant.dcast %input : tensor<*x!quant.uniform<i8:f32:1, {2.0, 3.0}>> to tensor<*xf32>
```
"""
function dcast(input::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "quant.dcast", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`qcast`

Convert a floating-point value to a quantized type. The quantization
process consists of the following steps:

```
def quantize(expressedValue: expressedType) -> quantizedType:
    zeroPointFloat = convertIntToFloat(zeroPoint, expressedType)
    scaledValue = expressedValue / scale
    storedValueFloat = scaledValue + zeroPointFloat
    storedValue = convertFloatToInt(storedValueFloat, storageType)
    storedValueClamped = clamp(storedValue, storageMin, storageMax)
    quantizedValue = reinterpretCast(storedValueClamped, quantizedType)
    return quantizedValue
```

Here, `storageType`, `storageMin`, `storageMax`, `expressedType`, `scale`,
and `zeroPoint` are obtained from the corresponding parameters encoded in
`quantizedType`. For per-channel quantization, the appropriate `scale` and
`zeroPoint` values are used for each tensor element computation according
to the channel the element belongs to.

The numerical results produced by the algorithm above may vary depending on
the rounding methods used by `convertIntToFloat()`, `convertFloatToInt()`,
`clamp()`, division (`/`), and addition (`+`). This operation does not
define specific rounding methods; instead, it is the responsibility of a
transform pipeline to determine which rounding method to apply when this
operation is broken down into lower-level dialects.

The operation must satisfy the following syntactic constraints:

- Operand `input` must be a floating-point scalar or tensor.

- The result type must be a scalar or tensor of type `!quant.uniform`.

- The `expressedType` parameter in the `!quant.uniform` type of the result
  must match the floating-point type of the input.

- The operand and result types must be both scalars or both tensors. If
  tensors, they must be both ranked or both unranked. If ranked, both must
  have the same shape, including matching static and dynamic dimensions.

- If the result uses per-channel quantization, its `!quant.uniform` type
  must adhere to the [Per-axis quantization
  integrity](#per-axis-quantization-integrity) guidelines.

Examples:

```
// Quantize a scalar floating-point value
%result = quant.qcast %input : f32 to !quant.uniform<i8:f32, 2.0>

// Quantize a dynamically shaped tensor of quantized values
%result = quant.qcast %input : tensor<?xf32> to tensor<?x!quant.uniform<i8:f32, 2.0>>

// Quantize an unranked tensor using per-axis quantization information
%result = quant.qcast %input : tensor<*xf32> to tensor<*x!quant.uniform<i8:f32:1, {2.0, 3.0}>>
```
"""
function qcast(input::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "quant.qcast", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`scast`

Convert a value from a quantized type to the corresponding signless integer
storage type, or vice versa. This conversion simply involves a
reinterpretation of the input bits and does not involve any data
manipulation.

The following syntactic restrictions must be met:

- Operand `input` must be a scalar or tensor of a signless integer or
  `!quant.uniform` type.

- The result must be a scalar or tensor of a signless integer or
  `!quant.uniform` type.

- If the operand is a scalar or tensor of type integer, the result must be
  a scalar or tensor of type `!quant.uniform`, and vice versa.

- The operand and result must be both scalars or both tensors. If tensors,
  they must be both ranked or both unranked. If ranked, both must have the
  same shape, including matching static and dynamic dimensions.

- The width of the `storageType` parameter of the quantized type of the
  operand or result must match the width of the signless integer type of
  the operand or result.

- If the operand or result uses per-channel quantization, its
  `!quant.uniform` type must adhere to the [Per-axis quantization
  integrity](#per-axis-quantization-integrity) guidelines.

Examples:

```
// Cast a scalar quantized value into its storage type
%result = quant.scast %input : !quant.uniform<i8:f32, 2.0> to i8

// Cast a dynamically shaped tensor of quantized values into their storage type
%result = quant.scast %input : tensor<?x!quant.uniform<i8:f32, 2.0>> to tensor<?xi8>

// Cast an unranked tensor of signless integers into a quantized type using
// per-channel quantization
%result = quant.scast %input : tensor<*xi8> to tensor<*x!quant.uniform<i8:f32:1, {2.0, 3.0}>>
```
"""
function scast(input::Value; result::IR.Type, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "quant.scast", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

end # quant

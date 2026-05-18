module tosa

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes
import ...API


"""
`abs`

Elementwise absolute value operation.

# Example

```mlir
%output = tosa.abs(%input1) : (tensor<21x3xf32>) -> tensor<21x3xf32>
```
"""
function abs(input1::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.abs", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`add`

Elementwise addition of input1 and input2. Axis of size 1 will be broadcast,
as necessary. Rank of input tensors must match.

# Example

```mlir
// Elementwise addition.
%out = tosa.add %input1, %input2 : tensor<12x6xf32>, tensor<12x6xf32> -> tensor<12x6xf32>

// Elementwise addition with broadcasting.
%out = tosa.add %input1, %input2 : tensor<12x6xsi32>, tensor<1x1xsi32> -> tensor<12x6xsi32>
```
"""
function add(input1::Value, input2::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.add", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`apply_scale`

Applies rescaling for fixed point values. This behavior is replicated in
multiple quantized operations (mul, convolution, rescale, matmul, pooling).

The commonplace implementation is to use i64 operations to avoid integer
overflow with target specific implementations can use native operations to
avoid wider than necessary types.
"""
function apply_scale(value::Value, multiplier::Value, shift::Value; output::IR.Type, rounding_mode, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[value, multiplier, shift, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("rounding_mode", rounding_mode), ]
    
    create_operation(
        "tosa.apply_scale", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`argmax`

This returns the index with the largest value across the given axis of the
input tensor. If multiple locations have equal values, returns the first
match along the search axis.
"""
function argmax(input::Value; output::IR.Type, axis, nan_mode=nothing, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("axis", axis), ]
    !isnothing(nan_mode) && push!(attributes, NamedAttribute("nan_mode", nan_mode))
    
    create_operation(
        "tosa.argmax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`arithmetic_right_shift`

Elementwise arithmetic right shift of input1 by the amount specified in
input2. Axis of size 1 will be broadcast, as necessary. Rank of input tensors
must match.
"""
function arithmetic_right_shift(input1::Value, input2::Value; output::IR.Type, round, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("round", round), ]
    
    create_operation(
        "tosa.arithmetic_right_shift", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`avg_pool2d`

This performs an average pooling over the given input tensor. A sliding
window of size given by <kernel size> is passed over the input tensor, with
the mean value being placed in the output tensor. When calculating the
average, only the number of valid input tensor values, but not padding, are
used to calculate the divisor.
"""
function avg_pool2d(input::Value, input_zp::Value, output_zp::Value; output::IR.Type, kernel, stride, pad, acc_type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, input_zp, output_zp, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("kernel", kernel), NamedAttribute("stride", stride), NamedAttribute("pad", pad), NamedAttribute("acc_type", acc_type), ]
    
    create_operation(
        "tosa.avg_pool2d", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`bitwise_and`

Elementwise bitwise AND of input1 and input2. Axis of size 1
will be broadcast as necessary. Rank of input tensors must match.
"""
function bitwise_and(input1::Value, input2::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.bitwise_and", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`bitwise_not`

Elementwise bitwise NOT of input tensor.
"""
function bitwise_not(input1::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.bitwise_not", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`bitwise_or`

Elementwise bitwise OR of input1 and input2. Axis of size 1 will be
broadcast as necessary. Rank of input tensors must match.
"""
function bitwise_or(input1::Value, input2::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.bitwise_or", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`bitwise_xor`

Elementwise bitwise XOR of input1 and input2. Axis of size 1 will be
broadcast as necessary. Rank of input tensors must match.
"""
function bitwise_xor(input1::Value, input2::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.bitwise_xor", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cast`

Casts a tensor from one data type to another.
* This table is showing the supported conversions from the TOSA Specification.
* The MLIR dialect here can be used to represent other conversions.

| Mode                     | Input   | Output  |
|--------------------------|---------|---------|
| fp16 to fp32             | float16 | float32 |
| fp16 to int 16           | float16 | int16   |
| fp16 to int 32           | float16 | int32   |
| fp16 to int 8            | float16 | int8    |
| fp32 to fp16             | float32 | float16 |
| fp32 to int 16           | float32 | int16   |
| fp32 to int 32           | float32 | int32   |
| fp32 to int 8            | float32 | int8    |
| int 16 to fp16           | int16   | float16 |
| int 16 to fp32           | int16   | float32 |
| int 32 to fp16           | int32   | float16 |
| int 32 to fp32           | int32   | float32 |
| int 8 to fp16            | int8    | float16 |
| int 8 to fp32            | int8    | float32 |
| bool to int 16           | Boolean | int16   |
| bool to int 32           | Boolean | int32   |
| bool to int 8            | Boolean | int8    |
| int 16 to bool           | int16   | Boolean |
| int 16 to int 32         | int16   | int32   |
| int 16 to int 8          | int16   | int8    |
| int 32 to bool           | int32   | Boolean |
| int 32 to int 16         | int32   | int16   |
| int 32 to int 8          | int32   | int8    |
| int 8 to bool            | int8    | Boolean |
| int 8 to int 16          | int8    | int16   |
| int 8 to int 32          | int8    | int32   |
| bf16 to fp32             | bf16    | float32 |
| bf16 to int 16           | bf16    | int16   |
| bf16 to int 32           | bf16    | int32   |
| bf16 to int 8            | bf16    | int8    |
| fp32 to bf16             | float32 | bf16    |
| int 16 to bf16           | int16   | bf16    |
| int 32 to bf16           | int32   | bf16    |
| int 8 to bf16            | int8    | bf16    |
| bf16 to fp8e4m3          | bf16    | fp8e4m3 |
| fp8e4m3 to bf16          | fp8e4m3 | bf16    |
| bf16 to fp8e5m2          | bf16    | fp8e5m2 |
| fp8e5m2 to bf16          | fp8e5m2 | bf16    |
| fp16 to fp8e4m3          | float16 | fp8e4m3 |
| fp32 to fp8e4m3          | float32 | fp8e4m3 |
| fp8e4m3 to fp16          | fp8e4m3 | float16 |
| fp8e4m3 to fp32          | fp8e4m3 | float32 |
| fp16 to fp8e5m2          | float16 | fp8e5m2 |
| fp32 to fp8e5m2          | float32 | fp8e5m2 |
| fp8e5m2 to fp16          | fp8e5m2 | float16 |
| fp8e5m2 to fp32          | fp8e5m2 | float32 |
"""
function cast(input::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.cast", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`ceil`

Elementwise ceiling operation.
"""
function ceil(input1::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.ceil", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`clamp`

Clamp to an arbitrary minimum and maximum value.
Maximum and minimum values are specified as values in the range of the
input type.
No zero point subtraction is done to the values, thus to clamp to the zero
point value, the zero point itself should be supplied as the minimum value.
"""
function clamp(input::Value; output::IR.Type, min_val, max_val, nan_mode=nothing, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("min_val", min_val), NamedAttribute("max_val", max_val), ]
    !isnothing(nan_mode) && push!(attributes, NamedAttribute("nan_mode", nan_mode))
    
    create_operation(
        "tosa.clamp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`clz`

Elementwise count leading zeros operation.
"""
function clz(input1::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.clz", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`concat`

Concatenate a list of tensors along a given axis.
No data conversion happens during a concat operation.
"""
function concat(input1::Vector{Value}; output=nothing::Union{Nothing, IR.Type}, axis, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input1..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("axis", axis), ]
    !isnothing(output) && push!(op_ty_results, output)
    
    create_operation(
        "tosa.concat", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`const_`

A node containing constant data for use as the input to an operation. May
hold data in any of the supported data formats.

# Example

```mlir
// Generic form
%out = \"tosa.const\"() {values = dense<0> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
```
"""
function const_(; output=nothing::Union{Nothing, IR.Type}, values, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("values", values), ]
    !isnothing(output) && push!(op_ty_results, output)
    
    create_operation(
        "tosa.const", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`const_shape`

A node containing a constant shape.

# Example

```mlir
// Generic form
%out = \"tosa.const_shape\"() {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
```
"""
function const_shape(; output::IR.Type, values, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("values", values), ]
    
    create_operation(
        "tosa.const_shape", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`conv2d`

Performs a 2D convolution over the given tensor input, using the weight
tensor. Implementations may choose to skip calculation of multiplies in
the padding area.
"""
function conv2d(input::Value, weight::Value, bias::Value, input_zp::Value, weight_zp::Value; output::IR.Type, pad, stride, dilation, acc_type, local_bound=nothing, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, weight, bias, input_zp, weight_zp, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("pad", pad), NamedAttribute("stride", stride), NamedAttribute("dilation", dilation), NamedAttribute("acc_type", acc_type), ]
    !isnothing(local_bound) && push!(attributes, NamedAttribute("local_bound", local_bound))
    
    create_operation(
        "tosa.conv2d", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`conv3d`

Performs a 3D convolution over the given input tensor. Implementations
may choose to skip calculation of multiplies in the padding area.
"""
function conv3d(input::Value, weight::Value, bias::Value, input_zp::Value, weight_zp::Value; output::IR.Type, pad, stride, dilation, acc_type, local_bound=nothing, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, weight, bias, input_zp, weight_zp, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("pad", pad), NamedAttribute("stride", stride), NamedAttribute("dilation", dilation), NamedAttribute("acc_type", acc_type), ]
    !isnothing(local_bound) && push!(attributes, NamedAttribute("local_bound", local_bound))
    
    create_operation(
        "tosa.conv3d", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cos`

Elementwise cosine operation for values given in radians.
"""
function cos(input1::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.cos", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`custom`

Hardware implementing TOSA may choose to add additional custom operators
that are not expressed in the existing TOSA operations. These operators are
not expected to be portable across TOSA implementations. The input and
output signatures must be expressed in the corresponding TOSA node.

`operator_name` is a string that tells the backend which custom operator is
being called.

`domain_name` is a string identifier which can help avoid name collisions on
the identifier field.

`implementation_attrs` is a string which is a backend and identifier specific
set of attributes to the custom operator.

`input_list` is the set of tensor inputs to the custom operator.

`output_list` is the list of tensors returned by the operator. The number of operators
is backend specific.

# Example

```mlir
%out = tosa.custom %in {domain_name = \"tosa_mlir_test\", operator_name =
       \"custom_test\", implementation_attrs = \"\"}: (tensor<10xi32>) ->
       (tensor<10xi32>)
```
"""
function custom(input_list::Vector{Value}; output_list::Vector{IR.Type}, operator_name, domain_name, implementation_attrs, location=Location())
    op_ty_results = IR.Type[output_list..., ]
    operands = Value[input_list..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("operator_name", operator_name), NamedAttribute("domain_name", domain_name), NamedAttribute("implementation_attrs", implementation_attrs), ]
    
    create_operation(
        "tosa.custom", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`depthwise_conv2d`

Performs 2D convolutions separately over each channel of the given tensor
input, using the weight tensor. Implementations may choose to skip
calculation of multiplies in the padding area.
"""
function depthwise_conv2d(input::Value, weight::Value, bias::Value, input_zp::Value, weight_zp::Value; output::IR.Type, pad, stride, dilation, acc_type, local_bound=nothing, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, weight, bias, input_zp, weight_zp, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("pad", pad), NamedAttribute("stride", stride), NamedAttribute("dilation", dilation), NamedAttribute("acc_type", acc_type), ]
    !isnothing(local_bound) && push!(attributes, NamedAttribute("local_bound", local_bound))
    
    create_operation(
        "tosa.depthwise_conv2d", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`equal`

Elementwise comparison operation.
"""
function equal(input1::Value, input2::Value; output=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(output) && push!(op_ty_results, output)
    
    create_operation(
        "tosa.equal", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`erf`

Gauss error function: \$ erf(x) = \\frac{2}{\\sqrt{\\pi}} \\int_{0}^{x} e^{-t^2} dt \$
For quantized integer data types, the TABLE operator should be used instead
with the following definition. The ERF table has 513 entries each of
16-bit precision and covering the input range -4.0 to +4.0 in steps of 1/64.
"""
function erf(input::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.erf", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`exp`

Elementwise e to the x operation
"""
function exp(input1::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.exp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`fft2d`

Performs a batched complex 2D Fast Fourier Transform over the input. The
complex input values are constructed from the corresponding values in the
input_real and input_imag tensors. The resulting values in the output are
split into the output_real and output_imag tensors. No normalization is
applied on either the forward or inverse versions of the operation.

# Example

```mlir
 %output_real, %output_imag = tosa.fft2d %input_real, %input_imag : (tensor<8x9xf32>, tensor<8x9xf32>) -> (tensor<8x9xf32>, tensor<8x9xf32>)
```
"""
function fft2d(input_real::Value, input_imag::Value; output_real::IR.Type, output_imag::IR.Type, inverse, local_bound=nothing, location=Location())
    op_ty_results = IR.Type[output_real, output_imag, ]
    operands = Value[input_real, input_imag, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("inverse", inverse), ]
    !isnothing(local_bound) && push!(attributes, NamedAttribute("local_bound", local_bound))
    
    create_operation(
        "tosa.fft2d", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`floor`

Elementwise floor operation.
"""
function floor(input1::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.floor", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`gather`

Generate a tensor for which each element in the output is a subtensor of the
values tensor based on the indices. N is the number of batches, W the number
of indices in each batch, K the range of each index and C the number data
channels for each index.
"""
function gather(values::Value, indices::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[values, indices, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.gather", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`greater_equal`

Elementwise comparison operation.
"""
function greater_equal(input1::Value, input2::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.greater_equal", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`greater`

Elementwise greater than comparison operation.
"""
function greater(input1::Value, input2::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.greater", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`identity`

Returns a tensor with the same shape, type, and contents as the input.
"""
function identity(input1::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.identity", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cond_if`

Evaluates a Boolean condition and then takes one of two distinct execution
paths. This implements the semantic If-then-else structure.
"""
function cond_if(condition::Value, input_list::Vector{Value}; output_list::Vector{IR.Type}, then_graph::Region, else_graph::Region, location=Location())
    op_ty_results = IR.Type[output_list..., ]
    operands = Value[condition, input_list..., ]
    owned_regions = Region[then_graph, else_graph, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.cond_if", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intdiv`

Elementwise integer divide of input1 by input2. Axis of size 1 will be
broadcast as necessary. Rank of input tensors must match. The result of the
divide is truncated towards zero. Expected use is for operations on
non-scaled integers. Floating point divide should use RECIPROCAL and MUL.
Quantized integer divide should use TABLE (for 1/x) and MUL.
"""
function intdiv(input1::Value, input2::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.intdiv", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`log`

Elementwise natural logarithm operation
"""
function log(input1::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.log", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`logical_and`

Elementwise logical AND of input1 and input2. Axis of size 1 will be
broadcast, as necessary. Rank of input tensors must match.
"""
function logical_and(input1::Value, input2::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.logical_and", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`logical_left_shift`

Elementwise logical left-shift of input1 by the amount specified in input2.
Axis of size 1 will be broadcast, as necessary.
Rank of input tensors must match.
"""
function logical_left_shift(input1::Value, input2::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.logical_left_shift", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`logical_not`

Elementwise logical NOT of input.
"""
function logical_not(input1::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.logical_not", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`logical_or`

Elementwise logical OR of input1 and input2. Axis of size 1 will be
broadcast as necessary. Rank of input tensors must match.
"""
function logical_or(input1::Value, input2::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.logical_or", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`logical_right_shift`

Elementwise logical right shift of input1 by the amount specified in input2.
Axis of size 1 will be broadcast, as necessary. Rank of input tensors must
match.
"""
function logical_right_shift(input1::Value, input2::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.logical_right_shift", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`logical_xor`

Elementwise logical XOR of input1 and input2. Axis of size 1 will be
broadcast as necessary. Rank of input tensors must match.
"""
function logical_xor(input1::Value, input2::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.logical_xor", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`matmul`

Performs two dimensional matrix multiplications.
"""
function matmul(a::Value, b::Value, a_zp::Value, b_zp::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[a, b, a_zp, b_zp, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.matmul", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`max_pool2d`

This performs a max pooling over the given input tensor. A sliding window of
size given by <kernel size> is passed over the input tensor, with the
maximum value being placed in the
output tensor.
"""
function max_pool2d(input::Value; output::IR.Type, kernel, stride, pad, nan_mode=nothing, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("kernel", kernel), NamedAttribute("stride", stride), NamedAttribute("pad", pad), ]
    !isnothing(nan_mode) && push!(attributes, NamedAttribute("nan_mode", nan_mode))
    
    create_operation(
        "tosa.max_pool2d", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`maximum`

Elementwise max of input1 and input2. Axis of size 1 will be broadcast, as
necessary. Rank of input tensors must match.
"""
function maximum(input1::Value, input2::Value; output::IR.Type, nan_mode=nothing, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(nan_mode) && push!(attributes, NamedAttribute("nan_mode", nan_mode))
    
    create_operation(
        "tosa.maximum", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`minimum`

Elementwise minimum of input1 and input2. Axis of size 1
will be broadcast, as necessary. Rank of input tensors must match.
"""
function minimum(input1::Value, input2::Value; output::IR.Type, nan_mode=nothing, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(nan_mode) && push!(attributes, NamedAttribute("nan_mode", nan_mode))
    
    create_operation(
        "tosa.minimum", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mul`

Elementwise multiplication (Hadamard product) of input1 and input2.
Axis of size 1 will be broadcast, as necessary. Rank of input tensors must
match.
"""
function mul(input1::Value, input2::Value, shift::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, shift, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.mul", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`negate`

Elementwise negation operation.
"""
function negate(input1::Value, input1_zp::Value, output_zp::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input1_zp, output_zp, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.negate", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`pad`

Pads a tensor along the borders of each dimension with a supplied value.
Returns a new tensor with the padding included. The pad_const value includes
the zero point if the tensor uses a zero point.

# Example

```mlir
%pad_const = \"tosa.const\"() {values = dense<3.14> : tensor<1xf32>} : () -> tensor<1xf32>
%padding = tosa.const_shape {values = dense<[1, 2, 3, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
tosa.pad %arg0, %padding, %pad_const: (tensor<1x2xf32>, !tosa.shape<4>, tensor<1xf32>)  -> (tensor<4x9xf32>)
```

Example 2:

```mlir
%pad_const = \"tosa.const\"() {values = dense<3.14> : tensor<1xf32>} : () -> tensor<1xf32>
%padding = tosa.const_shape {values = dense<[-1, 2, 3, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
tosa.pad %arg0, %padding, %pad_const : (tensor<1x2xf32>, !tosa.shape<4>, tensor<1xf32>)  -> (tensor<?x9xf32>)
```
"""
function pad(input1::Value, padding::Value, pad_const::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, padding, pad_const, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.pad", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`pow`

Elementwise input1 value raised to the power of input2.
Axis of size 1 will be broadcast, as necessary. Rank of input tensors must
match.
"""
function pow(input1::Value, input2::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.pow", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`rfft2d`

Performs a batched 2D real-valued Fast Fourier Transform over the input where
the input tensor consists of real values producing complex valued output. The
complex output values will be split into the output_real and output_imag
tensor arguments. RFFT2D takes advantage of Hermitian symmetry to only
calculate the first half of the final output axis. Implementations may choose
to skip calculation of the imaginary values at (0,0), (0,W/2), (H/2,0), and
(H/2, W/2). If the calculation is skipped, the result at that location must be
zero.

# Example

```mlir
 %ouput_real, %output_imag = tosa.rfft2d %input_real : (tensor<8x16xf32>) -> (tensor<8x9xf32>, tensor<8x9xf32>)
```
"""
function rfft2d(input_real::Value; output_real::IR.Type, output_imag::IR.Type, local_bound=nothing, location=Location())
    op_ty_results = IR.Type[output_real, output_imag, ]
    operands = Value[input_real, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(local_bound) && push!(attributes, NamedAttribute("local_bound", local_bound))
    
    create_operation(
        "tosa.rfft2d", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`reciprocal`

Elementwise reciprocal operation. For integer operation, a TABLE should be
used with the appropriate ranges.
"""
function reciprocal(input1::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.reciprocal", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`reduce_all`

Reduce a tensor along the given axis with a logical AND operation.
"""
function reduce_all(input::Value; output=nothing::Union{Nothing, IR.Type}, axis, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("axis", axis), ]
    !isnothing(output) && push!(op_ty_results, output)
    
    create_operation(
        "tosa.reduce_all", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`reduce_any`

Reduce a tensor along the given axis with a logical OR operation.
"""
function reduce_any(input::Value; output=nothing::Union{Nothing, IR.Type}, axis, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("axis", axis), ]
    !isnothing(output) && push!(op_ty_results, output)
    
    create_operation(
        "tosa.reduce_any", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`reduce_max`

Reduce a tensor along the given axis with a maximum operation.
"""
function reduce_max(input::Value; output=nothing::Union{Nothing, IR.Type}, axis, nan_mode=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("axis", axis), ]
    !isnothing(output) && push!(op_ty_results, output)
    !isnothing(nan_mode) && push!(attributes, NamedAttribute("nan_mode", nan_mode))
    
    create_operation(
        "tosa.reduce_max", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`reduce_min`

Reduce a tensor along the given axis with a minimum operation.
"""
function reduce_min(input::Value; output=nothing::Union{Nothing, IR.Type}, axis, nan_mode=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("axis", axis), ]
    !isnothing(output) && push!(op_ty_results, output)
    !isnothing(nan_mode) && push!(attributes, NamedAttribute("nan_mode", nan_mode))
    
    create_operation(
        "tosa.reduce_min", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`reduce_product`

Reduce a tensor along the given axis by computing the product of the axis.
"""
function reduce_product(input::Value; output=nothing::Union{Nothing, IR.Type}, axis, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("axis", axis), ]
    !isnothing(output) && push!(op_ty_results, output)
    
    create_operation(
        "tosa.reduce_product", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`reduce_sum`

Reduce a tensor along the given axis by computing the sum of the axis.
"""
function reduce_sum(input::Value; output=nothing::Union{Nothing, IR.Type}, axis, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("axis", axis), ]
    !isnothing(output) && push!(op_ty_results, output)
    
    create_operation(
        "tosa.reduce_sum", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`rescale`

RESCALE is defined using an integer multiply, add, and shift.

Rescale supports two precisions of multiplier: 16-bit and 32-bit. The 32-bit multiplier
version supports two rounding modes to enable simpler lowering of existing frameworks
that use two stage rounding. All arithmetic is designed so that it does not overflow a
64-bit accumulator and that the result fits in 32 bits. In particular, a 48-bit value
cannot be scaled with the 32-bit multiplier because the accumulator would need to have
80 bits.

The shift and value range are limited to allow a variety of implementations. The limit
of 62 on shift allows the shift to be decomposed as two right shifts of 31.

Supported rescalings:
* This table is showing the supported conversions from the TOSA Specification.
* The MLIR dialect here can be used to represent other conversions.

| Mode                   | Input | Output | Unsigned input | Unsigned output |
|------------------------|-------|--------|----------------|-----------------|
| signed 16 to 16        | int16 | int16  |  false         |  false          |
| signed 16 to 32        | int16 | int32  |  false         |  false          |
| signed 16 to 8         | int16 | int8   |  false         |  false          |
| signed 32 to 16        | int32 | int16  |  false         |  false          |
| signed 32 to 32        | int32 | int32  |  false         |  false          |
| signed 32 to 8         | int32 | int8   |  false         |  false          |
| signed 8 to 16         | int8  | int16  |  false         |  false          |
| signed 8 to 32         | int8  | int32  |  false         |  false          |
| signed 8 to 8          | int8  | int8   |  false         |  false          |
| signed 48 to 16        | int48 | int16  |  false         |  false          |
| signed 48 to 32        | int48 | int32  |  false         |  false          |
| signed 48 to 8         | int48 | int8   |  false         |  false          |
| unsigned 8 to signed 8 | uint8 | int8   |  true          |  false          |
| signed 8 to unsigned 8 | int8  | uint8  |  false         |  true           |
"""
function rescale(input::Value, multiplier::Value, shift::Value, input_zp::Value, output_zp::Value; output::IR.Type, scale32, rounding_mode, per_channel, input_unsigned, output_unsigned, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, multiplier, shift, input_zp, output_zp, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("scale32", scale32), NamedAttribute("rounding_mode", rounding_mode), NamedAttribute("per_channel", per_channel), NamedAttribute("input_unsigned", input_unsigned), NamedAttribute("output_unsigned", output_unsigned), ]
    
    create_operation(
        "tosa.rescale", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`reshape`

Returns a tensor with the same type/values as the input, with a new shape
specified by the shape argument. Reshape may operate on tensors of any rank.
No data conversion happens during a reshape operation.
"""
function reshape(input1::Value, shape::Value; output=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input1, shape, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(output) && push!(op_ty_results, output)
    
    create_operation(
        "tosa.reshape", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`resize`

Resizes a tensor. Resize is only allowed in the H and W dimensions.

The height dimension is scaled by factor (scale_y_n/scale_y_d). The width
dimension is scaled by factor (scale_x_n/scale_x_d).

The NEAREST_NEIGHBOR mode returns the value of the input tensor closest to
the calculated sample position for both floating-point and integer data
formats.

Floating-point BILINEAR mode returns a bilinearly interpolated output value
based on the four closest input sample positions.

For integer BILINEAR interpolation mode, the output value must be scaled by
1/(scale_y_n * scale_x_n) in a following operation to complete the
interpolation (for example with a RESCALE operator).

The output dimensions can be derived from the input dimensions by inverting
the scale as described in the pseudocode. The [border_y, border_x] values
adjust the output size to allow fractional sampling beyond integer input
position (IH - 1,IW - 1).

The limit MAX_SCALE is applied to each scale ratio after reduction of the
ratio. Individual scale numerator and denominator values are allowed to be
larger than MAX_SCALE.
"""
function resize(input::Value, scale::Value, offset::Value, border::Value; output::IR.Type, mode, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, scale, offset, border, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mode", mode), ]
    
    create_operation(
        "tosa.resize", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`reverse`

Returns a tensor with the same type/values as the input, with the data
reversed along the given axis. No data conversion happens during a reverse
operation.
"""
function reverse(input1::Value; output::IR.Type, axis, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("axis", axis), ]
    
    create_operation(
        "tosa.reverse", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`rsqrt`

Elementwise reciprocal square root operation. For integer operation, a TABLE
should be used with the appropriate ranges.
"""
function rsqrt(input1::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.rsqrt", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`scatter`

The values_out tensor is set to the values_in tensor with data modified as
follows: data from the input tensor is inserted at the positions specified
by the indices tensor. N is the number of batches, W the number of indices
in each batch, K the range of each index and C the number data channels for
each index. It is not permitted to repeat the same output index within a
single SCATTER operation and so each output index occurs at most once. It
follows that K >= W. In use cases that require multiple updates to the same
output position, these must be decomposed into multiple SCATTER operations.
"""
function scatter(values_in::Value, indices::Value, input::Value; values_out::IR.Type, location=Location())
    op_ty_results = IR.Type[values_out, ]
    operands = Value[values_in, indices, input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.scatter", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`select`

Elementwise select of the output based on a condition.
"""
function select(input1::Value, input2::Value, input3::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, input3, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.select", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`sigmoid`

Applies the sigmoid logistic function to each element of the input tensor:
\$ sigmoid(x) = \\frac{1}{1 + e^{-x}} \$.

For quantized integer data types, the TABLE operator should be used instead.
Each implementation may choose an appropriate TABLE given the scale and zero
point of the input data. Eight or sixteen bit precision tables may be used
based on the input tensor to the sigmoid function.
"""
function sigmoid(input::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.sigmoid", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`sin`

Elementwise sine operation for values given in radians.
"""
function sin(input1::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.sin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`slice`

Extracts a slice of input1, beginning at the start coordinates,
and extending for size elements in each direction.
No data conversion happens during a slice operation.
"""
function slice(input1::Value, start::Value, size::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, start, size, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.slice", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`sub`

Elementwise subtraction of input1 and input2. Axis of size 1 will be
broadcast as necessary. Rank of input tensors must match.
"""
function sub(input1::Value, input2::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, input2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.sub", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`table`

Table lookup operation. For int8_t TABLE operation, perform a 256 entry
table lookup returning an int8_t value. For int16_t tables, the int16_t
input is treated as a fixed-point 9.7 value. The most significant 9 bits
are used to index into the table. The fractional 7 bits are used to
interpolate based on table[index] and table[index+1]. For int16_t inputs,
the TABLE operator returns a 16.7 interpolated value in an int32_t. This
value can then be input to the RESCALE operator to scale to the required
output data type. Note that int16_t table has 513 values to handle
table[index+1] when index=511.

An int16_t to int16_t table lookup can be constructed in TOSA as follows:
* Use the TABLE operator to produce a fixed point 16.7 interpolated result
* Use RESCALE (in_t=int32_t, out_t=int16_t, scale=1<<14, shift=21) to
  scale the output to int16_t range (or alternate scale as required)
"""
function table(input1::Value, table::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, table, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.table", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tanh`

Parameterized hyperbolic tangent: \$ tanh(x) = \\frac{1 - e^{-2x}}{1 + e^{-2x}} \$.

For quantized integer data types, the TABLE operator should be used instead.
Each implementation may choose an appropriate TABLE given the scale and zero
point of the input data. Eight or sixteen bit precision tables may be used
based on the input tensor to the tanh function.
"""
function tanh(input::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.tanh", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tile`

Replicates input1 multiples times along each dimension.
"""
function tile(input1::Value, multiples::Value; output::IR.Type, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, multiples, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.tile", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`transpose_conv2d`

Performs a 2D transposed convolution over the given tensor input, using the
weights tensor. Implementations may choose to skip calculation of multiplies
by zero at fractional input positions.
"""
function transpose_conv2d(input::Value, weight::Value, bias::Value, input_zp::Value, weight_zp::Value; output::IR.Type, out_pad, stride, acc_type, local_bound=nothing, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input, weight, bias, input_zp, weight_zp, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("out_pad", out_pad), NamedAttribute("stride", stride), NamedAttribute("acc_type", acc_type), ]
    !isnothing(local_bound) && push!(attributes, NamedAttribute("local_bound", local_bound))
    
    create_operation(
        "tosa.transpose_conv2d", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`transpose`

Permutes the dimensions of the input tensor input1 based on the perms
argument. Each value in the perms list must be a valid dimension of the
input tensor and may not be repeated.
"""
function transpose(input1::Value; output::IR.Type, perms, location=Location())
    op_ty_results = IR.Type[output, ]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("perms", perms), ]
    
    create_operation(
        "tosa.transpose", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`variable`

Defines a new TOSA variable. This is a persistent mutable value across multiple
TOSA graph invocations. Modifications are expressed using read/write semantics.
"""
function variable(; name, var_shape, type, initial_value=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("name", name), NamedAttribute("var_shape", var_shape), NamedAttribute("type", type), ]
    !isnothing(initial_value) && push!(attributes, NamedAttribute("initial_value", initial_value))
    
    create_operation(
        "tosa.variable", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`variable_read`

Reads the value from a pseudo-buffer resource holding a persistent mutable tensor.
"""
function variable_read(; output1::IR.Type, name, location=Location())
    op_ty_results = IR.Type[output1, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("name", name), ]
    
    create_operation(
        "tosa.variable_read", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`variable_write`

Assigns a value to the pseudo-buffer resource holding a persistent mutable tensor.
"""
function variable_write(input1::Value; name, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[input1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("name", name), ]
    
    create_operation(
        "tosa.variable_write", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`while_loop`

Generates and evaluates a Boolean condition and either executes a loop body
or exits the loop. This action is performed repeatedly after
updating and re-evaluating the Boolean condition every iteration. This
implements the semantic foreach or while iterative loop structure.
"""
function while_loop(input_list::Vector{Value}; output_list::Vector{IR.Type}, cond_graph::Region, body_graph::Region, location=Location())
    op_ty_results = IR.Type[output_list..., ]
    operands = Value[input_list..., ]
    owned_regions = Region[cond_graph, body_graph, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.while_loop", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`yield`

return operation within the conditional and body of
structured control flow. Operation takes variadic operands
but produces no results of its own.
"""
function yield(inputs::Vector{Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[inputs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "tosa.yield", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

end # tosa

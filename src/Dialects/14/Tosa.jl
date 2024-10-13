module tosa

import ...IR: IR, NamedAttribute, Value, value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`abs`

Elementwise absolute value operation
"""
function abs(input1; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.abs", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`add`

Elementwise addition of input1 and input2. Axis of size 1 will be broadcast,
as necessary.
"""
function add(input1, input2; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.add", location;
        operands, owned_regions, successors, attributes,
        results=results,
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
function apply_scale(value, multiplier, shift; output::IR.Type, double_round, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(value), value(multiplier), value(shift), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("double_round", double_round), ]
    
    IR.create_operation(
        "tosa.apply_scale", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`argmax`

This returns the index with the largest value across the given axis of the
input tensor.
"""
function argmax(input; output::IR.Type, axis, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis), ]
    
    IR.create_operation(
        "tosa.argmax", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`arithmetic_right_shift`

Elementwise arithmetic right shift of input1 by the amount specified in
input2. Axis of size 1 will be broadcast, as necessary.
"""
function arithmetic_right_shift(input1, input2; output::IR.Type, round, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("round", round), ]
    
    IR.create_operation(
        "tosa.arithmetic_right_shift", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`avg_pool2d`

This performs an average pooling over the given input tensor. A sliding
window of size given by <kernel size> is passed over the input tensor, with
the mean value being placed in the output tensor.
"""
function avg_pool2d(input; output::IR.Type, kernel, stride, pad, quantization_info=nothing, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("kernel", kernel), namedattribute("stride", stride), namedattribute("pad", pad), ]
    !isnothing(quantization_info) && push!(attributes, namedattribute("quantization_info", quantization_info))
    
    IR.create_operation(
        "tosa.avg_pool2d", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`bitwise_and`

Elementwise bitwise AND of input1 and input2. Axis of size 1
will be broadcast as necessary.
"""
function bitwise_and(input1, input2; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.bitwise_and", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`bitwise_not`

Elementwise bitwise NOT of input tensor.
"""
function bitwise_not(input1; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.bitwise_not", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`bitwise_or`

Elementwise bitwise OR of input1 and input2. Axis of size 1 will be
broadcast as necessary.
"""
function bitwise_or(input1, input2; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.bitwise_or", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`bitwise_xor`

Elementwise bitwise XOR of input1 and input2. Axis of size 1 will be
broadcast as necessary.
"""
function bitwise_xor(input1, input2; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.bitwise_xor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`cast`

Performs a set of permissible cast operations
    Mode                    Input   Output
    ---------------------------------------
    signed 8 to bool        int8    Boolean
    signed 16 to bool       int16   Boolean
    signed 32 to bool       int32   Boolean
    bool to 8               Boolean int8
    bool to 16              Boolean int16
    bool to 32              Boolean int32
    signed 8 to signed 16   int8    int16
    signed 8 to signed 32   int8    int32
    signed 16 to signed 8   int16   int8
    signed 16 to signed 32  int16   int32
    signed 32 to signed 8   int32   int8
    signed 32 to signed 16  int32   int16
    float to signed 8       float   int8
    float to signed 16      float   int16
    signed 8 to float       int8    float
    signed 16 to float      int16   float
"""
function cast(input; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.cast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ceil`

Elementwise ceiling operation
"""
function ceil(input1; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.ceil", location;
        operands, owned_regions, successors, attributes,
        results=results,
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
function clamp(input; output::IR.Type, min_int, max_int, min_fp, max_fp, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("min_int", min_int), namedattribute("max_int", max_int), namedattribute("min_fp", min_fp), namedattribute("max_fp", max_fp), ]
    
    IR.create_operation(
        "tosa.clamp", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`clz`

Elementwise count leading zeros operation
"""
function clz(input1; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.clz", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`concat`

Concatenate a variadic amount of tensors along a given axis. No data
conversion happens during a concat operation.
"""
function concat(input1; output::IR.Type, axis, location=Location())
    results = IR.Type[output, ]
    operands = Value[value.(input1)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis), ]
    
    IR.create_operation(
        "tosa.concat", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`const_`

A node containing constant data for use as the input to an operation. May
hold data in any of the supported data formats.
"""
function const_(; output=nothing::Union{Nothing,IR.Type}, value, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("value", value),]
    !isnothing(output) && push!(_results, output)

    return IR.create_operation(
        "tosa.const",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`conv2d`

Performs a 2D convolution over the given tensor input, using the weight
tensor.
"""
function conv2d(input, weight, bias; output::IR.Type, pad, stride, dilation, quantization_info=nothing, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), value(weight), value(bias), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("pad", pad), namedattribute("stride", stride), namedattribute("dilation", dilation), ]
    !isnothing(quantization_info) && push!(attributes, namedattribute("quantization_info", quantization_info))
    
    IR.create_operation(
        "tosa.conv2d", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`conv3d`

Performs a 3D convolution over the given input tensor.
"""
function conv3d(input, weight, bias; output::IR.Type, pad, stride, dilation, quantization_info=nothing, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), value(weight), value(bias), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("pad", pad), namedattribute("stride", stride), namedattribute("dilation", dilation), ]
    !isnothing(quantization_info) && push!(attributes, namedattribute("quantization_info", quantization_info))
    
    IR.create_operation(
        "tosa.conv3d", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`custom`

Hardware implementing TOSA may choose to add additional custom operators
that are not expressed in the existing TOSA operations. These operators are
not expected to be portable across TOSA implementations. The input and
output signatures must be expressed in the corresponding TOSA node.
"""
function custom(inputs; outputs::Vector{IR.Type}, identifier, location=Location())
    results = IR.Type[outputs..., ]
    operands = Value[value.(inputs)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("identifier", identifier), ]
    
    IR.create_operation(
        "tosa.custom", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`depthwise_conv2d`

Performs 2D convolutions separately over each channel of the given tensor
input, using the weight tensor.
"""
function depthwise_conv2d(input, weight, bias; output::IR.Type, pad, stride, dilation, quantization_info=nothing, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), value(weight), value(bias), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("pad", pad), namedattribute("stride", stride), namedattribute("dilation", dilation), ]
    !isnothing(quantization_info) && push!(attributes, namedattribute("quantization_info", quantization_info))
    
    IR.create_operation(
        "tosa.depthwise_conv2d", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`div`

Elementwise integer divide operator of input1 by input2. Axis of size 1
will be broadcast, as necessary.
"""
function div(input1, input2; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.div", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`equal`

Elementwise comparison operation
"""
function equal(input1, input2; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.equal", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`exp`

Elementwise e to the x operation
"""
function exp(input1; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.exp", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`floor`

Elementwise floor operation
"""
function floor(input1; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.floor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fully_connected`

Performs a fully connected network.
"""
function fully_connected(input, weight, bias; output::IR.Type, quantization_info=nothing, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), value(weight), value(bias), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(quantization_info) && push!(attributes, namedattribute("quantization_info", quantization_info))
    
    IR.create_operation(
        "tosa.fully_connected", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`gather`

Generate a tensor for which each element in the output is a slice of the
values tensor based on the value of indices.
"""
function gather(values, indices; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(values), value(indices), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.gather", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`greater_equal`

Elementwise comparison operation
"""
function greater_equal(input1, input2; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.greater_equal", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`greater`

Elementwise greater than comparison operation
"""
function greater(input1, input2; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.greater", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`identity`

Returns a tensor with the same shape, size, type
and content as the input.
"""
function identity(input1; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.identity", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`cond_if`

Evaluates a Boolean condition and then takes one of two distinct execution
paths. This implements the semantic If-then-else structure.
"""
function cond_if(cond, inputs; output::Vector{IR.Type}, then_branch::Region, else_branch::Region, location=Location())
    results = IR.Type[output..., ]
    operands = Value[value(cond), value.(inputs)..., ]
    owned_regions = Region[then_branch, else_branch, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.cond_if", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`log`

Elementwise natural logarithm operation
"""
function log(input1; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.log", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`logical_and`

Elementwise logical AND of input1 and input2. Axis of size 1 will be
broadcast, as necessary.
"""
function logical_and(input1, input2; z::IR.Type, location=Location())
    results = IR.Type[z, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.logical_and", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`logical_left_shift`

Elementwise left shift of input1 and input2. Axis of size 1 will be
broadcast, as necessary.
"""
function logical_left_shift(input1, input2; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.logical_left_shift", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`logical_not`

Elementwise logical NOT of input.
"""
function logical_not(input1; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.logical_not", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`logical_or`

Elementwise logical OR of input1 and input2. Axis of size 1 will be
broadcast as necessary.
"""
function logical_or(input1, input2; z::IR.Type, location=Location())
    results = IR.Type[z, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.logical_or", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`logical_right_shift`

Elementwise logical right shift of input1 by the amount specified in input2.
Axis of size 1 will be broadcast, as necessary.
"""
function logical_right_shift(input1, input2; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.logical_right_shift", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`logical_xor`

Elementwise logical XOR of input1 and input2.  Axis of size 1 will be
broadcast as necessary.
"""
function logical_xor(input1, input2; z::IR.Type, location=Location())
    results = IR.Type[z, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.logical_xor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`matmul`

Performs a two dimensional matrix multiplication. This allows both inputs to
be activations, rather than reserving weights as an attribute in the
FULLY_CONNECTED operator.
"""
function matmul(a, b; c::IR.Type, quantization_info=nothing, location=Location())
    results = IR.Type[c, ]
    operands = Value[value(a), value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(quantization_info) && push!(attributes, namedattribute("quantization_info", quantization_info))
    
    IR.create_operation(
        "tosa.matmul", location;
        operands, owned_regions, successors, attributes,
        results=results,
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
function max_pool2d(input; output::IR.Type, kernel, stride, pad, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("kernel", kernel), namedattribute("stride", stride), namedattribute("pad", pad), ]
    
    IR.create_operation(
        "tosa.max_pool2d", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`maximum`

Elementwise max of input1 and input2. Axis of size 1 will be broadcast, as
necessary.
"""
function maximum(input1, input2; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.maximum", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`minimum`

Elementwise minimum of input1 and input2. Axis of size 1
will be broadcast, as necessary.
"""
function minimum(input1, input2; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.minimum", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`mul`

Elementwise multiplication (Hadamard product) of input1 and input2.
Axis of size 1 will be broadcast, as necessary.
"""
function mul(input1, input2; output::IR.Type, shift, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("shift", shift), ]
    
    IR.create_operation(
        "tosa.mul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`negate`

Elementwise negation operation
"""
function negate(input1; output::IR.Type, quantization_info=nothing, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(quantization_info) && push!(attributes, namedattribute("quantization_info", quantization_info))
    
    IR.create_operation(
        "tosa.negate", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pad`

Pads a tensor along borders of each dimension with pad_value.
"""
function pad(input1, padding, pad_const=nothing; output::IR.Type, quantization_info=nothing, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(padding), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(pad_const) && push!(operands, value(pad_const))
    !isnothing(quantization_info) && push!(attributes, namedattribute("quantization_info", quantization_info))
    
    IR.create_operation(
        "tosa.pad", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pow`

Elementwise input1 raised to the power of input2.
Axis of size 1 will be broadcast, as necessary.
"""
function pow(input1, input2; z::IR.Type, location=Location())
    results = IR.Type[z, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.pow", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reciprocal`

Elementwise reciprocal operation. For integer operation, a TABLE should be
used with the appropriate ranges.
"""
function reciprocal(input1; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.reciprocal", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reduce_all`

Reduce a tensor along the given axis with a logical AND operation
"""
function reduce_all(input; output::IR.Type, axis, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis), ]
    
    IR.create_operation(
        "tosa.reduce_all", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reduce_any`

Reduce a tensor along the given axis with a logical OR operation
"""
function reduce_any(input; output::IR.Type, axis, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis), ]
    
    IR.create_operation(
        "tosa.reduce_any", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reduce_max`

Reduce a tensor along the given axis with a maximum operation
"""
function reduce_max(input; output::IR.Type, axis, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis), ]
    
    IR.create_operation(
        "tosa.reduce_max", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reduce_min`

Reduce a tensor along the given axis with a minimum operation
"""
function reduce_min(input; output::IR.Type, axis, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis), ]
    
    IR.create_operation(
        "tosa.reduce_min", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reduce_prod`

Reduce a tensor along the given axis by computing the product of the axis.
"""
function reduce_prod(input; output::IR.Type, axis, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis), ]
    
    IR.create_operation(
        "tosa.reduce_prod", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reduce_sum`

Reduce a tensor along the given axis by computing the sum of the axis.
"""
function reduce_sum(input; output::IR.Type, axis, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis), ]
    
    IR.create_operation(
        "tosa.reduce_sum", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reluN`

ReLU with a scalar maximum value.
"""
function reluN(input; output::IR.Type, max_int, max_fp, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("max_int", max_int), namedattribute("max_fp", max_fp), ]
    
    IR.create_operation(
        "tosa.reluN", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`rescale`

Rescale quantized values into a new domain. Supported rescalings are:
Mode                    Input   Output
signed 8 to 8           int8    int8
signed 8 to 16          int8    int16
signed 8 to 32          int8    int32
signed 16 to 8          int16   int8
signed 16 to 16         int16   int16
signed 16 to 32         int16   int32
signed 32 to 8          int32   int8
signed 32 to 16         int32   int16
signed 32 to 32         int32   int32
signed 48 to 8          int48   int8
signed 48 to 16         int48   int16
signed 48 to 32         int48   int32
unsigned 8 to signed 8  uint8   int8
signed 8 to unsigned 8  int8    uint8
"""
function rescale(input; output::IR.Type, input_zp, output_zp, multiplier, shift, scale32, double_round, per_channel, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("input_zp", input_zp), namedattribute("output_zp", output_zp), namedattribute("multiplier", multiplier), namedattribute("shift", shift), namedattribute("scale32", scale32), namedattribute("double_round", double_round), namedattribute("per_channel", per_channel), ]
    
    IR.create_operation(
        "tosa.rescale", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reshape`

Returns a tensor with the same type/values as the input, with a new shape
specified by the shape argument. Reshape may operate on tensors of any rank.
No data conversion happens during a reshape operation.
"""
function reshape(input1; output::IR.Type, new_shape, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("new_shape", new_shape), ]
    
    IR.create_operation(
        "tosa.reshape", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`resize`

Resizes a tensor. Resize is only allowed in the H and W dimensions. In
expected use, stride_y is approximately (IH<<shift)/OH and stride_x is
approximately (IW<<shift)/OW.  OH and OW are also supplied as inputs since
there may be off by one errors if calculating OH and OW from the strides.
"""
function resize(input; output::IR.Type, output_size, stride, offset, shift, stride_fp, offset_fp, mode, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("output_size", output_size), namedattribute("stride", stride), namedattribute("offset", offset), namedattribute("shift", shift), namedattribute("stride_fp", stride_fp), namedattribute("offset_fp", offset_fp), namedattribute("mode", mode), ]
    
    IR.create_operation(
        "tosa.resize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reverse`

Returns a tensor with the same type/values as the input, with the data
reversed along the given axis. No data conversion happens during a reverse
operation.
"""
function reverse(input; output::IR.Type, axis, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("axis", axis), ]
    
    IR.create_operation(
        "tosa.reverse", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`rsqrt`

Elementwise reciprocal square root operation. For integer operation, a TABLE
should be used with the appropriate ranges.
"""
function rsqrt(input1; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.rsqrt", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`scatter`

The values_out tensor is set to the values_in tensor with data modified as follows:
data from the input tensor is inserted at the positions specified by the indices tensor.
"""
function scatter(values_in, indices, input; values_out::IR.Type, location=Location())
    results = IR.Type[values_out, ]
    operands = Value[value(values_in), value(indices), value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.scatter", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`select`

Elementwise select of the output based on a condition.
"""
function select(input1, input2, input3; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), value(input3), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.select", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sigmoid`

Sigmoid function: output = 1 / (1 + exp(-input))
For quantized integer data types, the TABLE operator should be used instead
with the following definition.  The sigmoid table has 513 entries each of
16-bit precision and covering the input range -16.0 to +16.0
in steps of 1/16.
"""
function sigmoid(input; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.sigmoid", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`slice`

Extracts a slice of the input1 on the given axis, beginning at the
start coordinates, and extending for size elements in each direction.  No
data conversion happens during a slice operation.
"""
function slice(input; output::IR.Type, start, size, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("start", start), namedattribute("size", size), ]
    
    IR.create_operation(
        "tosa.slice", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sub`

Elementwise subtraction of input1 and input2. Axis of size 1 will be
broadcast as necessary.
"""
function sub(input1, input2; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(input2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.sub", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`table`

Interpolated table lookup operation. Input values are scaled to create a
fixed-point 9.7 value.    The high 9 bits are used to index into the table.
The fractional bits are used to interpolate based on the looked up value and
the index+1 value in the table. The TABLE operator then returns a 16.7
interpolated value. Note that there must be 513 values to handle the full
range of inputs.

The TABLE operator is expected to be used as follows:
* A RESCALE node is expected before the TABLE operator to scale the input
  to a full int16_t range for the table lookup
* If an int16_t result is required then follow the TABLE operator with a
  RESCALE with a right shift of 7
* If an int8_t result is required then follow the TABLE operator with a
  RESCALE with a right shift of 15
"""
function table(input, table; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), value(table), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.table", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tanh`

Parameterized hyperbolic tangent.
For quantized integer data types, the TABLE operator should be used instead
with the following definition.  The tanh_table has 513 entries each of
16-bit precision and covering the input range -8.0 to +8.0 in steps of 1/32.
"""
function tanh(input; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.tanh", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tile`

Replicates input 0 multiplies times along each dimension.
"""
function tile(input1; output::IR.Type, multiples, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("multiples", multiples), ]
    
    IR.create_operation(
        "tosa.tile", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`transpose_conv2d`

Performs a 2D transposed convolution over the given tensor input, using the
weights tensor.
"""
function transpose_conv2d(input, filter, bias; output::IR.Type, out_pad, stride, dilation, out_shape, quantization_info=nothing, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input), value(filter), value(bias), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("out_pad", out_pad), namedattribute("stride", stride), namedattribute("dilation", dilation), namedattribute("out_shape", out_shape), ]
    !isnothing(quantization_info) && push!(attributes, namedattribute("quantization_info", quantization_info))
    
    IR.create_operation(
        "tosa.transpose_conv2d", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`transpose`

Permutes the dimensions based on perm.
"""
function transpose(input1, perms; output::IR.Type, location=Location())
    results = IR.Type[output, ]
    operands = Value[value(input1), value(perms), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.transpose", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`while_loop`

Generates and evaluates a Bool condition and either executes a loop body or
exits to another control point. This action is performed repeatedly after
updating and re-evaluating the Boolean condition every iteration. This
implements the semantic foreach or while iterative loop structure.
"""
function while_loop(inputs; output::Vector{IR.Type}, cond::Region, body::Region, location=Location())
    results = IR.Type[output..., ]
    operands = Value[value.(inputs)..., ]
    owned_regions = Region[cond, body, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.while_loop", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`yield`

return operation within the conditional and body of
structured control flow. Operation takes variadic operands
but produces no results of its own.
"""
function yield(inputs; location=Location())
    results = IR.Type[]
    operands = Value[value.(inputs)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "tosa.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # tosa

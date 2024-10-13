module pdl

import ...IR: IR, NamedAttribute, Value, value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`apply_native_constraint`

`pdl.apply_native_constraint` operations apply a native C++ constraint, that
has been registered externally with the consumer of PDL, to a given set of
entities. The constraint is permitted to accept any number of constant
valued parameters.

# Example

```mlir
// Apply `myConstraint` to the entities defined by `input`, `attr`, and
// `op`. `42`, `\"abc\"`, and `i32` are constant parameters passed to the
// constraint.
pdl.apply_native_constraint \"myConstraint\"[42, \"abc\", i32](%input, %attr, %op : !pdl.value, !pdl.attribute, !pdl.operation)
```
"""
function apply_native_constraint(args; name, constParams=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value.(args)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("name", name), ]
    !isnothing(constParams) && push!(attributes, namedattribute("constParams", constParams))
    
    IR.create_operation(
        "pdl.apply_native_constraint", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_native_rewrite`

`pdl.apply_native_rewrite` operations apply a native C++ function, that has
been registered externally with the consumer of PDL, to perform a rewrite
and optionally return a number of values. The native function may accept any
number of arguments and constant attribute parameters. This operation is
used within a pdl.rewrite region to enable the interleaving of native
rewrite methods with other pdl constructs.

# Example

```mlir
// Apply a native rewrite method that returns an attribute.
%ret = pdl.apply_native_rewrite \"myNativeFunc\"[42, \"gt\"](%arg0, %arg1) : !pdl.attribute
```

```c++
// The native rewrite as defined in C++:
static void myNativeFunc(ArrayRef<PDLValue> args, ArrayAttr constantParams,
                         PatternRewriter &rewriter,
                         PDLResultList &results) {
  Value arg0 = args[0].cast<Value>();
  Value arg1 = args[1].cast<Value>();
  IntegerAttr param0 = constantParams[0].cast<IntegerAttr>();
  StringAttr param1 = constantParams[1].cast<StringAttr>();

  // Just push back the first param attribute.
  results.push_back(param0);
}

void registerNativeRewrite(PDLPatternModule &pdlModule) {
  pdlModule.registerRewriteFunction(\"myNativeFunc\", myNativeFunc);
}
```
"""
function apply_native_rewrite(args; results_::Vector{IR.Type}, name, constParams=nothing, location=Location())
    results = IR.Type[results_..., ]
    operands = Value[value.(args)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("name", name), ]
    !isnothing(constParams) && push!(attributes, namedattribute("constParams", constParams))
    
    IR.create_operation(
        "pdl.apply_native_rewrite", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`attribute`

`pdl.attribute` operations capture named attribute edges into an operation.
Instances of this operation define, and partially constrain, attributes of a
given operation. A `pdl.attribute` may partially constrain the input by
specifying an expected attribute value type (via a `pdl.type` operation), or
a constant value for the attribute (via `val`). Only one of these may be set
for a given input, as the type of the constant value provides the type. When
defined within a `pdl.rewrite` region, the constant value must be specified.

# Example

```mlir
// Define an attribute:
%attr = pdl.attribute

// Define an attribute with an expected type:
%type = pdl.type : i32
%attr = pdl.attribute : %type

// Define an attribute with a constant value:
%attr = pdl.attribute \"hello\"
```
"""
function attribute(type=nothing; attr::IR.Type, value=nothing, location=Location())
    results = IR.Type[attr, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(type) && push!(operands, value(type))
    !isnothing(value) && push!(attributes, namedattribute("value", value))
    
    IR.create_operation(
        "pdl.attribute", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`erase`

`pdl.erase` operations are used within `pdl.rewrite` regions to specify that
an input operation should be marked as erased. The semantics of this
operation correspond with the `eraseOp` method on a `PatternRewriter`.

# Example

```mlir
pdl.erase %root
```
"""
function erase(operation; location=Location())
    results = IR.Type[]
    operands = Value[value(operation), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "pdl.erase", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`operand`

`pdl.operand` operations capture external operand edges into an operation
node that originate from operations or block arguments not otherwise
specified within the pattern (i.e. via `pdl.result` or `pdl.results`). These
operations define individual operands of a given operation. A `pdl.operand`
may partially constrain an operand by specifying an expected value type
(via a `pdl.type` operation).

# Example

```mlir
// Define an external operand:
%operand = pdl.operand

// Define an external operand with an expected type:
%type = pdl.type : i32
%operand = pdl.operand : %type
```
"""
function operand(type=nothing; val::IR.Type, location=Location())
    results = IR.Type[val, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(type) && push!(operands, value(type))
    
    IR.create_operation(
        "pdl.operand", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`operands`

`pdl.operands` operations capture external operand range edges into an
operation node that originate from operations or block arguments not
otherwise specified within the pattern (i.e. via `pdl.result` or
`pdl.results`). These operations define groups of input operands into a
given operation. A `pdl.operands` may partially constrain a set of input
operands by specifying expected value types (via `pdl.types` operations).

# Example

```mlir
// Define a range of input operands:
%operands = pdl.operands

// Define a range of input operands with expected types:
%types = pdl.types : [i32, i64, i32]
%typed_operands = pdl.operands : %types
```
"""
function operands_(type=nothing; val::IR.Type, location=Location())
    results = IR.Type[val, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(type) && push!(operands, value(type))
    
    IR.create_operation(
        "pdl.operands", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`operation`

`pdl.operation` operations define operation nodes within a pattern. Within
a match sequence, i.e. when directly nested within a `pdl.pattern`, these
operations correspond to input operations, or those that already existing
within the MLIR module. Inside of a `pdl.rewrite`, these operations
correspond to operations that should be created as part of the replacement
sequence.

`pdl.operation`s are composed of a name, and a set of attribute, operand,
and result type values, that map to what those that would be on a
constructed instance of that operation. The results of a `pdl.operation` are
a handle to the operation itself. Handles to the results of the operation
can be extracted via `pdl.result`.

# Example

```mlir
// Define an instance of a `foo.op` operation.
%op = pdl.operation \"foo.op\"(%arg0, %arg1 : !pdl.value, !pdl.value)
  {\"attrA\" = %attr0} -> (%type, %type : !pdl.type, !pdl.type)
```

When used within a matching context, the name of the operation may be
omitted.

When used within a rewriting context, i.e. when defined within a
`pdl.rewrite`, all of the result types must be \"inferable\". This means that
the type must be attributable to either a constant type value or the result
type of another entity, such as an attribute, the result of a
`apply_native_rewrite`, or the result type of another operation. If the
result type value does not meet any of these criteria, the operation must
override the `InferTypeOpInterface` to ensure that the result types can be
inferred.

The operands of the operation are interpreted in the following ways:

1) A single !pdl.range<value>:

In this case, the single range is treated as all of the operands of the
operation.

```mlir
// Define an instance with single range of operands.
%op = pdl.operation \"std.return\"(%allArgs : !pdl.range<value>)
```

2) A variadic number of either !pdl.value or !pdl.range<value>:

In this case, the inputs are expected to correspond with the operand groups
defined on the operation in ODS.

```tablgen
// Given the following operation definition in ODS:
def MyIndirectCallOp {
  let results = (outs FunctionType:\$call, Variadic<AnyType>:\$args);
}
```

```mlir
// We can match the operands as so:
%op = pdl.operation \"my.indirect_call\"(%call, %args : !pdl.value, !pdl.range<value>)
```

The results of the operation are interpreted in the following ways:

1) A single !pdl.range<type>:

In this case, the single range is treated as all of the result types of the
operation.

```mlir
// Define an instance with single range of types.
%allResultTypes = pdl.types
%op = pdl.operation \"builtin.unrealized_conversion_cast\" -> (%allResultTypes : !pdl.types)
```

2) A variadic number of either !pdl.type or !pdl.range<type>:

In this case, the inputs are expected to correspond with the result groups
defined on the operation in ODS.

```tablgen
// Given the following operation definition in ODS:
def MyOp {
  let results = (outs SomeType:\$result, Variadic<SomeType>:\$otherResults);
}
```

```mlir
// We can match the results as so:
%result = pdl.type
%otherResults = pdl.types
%op = pdl.operation \"foo.op\" -> (%result, %otherResults : !pdl.type, !pdl.range<type>)
```
"""
function operation(operands_, attributes_, types; op::IR.Type, name=nothing, attributeNames, location=Location())
    results = IR.Type[op, ]
    operands = Value[value.(operands_)..., value.(attributes_)..., value.(types)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("attributeNames", attributeNames), ]
    push!(attributes, operandsegmentsizes([length(operands), length(attributes), length(types), ]))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    
    IR.create_operation(
        "pdl.operation", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pattern`

`pdl.pattern` operations provide a transformable representation for a
`RewritePattern`. The attributes on this operation correspond to the various
metadata on a `RewritePattern`, such as the benefit. The match section of
the pattern is specified within the region body, with the rewrite provided
by a terminating `pdl.rewrite`.

# Example

```mlir
// Provide a pattern matching \"foo.op\" that replaces the root with its
// operand.
pdl.pattern : benefit(1) {
  %resultType = pdl.type
  %inputOperand = pdl.operand
  %root = pdl.operation \"foo.op\"(%inputOperand) -> (%resultType)
  pdl.rewrite %root {
    pdl.replace %root with (%inputOperand)
  }
}
```
"""
function pattern(; benefit, sym_name=nothing, body::Region, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[body,]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("benefit", benefit),]
    !isnothing(sym_name) && push!(_attributes, namedattribute("sym_name", sym_name))

    return IR.create_operation(
        "pdl.pattern",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`replace`

`pdl.replace` operations are used within `pdl.rewrite` regions to specify
that an input operation should be marked as replaced. The semantics of this
operation correspond with the `replaceOp` method on a `PatternRewriter`. The
set of replacement values can be either:
* a single `Operation` (`replOperation` should be populated)
  - The operation will be replaced with the results of this operation.
* a set of `Value`s (`replValues` should be populated)
  - The operation will be replaced with these values.

# Example

```mlir
// Replace root node with 2 values:
pdl.replace %root with (%val0, %val1 : !pdl.value, !pdl.value)

// Replace root node with a range of values:
pdl.replace %root with (%vals : !pdl.range<value>)

// Replace root with another operation:
pdl.replace %root with %otherOp
```
"""
function replace(operation, replOperation=nothing; replValues, location=Location())
    results = IR.Type[]
    operands = Value[value(operation), value.(replValues)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(replOperation) && push!(operands, value(replOperation))
    push!(attributes, operandsegmentsizes([1, (replOperation==nothing) ? 0 : 1length(replValues), ]))
    
    IR.create_operation(
        "pdl.replace", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`result`

`pdl.result` operations extract result edges from an operation node within
a pattern or rewrite region. The provided index is zero-based, and
represents the concrete result to extract, i.e. this is not the result index
as defined by the ODS definition of the operation.

# Example

```mlir
// Extract a result:
%operation = pdl.operation ...
%pdl_result = pdl.result 1 of %operation

// Imagine the following IR being matched:
%result_0, %result_1 = foo.op ...

// If the example pattern snippet above were matching against `foo.op` in
// the IR snippet, `%pdl_result` would correspond to `%result_1`.
```
"""
function result(parent; val::IR.Type, index, location=Location())
    results = IR.Type[val, ]
    operands = Value[value(parent), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("index", index), ]
    
    IR.create_operation(
        "pdl.result", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`results`

`pdl.results` operations extract a result group from an operation within a
pattern or rewrite region. If an index is provided, this operation extracts
a result group as defined by the ODS definition of the operation. In this
case the result of this operation may be either a single `pdl.value` or
a `pdl.range<value>`, depending on the constraint of the result in ODS. If
no index is provided, this operation extracts the full result range of the
operation.

# Example

```mlir
// Extract all of the results of an operation:
%operation = pdl.operation ...
%results = pdl.results of %operation

// Extract the results in the first result group of an operation, which is
// variadic:
%operation = pdl.operation ...
%results = pdl.results 0 of %operation -> !pdl.range<value>

// Extract the results in the second result group of an operation, which is
// not variadic:
%operation = pdl.operation ...
%results = pdl.results 1 of %operation -> !pdl.value
```
"""
function results_(parent; val::IR.Type, index=nothing, location=Location())
    results = IR.Type[val, ]
    operands = Value[value(parent), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(index) && push!(attributes, namedattribute("index", index))
    
    IR.create_operation(
        "pdl.results", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`rewrite`

`pdl.rewrite` operations terminate the region of a `pdl.pattern` and specify
the main rewrite of a `pdl.pattern`, on the optional root operation. The
rewrite is specified either via a string name (`name`) to a native
rewrite function, or via the region body. The rewrite region, if specified,
must contain a single block. If the rewrite is external it functions
similarly to `pdl.apply_native_rewrite`, and takes a set of constant
parameters and a set of additional positional values defined within the
matcher as arguments. If the rewrite is external, the root operation is
passed to the native function as the leading arguments. The root operation,
if provided, specifies the starting point in the pattern for the subgraph
isomorphism search. Pattern matching will proceed from this node downward
(towards the defining operation) or upward (towards the users) until all
the operations in the pattern have been matched. If the root is omitted,
the pdl_interp lowering will automatically select the best root of the
pdl.rewrite among all the operations in the pattern.

# Example

```mlir
// Specify an external rewrite function:
pdl.rewrite %root with \"myExternalRewriter\"(%value : !pdl.value)

// Specify a rewrite inline using PDL with the given root:
pdl.rewrite %root {
  %op = pdl.operation \"foo.op\"(%arg0, %arg1)
  pdl.replace %root with %op
}

// Specify a rewrite inline using PDL, automatically selecting root:
pdl.rewrite {
  %op1 = pdl.operation \"foo.op\"(%arg0, %arg1)
  %op2 = pdl.operation \"bar.op\"(%arg0, %arg1)
  pdl.replace %root1 with %op1
  pdl.replace %root2 with %op2
}
```
"""
function rewrite(root=nothing; externalArgs, name=nothing, externalConstParams=nothing, body::Region, location=Location())
    results = IR.Type[]
    operands = Value[value.(externalArgs)..., ]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(root) && push!(operands, value(root))
    push!(attributes, operandsegmentsizes([(root==nothing) ? 0 : 1length(externalArgs), ]))
    !isnothing(name) && push!(attributes, namedattribute("name", name))
    !isnothing(externalConstParams) && push!(attributes, namedattribute("externalConstParams", externalConstParams))
    
    IR.create_operation(
        "pdl.rewrite", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`type`

`pdl.type` operations capture result type constraints of `Attributes`,
`Values`, and `Operations`. Instances of this operation define, and
partially constrain, results types of a given entity. A `pdl.type` may
partially constrain the result by specifying a constant `Type`.

# Example

```mlir
// Define a type:
%type = pdl.type

// Define a type with a constant value:
%type = pdl.type : i32
```
"""
function type(; result::IR.Type, type=nothing, location=Location())
    _results = IR.Type[result,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(type) && push!(_attributes, namedattribute("type", type))

    return IR.create_operation(
        "pdl.type",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`types`

`pdl.types` operations capture result type constraints of `Value`s, and
`Operation`s. Instances of this operation define results types of a given
entity. A `pdl.types` may partially constrain the results by specifying
an array of `Type`s.

# Example

```mlir
// Define a range of types:
%types = pdl.types

// Define a range of types with a range of constant values:
%types = pdl.types : [i32, i64, i32]
```
"""
function types(; result::IR.Type, types=nothing, location=Location())
    _results = IR.Type[result,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(types) && push!(_attributes, namedattribute("types", types))

    return IR.create_operation(
        "pdl.types",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # pdl

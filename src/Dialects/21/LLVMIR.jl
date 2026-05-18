module llvm

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes
import ...API



function ashr(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, isExact=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(isExact) && push!(attributes, NamedAttribute("isExact", isExact))
    
    create_operation(
        "llvm.ashr", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function add(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.add", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function addrspacecast(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.addrspacecast", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mlir_addressof`

Creates an SSA value containing a pointer to a global value (function,
variable or alias). The global value can be defined after its first
referenced. If the global value is a constant, storing into it is not
allowed.

Examples:

```mlir
func @foo() {
  // Get the address of a global variable.
  %0 = llvm.mlir.addressof @const : !llvm.ptr

  // Use it as a regular pointer.
  %1 = llvm.load %0 : !llvm.ptr -> i32

  // Get the address of a function.
  %2 = llvm.mlir.addressof @foo : !llvm.ptr

  // The function address can be used for indirect calls.
  llvm.call %2() : !llvm.ptr, () -> ()

  // Get the address of an aliased global.
  %3 = llvm.mlir.addressof @const_alias : !llvm.ptr
}

// Define the global.
llvm.mlir.global @const(42 : i32) : i32

// Define an alias.
llvm.mlir.alias @const_alias : i32 {
  %0 = llvm.mlir.addressof @const : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}
```
"""
function mlir_addressof(; res::IR.Type, global_name, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("global_name", global_name), ]
    
    create_operation(
        "llvm.mlir.addressof", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mlir_alias`

`llvm.mlir.alias` is a top level operation that defines a global alias for
global variables and functions. The operation is always initialized by
using a initializer region which could be a direct map to another global
value or contain some address computation on top of it.

It uses a symbol for its value, which will be uniqued by the module
with respect to other symbols in it.

Similarly to functions and globals, they can also have a linkage attribute.
This attribute is placed between `llvm.mlir.alias` and the symbol name. If
the attribute is omitted, `external` linkage is assumed by default.

Examples:

```mlir
// Global alias use @-identifiers.
llvm.mlir.alias external @foo_alias {addr_space = 0 : i32} : !llvm.ptr {
  %0 = llvm.mlir.addressof @some_function : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// More complex initialization.
llvm.mlir.alias linkonce_odr hidden @glob
{addr_space = 0 : i32, dso_local} : !llvm.array<32 x i32> {
  %0 = llvm.mlir.constant(1234 : i64) : i64
  %1 = llvm.mlir.addressof @glob.private : !llvm.ptr
  %2 = llvm.ptrtoint %1 : !llvm.ptr to i64
  %3 = llvm.add %2, %0 : i64
  %4 = llvm.inttoptr %3 : i64 to !llvm.ptr
  llvm.return %4 : !llvm.ptr
}
```
"""
function mlir_alias(; alias_type, sym_name, linkage, dso_local=nothing, thread_local_=nothing, unnamed_addr=nothing, visibility_=nothing, initializer::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[initializer, ]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("alias_type", alias_type), NamedAttribute("sym_name", sym_name), NamedAttribute("linkage", linkage), ]
    !isnothing(dso_local) && push!(attributes, NamedAttribute("dso_local", dso_local))
    !isnothing(thread_local_) && push!(attributes, NamedAttribute("thread_local_", thread_local_))
    !isnothing(unnamed_addr) && push!(attributes, NamedAttribute("unnamed_addr", unnamed_addr))
    !isnothing(visibility_) && push!(attributes, NamedAttribute("visibility_", visibility_))
    
    create_operation(
        "llvm.mlir.alias", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function alloca(arraySize::Value; res::IR.Type, alignment=nothing, elem_type, inalloca=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arraySize, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("elem_type", elem_type), ]
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))
    !isnothing(inalloca) && push!(attributes, NamedAttribute("inalloca", inalloca))
    
    create_operation(
        "llvm.alloca", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function and(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.and", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function cmpxchg(ptr::Value, cmp::Value, val::Value; res=nothing::Union{Nothing, IR.Type}, success_ordering, failure_ordering, syncscope=nothing, alignment=nothing, weak=nothing, volatile_=nothing, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, cmp, val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("success_ordering", success_ordering), NamedAttribute("failure_ordering", failure_ordering), ]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(syncscope) && push!(attributes, NamedAttribute("syncscope", syncscope))
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))
    !isnothing(weak) && push!(attributes, NamedAttribute("weak", weak))
    !isnothing(volatile_) && push!(attributes, NamedAttribute("volatile_", volatile_))
    !isnothing(access_groups) && push!(attributes, NamedAttribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "llvm.cmpxchg", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function atomicrmw(ptr::Value, val::Value; res=nothing::Union{Nothing, IR.Type}, bin_op, ordering, syncscope=nothing, alignment=nothing, volatile_=nothing, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("bin_op", bin_op), NamedAttribute("ordering", ordering), ]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(syncscope) && push!(attributes, NamedAttribute("syncscope", syncscope))
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, NamedAttribute("volatile_", volatile_))
    !isnothing(access_groups) && push!(attributes, NamedAttribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "llvm.atomicrmw", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function bitcast(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.bitcast", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`blockaddress`

Creates an SSA value containing a pointer to a basic block. The block
address information (function and block) is given by the `BlockAddressAttr`
attribute. This operation assumes an existing `llvm.blocktag` operation
identifying an existing MLIR block within a function. Example:

```mlir
llvm.mlir.global private @g() : !llvm.ptr {
  %0 = llvm.blockaddress <function = @fn, tag = <id = 0>> : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.func @fn() {
  llvm.br ^bb1
^bb1:  // pred: ^bb0
  llvm.blocktag <id = 0>
  llvm.return
}
```
"""
function blockaddress(; res::IR.Type, block_addr, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("block_addr", block_addr), ]
    
    create_operation(
        "llvm.blockaddress", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`blocktag`

This operation uses a `tag` to uniquely identify an MLIR block in a
function. The same tag is used by `llvm.blockaddress` in order to compute
the target address.

A given function should have at most one `llvm.blocktag` operation with a
given `tag`. This operation cannot be used as a terminator.

# Example

```mlir
llvm.func @f() -> !llvm.ptr {
  %addr = llvm.blockaddress <function = @f, tag = <id = 1>> : !llvm.ptr
  llvm.br ^bb1
^bb1:
  llvm.blocktag <id = 1>
  llvm.return %addr : !llvm.ptr
}
```
"""
function blocktag(; tag, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("tag", tag), ]
    
    create_operation(
        "llvm.blocktag", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function br(destOperands::Vector{Value}; loop_annotation=nothing, dest::Block, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[destOperands..., ]
    owned_regions = Region[]
    successors = Block[dest, ]
    attributes = NamedAttribute[]
    !isnothing(loop_annotation) && push!(attributes, NamedAttribute("loop_annotation", loop_annotation))
    
    create_operation(
        "llvm.br", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`call_intrinsic`

Call the specified llvm intrinsic. If the intrinsic is overloaded, use
the MLIR function type of this op to determine which intrinsic to call.
"""
function call_intrinsic(args::Vector{Value}, op_bundle_operands::Vector{Value}; results=nothing::Union{Nothing, IR.Type}, intrin, fastmathFlags=nothing, op_bundle_sizes, op_bundle_tags=nothing, arg_attrs=nothing, res_attrs=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[args..., op_bundle_operands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("intrin", intrin), NamedAttribute("op_bundle_sizes", op_bundle_sizes), ]
    push!(attributes, operandsegmentsizes([length(args), length(op_bundle_operands), ]))
    !isnothing(results) && push!(op_ty_results, results)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    !isnothing(op_bundle_tags) && push!(attributes, NamedAttribute("op_bundle_tags", op_bundle_tags))
    !isnothing(arg_attrs) && push!(attributes, NamedAttribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, NamedAttribute("res_attrs", res_attrs))
    
    create_operation(
        "llvm.call_intrinsic", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`call`

In LLVM IR, functions may return either 0 or 1 value. LLVM IR dialect
implements this behavior by providing a variadic `call` operation for 0- and
1-result functions. Even though MLIR supports multi-result functions, LLVM
IR dialect disallows them.

The `call` instruction supports both direct and indirect calls. Direct calls
start with a function name (`@`-prefixed) and indirect calls start with an
SSA value (`%`-prefixed). The direct callee, if present, is stored as a
function attribute `callee`. For indirect calls, the callee is of `!llvm.ptr` type
and is stored as the first value in `callee_operands`. If and only if the
callee is a variadic function, the `var_callee_type` attribute must carry
the variadic LLVM function type. The trailing type list contains the
optional indirect callee type and the MLIR function type, which differs from
the LLVM function type that uses an explicit void type to model functions
that do not return a value.

If this operatin has the `no_inline` attribute, then this specific function call
will never be inlined. The opposite behavior will occur if the call has `always_inline`
attribute. The `inline_hint` attribute indicates that it is desirable to inline
this function call.

Examples:

```mlir
// Direct call without arguments and with one result.
%0 = llvm.call @foo() : () -> (f32)

// Direct call with arguments and without a result.
llvm.call @bar(%0) : (f32) -> ()

// Indirect call with an argument and without a result.
%1 = llvm.mlir.addressof @foo : !llvm.ptr
llvm.call %1(%0) : !llvm.ptr, (f32) -> ()

// Direct variadic call.
llvm.call @printf(%0, %1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32

// Indirect variadic call
llvm.call %1(%0) vararg(!llvm.func<void (...)>) : !llvm.ptr, (i32) -> ()
```
"""
function call(callee_operands::Vector{Value}, op_bundle_operands::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, var_callee_type=nothing, callee=nothing, fastmathFlags=nothing, CConv=nothing, TailCallKind=nothing, memory_effects=nothing, convergent=nothing, no_unwind=nothing, will_return=nothing, op_bundle_sizes, op_bundle_tags=nothing, arg_attrs=nothing, res_attrs=nothing, no_inline=nothing, always_inline=nothing, inline_hint=nothing, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[callee_operands..., op_bundle_operands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("op_bundle_sizes", op_bundle_sizes), ]
    push!(attributes, operandsegmentsizes([length(callee_operands), length(op_bundle_operands), ]))
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(var_callee_type) && push!(attributes, NamedAttribute("var_callee_type", var_callee_type))
    !isnothing(callee) && push!(attributes, NamedAttribute("callee", callee))
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    !isnothing(CConv) && push!(attributes, NamedAttribute("CConv", CConv))
    !isnothing(TailCallKind) && push!(attributes, NamedAttribute("TailCallKind", TailCallKind))
    !isnothing(memory_effects) && push!(attributes, NamedAttribute("memory_effects", memory_effects))
    !isnothing(convergent) && push!(attributes, NamedAttribute("convergent", convergent))
    !isnothing(no_unwind) && push!(attributes, NamedAttribute("no_unwind", no_unwind))
    !isnothing(will_return) && push!(attributes, NamedAttribute("will_return", will_return))
    !isnothing(op_bundle_tags) && push!(attributes, NamedAttribute("op_bundle_tags", op_bundle_tags))
    !isnothing(arg_attrs) && push!(attributes, NamedAttribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, NamedAttribute("res_attrs", res_attrs))
    !isnothing(no_inline) && push!(attributes, NamedAttribute("no_inline", no_inline))
    !isnothing(always_inline) && push!(attributes, NamedAttribute("always_inline", always_inline))
    !isnothing(inline_hint) && push!(attributes, NamedAttribute("inline_hint", inline_hint))
    !isnothing(access_groups) && push!(attributes, NamedAttribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "llvm.call", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`comdat`

Provides access to object file COMDAT section/group functionality.

Examples:
```mlir
llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @any any
}
llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
```
"""
function comdat(; sym_name, body::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), ]
    
    create_operation(
        "llvm.comdat", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`comdat_selector`

Provides access to object file COMDAT section/group functionality.

Examples:
```mlir
llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @any any
}
llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
```
"""
function comdat_selector(; sym_name, comdat, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), NamedAttribute("comdat", comdat), ]
    
    create_operation(
        "llvm.comdat_selector", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function cond_br(condition::Value, trueDestOperands::Vector{Value}, falseDestOperands::Vector{Value}; branch_weights=nothing, loop_annotation=nothing, trueDest::Block, falseDest::Block, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[condition, trueDestOperands..., falseDestOperands..., ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([1, length(trueDestOperands), length(falseDestOperands), ]))
    !isnothing(branch_weights) && push!(attributes, NamedAttribute("branch_weights", branch_weights))
    !isnothing(loop_annotation) && push!(attributes, NamedAttribute("loop_annotation", loop_annotation))
    
    create_operation(
        "llvm.cond_br", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mlir_constant`

Unlike LLVM IR, MLIR does not have first-class constant values. Therefore,
all constants must be created as SSA values before being used in other
operations. `llvm.mlir.constant` creates such values for scalars, vectors,
strings, structs, and array of structs. It has a mandatory `value` attribute
whose type depends on the type of the constant value. The type of the constant
value must correspond to the attribute type converted to LLVM IR type.

When creating constant scalars, the `value` attribute must be either an
integer attribute or a floating point attribute. The type of the attribute
may be omitted for `i64` and `f64` types that are implied.

When creating constant vectors, the `value` attribute must be either an
array attribute, a dense attribute, or a sparse attribute that contains
integers or floats. The number of elements in the result vector must match
the number of elements in the attribute.

When creating constant strings, the `value` attribute must be a string
attribute. The type of the constant must be an LLVM array of `i8`s, and the
length of the array must match the length of the attribute.

When creating constant structs, the `value` attribute must be an array
attribute that contains integers or floats. The type of the constant must be
an LLVM struct type. The number of fields in the struct must match the
number of elements in the attribute, and the type of each LLVM struct field
must correspond to the type of the corresponding attribute element converted
to LLVM IR.

When creating an array of structs, the `value` attribute must be an array
attribute, itself containing zero, or undef, or array attributes for each
potential nested array type, and the elements of the leaf array attributes
for must match the struct element types or be zero or undef attributes.

Examples:

```mlir
// Integer constant, internal i32 is mandatory
%0 = llvm.mlir.constant(42 : i32) : i32

// It\'s okay to omit i64.
%1 = llvm.mlir.constant(42) : i64

// Floating point constant.
%2 = llvm.mlir.constant(42.0 : f32) : f32

// Splat dense vector constant.
%3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : vector<4xf32>
```
"""
function mlir_constant(; res::IR.Type, value, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("value", value), ]
    
    create_operation(
        "llvm.mlir.constant", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`dso_local_equivalent`

Creates an SSA value containing a pointer to a global value (function or
alias to function). It represents a function which is functionally
equivalent to a given function, but is always defined in the current
linkage unit. The target function may not have `extern_weak` linkage.

Examples:

```mlir
llvm.mlir.global external constant @const() : i64 {
  %0 = llvm.mlir.addressof @const : !llvm.ptr
  %1 = llvm.ptrtoint %0 : !llvm.ptr to i64
  %2 = llvm.dso_local_equivalent @func : !llvm.ptr
  %4 = llvm.ptrtoint %2 : !llvm.ptr to i64
  llvm.return %4 : i64
}
```
"""
function dso_local_equivalent(; res::IR.Type, function_name, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("function_name", function_name), ]
    
    create_operation(
        "llvm.dso_local_equivalent", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function extractelement(vector::Value, position::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vector, position, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.extractelement", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function extractvalue(container::Value; res::IR.Type, position, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[container, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("position", position), ]
    
    create_operation(
        "llvm.extractvalue", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function fadd(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.fadd", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function fcmp(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, predicate, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("predicate", predicate), ]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.fcmp", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function fdiv(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.fdiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function fmul(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.fmul", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function fneg(operand::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.fneg", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function fpext(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.fpext", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function fptosi(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.fptosi", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function fptoui(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.fptoui", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function fptrunc(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.fptrunc", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function frem(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.frem", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function fsub(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.fsub", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function fence(; ordering, syncscope=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("ordering", ordering), ]
    !isnothing(syncscope) && push!(attributes, NamedAttribute("syncscope", syncscope))
    
    create_operation(
        "llvm.fence", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function freeze(val::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.freeze", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`getelementptr`

This operation mirrors LLVM IRs \'getelementptr\' operation that is used to
perform pointer arithmetic.

Like in LLVM IR, it is possible to use both constants as well as SSA values
as indices. In the case of indexing within a structure, it is required to
either use constant indices directly, or supply a constant SSA value.

The no-wrap flags can be used to specify the low-level pointer arithmetic
overflow behavior that LLVM uses after lowering the operation to LLVM IR.
Valid options include \'inbounds\' (pointer arithmetic must be within object
bounds), \'nusw\' (no unsigned signed wrap), and \'nuw\' (no unsigned wrap).
Note that \'inbounds\' implies \'nusw\' which is ensured by the enum
definition. The flags can be set individually or in combination.

Examples:

```mlir
// GEP with an SSA value offset
%0 = llvm.getelementptr %1[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32

// GEP with a constant offset and the inbounds attribute set
%0 = llvm.getelementptr inbounds %1[3] : (!llvm.ptr) -> !llvm.ptr, f32

// GEP with constant offsets into a structure
%0 = llvm.getelementptr %1[0, 1]
   : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f32)>
```
"""
function getelementptr(base::Value, dynamicIndices::Vector{Value}; res::IR.Type, rawConstantIndices, elem_type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[base, dynamicIndices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("rawConstantIndices", rawConstantIndices), NamedAttribute("elem_type", elem_type), ]
    
    create_operation(
        "llvm.getelementptr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mlir_global_ctors`

Specifies a list of constructor functions, priorities, and associated data.
The functions referenced by this array will be called in ascending order
of priority (i.e. lowest first) when the module is loaded. The order of
functions with the same priority is not defined. This operation is
translated to LLVM\'s global_ctors global variable. The initializer
functions are run at load time. However, if the associated data is not
`#llvm.zero`, functions only run if the data is not discarded.

Examples:

```mlir
llvm.func @ctor() {
  ...
  llvm.return
}
llvm.mlir.global_ctors ctors = [@ctor], priorities = [0],
                               data = [#llvm.zero]
```
"""
function mlir_global_ctors(; ctors, priorities, data, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("ctors", ctors), NamedAttribute("priorities", priorities), NamedAttribute("data", data), ]
    
    create_operation(
        "llvm.mlir.global_ctors", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mlir_global_dtors`

Specifies a list of destructor functions and priorities. The functions
referenced by this array will be called in descending order of priority
(i.e. highest first) when the module is unloaded. The order of functions
with the same priority is not defined. This operation is translated to
LLVM\'s global_dtors global variable. The destruction functions are run at
load time. However, if the associated data is not `#llvm.zero`, functions
only run if the data is not discarded.

Examples:

```mlir
llvm.func @dtor() {
  llvm.return
}
llvm.mlir.global_dtors dtors = [@dtor], priorities = [0],
                               data = [#llvm.zero]
```
"""
function mlir_global_dtors(; dtors, priorities, data, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dtors", dtors), NamedAttribute("priorities", priorities), NamedAttribute("data", data), ]
    
    create_operation(
        "llvm.mlir.global_dtors", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mlir_global`

Since MLIR allows for arbitrary operations to be present at the top level,
global variables are defined using the `llvm.mlir.global` operation. Both
global constants and variables can be defined, and the value may also be
initialized in both cases.

There are two forms of initialization syntax. Simple constants that can be
represented as MLIR attributes can be given in-line:

```mlir
llvm.mlir.global @variable(32.0 : f32) : f32
```

This initialization and type syntax is similar to `llvm.mlir.constant` and
may use two types: one for MLIR attribute and another for the LLVM value.
These types must be compatible.

More complex constants that cannot be represented as MLIR attributes can be
given in an initializer region:

```mlir
// This global is initialized with the equivalent of:
//   i32* getelementptr (i32* @g2, i32 2)
llvm.mlir.global constant @int_gep() : !llvm.ptr {
  %0 = llvm.mlir.addressof @g2 : !llvm.ptr
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.getelementptr %0[%1]
     : (!llvm.ptr, i32) -> !llvm.ptr, i32
  // The initializer region must end with `llvm.return`.
  llvm.return %2 : !llvm.ptr
}
```

Only one of the initializer attribute or initializer region may be provided.

`llvm.mlir.global` must appear at top-level of the enclosing module. It uses
an @-identifier for its value, which will be uniqued by the module with
respect to other @-identifiers in it.

Examples:

```mlir
// Global values use @-identifiers.
llvm.mlir.global constant @cst(42 : i32) : i32

// Non-constant values must also be initialized.
llvm.mlir.global @variable(32.0 : f32) : f32

// Strings are expected to be of wrapped LLVM i8 array type and do not
// automatically include the trailing zero.
llvm.mlir.global @string(\"abc\") : !llvm.array<3 x i8>

// For strings globals, the trailing type may be omitted.
llvm.mlir.global constant @no_trailing_type(\"foo bar\")

// A complex initializer is constructed with an initializer region.
llvm.mlir.global constant @int_gep() : !llvm.ptr {
  %0 = llvm.mlir.addressof @g2 : !llvm.ptr
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.getelementptr %0[%1]
     : (!llvm.ptr, i32) -> !llvm.ptr, i32
  llvm.return %2 : !llvm.ptr
}
```

Similarly to functions, globals have a linkage attribute. In the custom
syntax, this attribute is placed between `llvm.mlir.global` and the optional
`constant` keyword. If the attribute is omitted, `external` linkage is
assumed by default.

Examples:

```mlir
// A constant with internal linkage will not participate in linking.
llvm.mlir.global internal constant @cst(42 : i32) : i32

// By default, \"external\" linkage is assumed and the global participates in
// symbol resolution at link-time.
llvm.mlir.global @glob(0 : f32) : f32

// Alignment is optional
llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) : !llvm.array<8 x f32>
```

Like global variables in LLVM IR, globals can have an (optional)
alignment attribute using keyword `alignment`. The integer value of the
alignment must be a positive integer that is a power of 2.

Examples:

```mlir
// Alignment is optional
llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) { alignment = 32 : i64 } : !llvm.array<8 x f32>
```
"""
function mlir_global(; global_type, constant=nothing, sym_name, linkage, dso_local=nothing, thread_local_=nothing, externally_initialized=nothing, value=nothing, alignment=nothing, addr_space=nothing, unnamed_addr=nothing, section=nothing, comdat=nothing, dbg_exprs=nothing, visibility_=nothing, initializer::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[initializer, ]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("global_type", global_type), NamedAttribute("sym_name", sym_name), NamedAttribute("linkage", linkage), ]
    !isnothing(constant) && push!(attributes, NamedAttribute("constant", constant))
    !isnothing(dso_local) && push!(attributes, NamedAttribute("dso_local", dso_local))
    !isnothing(thread_local_) && push!(attributes, NamedAttribute("thread_local_", thread_local_))
    !isnothing(externally_initialized) && push!(attributes, NamedAttribute("externally_initialized", externally_initialized))
    !isnothing(value) && push!(attributes, NamedAttribute("value", value))
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))
    !isnothing(addr_space) && push!(attributes, NamedAttribute("addr_space", addr_space))
    !isnothing(unnamed_addr) && push!(attributes, NamedAttribute("unnamed_addr", unnamed_addr))
    !isnothing(section) && push!(attributes, NamedAttribute("section", section))
    !isnothing(comdat) && push!(attributes, NamedAttribute("comdat", comdat))
    !isnothing(dbg_exprs) && push!(attributes, NamedAttribute("dbg_exprs", dbg_exprs))
    !isnothing(visibility_) && push!(attributes, NamedAttribute("visibility_", visibility_))
    
    create_operation(
        "llvm.mlir.global", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function icmp(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, predicate, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("predicate", predicate), ]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.icmp", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`indirectbr`

Transfer control flow to address in `\$addr`. A list of possible target
blocks in `\$successors` can be provided and maybe used as a hint in LLVM:

```mlir
...
llvm.func @g(...
  %dest = llvm.blockaddress <function = @g, tag = <id = 0>> : !llvm.ptr
  llvm.indirectbr %dest : !llvm.ptr, [
    ^head
  ]
^head:
  llvm.blocktag <id = 0>
  llvm.return %arg0 : i32
  ...
```

It also supports a list of operands that can be passed to a target block:

```mlir
  llvm.indirectbr %dest : !llvm.ptr, [
    ^head(%arg0 : i32),
    ^tail(%arg1, %arg0 : i32, i32)
  ]
^head(%r0 : i32):
  llvm.return %r0 : i32
^tail(%r1 : i32, %r2 : i32):
  ...
```
"""
function indirectbr(addr::Value, succOperands::Vector{Value}; indbr_operand_segments, successors::Vector{Block}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, succOperands..., ]
    owned_regions = Region[]
    successors = Block[successors..., ]
    attributes = NamedAttribute[NamedAttribute("indbr_operand_segments", indbr_operand_segments), ]
    
    create_operation(
        "llvm.indirectbr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`inline_asm`

The InlineAsmOp mirrors the underlying LLVM semantics with a notable
exception: the embedded `asm_string` is not allowed to define or reference
any symbol or any global variable: only the operands of the op may be read,
written, or referenced.
Attempting to define or reference any symbol or any global behavior is
considered undefined behavior at this time.
If `tail_call_kind` is used, the operation behaves like the specified
tail call kind. The `musttail` kind it\'s not available for this operation,
since it isn\'t supported by LLVM\'s inline asm.
"""
function inline_asm(operands::Vector{Value}; res=nothing::Union{Nothing, IR.Type}, asm_string, constraints, has_side_effects=nothing, is_align_stack=nothing, tail_call_kind=nothing, asm_dialect=nothing, operand_attrs=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("asm_string", asm_string), NamedAttribute("constraints", constraints), ]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(has_side_effects) && push!(attributes, NamedAttribute("has_side_effects", has_side_effects))
    !isnothing(is_align_stack) && push!(attributes, NamedAttribute("is_align_stack", is_align_stack))
    !isnothing(tail_call_kind) && push!(attributes, NamedAttribute("tail_call_kind", tail_call_kind))
    !isnothing(asm_dialect) && push!(attributes, NamedAttribute("asm_dialect", asm_dialect))
    !isnothing(operand_attrs) && push!(attributes, NamedAttribute("operand_attrs", operand_attrs))
    
    create_operation(
        "llvm.inline_asm", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function insertelement(vector::Value, value::Value, position::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vector, value, position, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.insertelement", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function insertvalue(container::Value, value::Value; res=nothing::Union{Nothing, IR.Type}, position, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[container, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("position", position), ]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.insertvalue", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function inttoptr(arg::Value; res::IR.Type, dereferenceable=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(dereferenceable) && push!(attributes, NamedAttribute("dereferenceable", dereferenceable))
    
    create_operation(
        "llvm.inttoptr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function invoke(callee_operands::Vector{Value}, normalDestOperands::Vector{Value}, unwindDestOperands::Vector{Value}, op_bundle_operands::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, var_callee_type=nothing, callee=nothing, arg_attrs=nothing, res_attrs=nothing, branch_weights=nothing, CConv=nothing, op_bundle_sizes, op_bundle_tags=nothing, normalDest::Block, unwindDest::Block, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[callee_operands..., normalDestOperands..., unwindDestOperands..., op_bundle_operands..., ]
    owned_regions = Region[]
    successors = Block[normalDest, unwindDest, ]
    attributes = NamedAttribute[NamedAttribute("op_bundle_sizes", op_bundle_sizes), ]
    push!(attributes, operandsegmentsizes([length(callee_operands), length(normalDestOperands), length(unwindDestOperands), length(op_bundle_operands), ]))
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(var_callee_type) && push!(attributes, NamedAttribute("var_callee_type", var_callee_type))
    !isnothing(callee) && push!(attributes, NamedAttribute("callee", callee))
    !isnothing(arg_attrs) && push!(attributes, NamedAttribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, NamedAttribute("res_attrs", res_attrs))
    !isnothing(branch_weights) && push!(attributes, NamedAttribute("branch_weights", branch_weights))
    !isnothing(CConv) && push!(attributes, NamedAttribute("CConv", CConv))
    !isnothing(op_bundle_tags) && push!(attributes, NamedAttribute("op_bundle_tags", op_bundle_tags))
    
    create_operation(
        "llvm.invoke", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`func`

MLIR functions are defined by an operation that is not built into the IR
itself. The LLVM dialect provides an `llvm.func` operation to define
functions compatible with LLVM IR. These functions have LLVM dialect
function type but use MLIR syntax to express it. They are required to have
exactly one result type. LLVM function operation is intended to capture
additional properties of LLVM functions, such as linkage and calling
convention, that may be modeled differently by the built-in MLIR function.

```mlir
// The type of @bar is !llvm<\"i64 (i64)\">
llvm.func @bar(%arg0: i64) -> i64 {
  llvm.return %arg0 : i64
}

// Type type of @foo is !llvm<\"void (i64)\">
// !llvm.void type is omitted
llvm.func @foo(%arg0: i64) {
  llvm.return
}

// A function with `internal` linkage.
llvm.func internal @internal_func() {
  llvm.return
}
```
"""
function func(; sym_name, sym_visibility=nothing, function_type, linkage=nothing, dso_local=nothing, CConv=nothing, comdat=nothing, convergent=nothing, personality=nothing, garbageCollector=nothing, passthrough=nothing, arg_attrs=nothing, res_attrs=nothing, function_entry_count=nothing, memory_effects=nothing, visibility_=nothing, arm_streaming=nothing, arm_locally_streaming=nothing, arm_streaming_compatible=nothing, arm_new_za=nothing, arm_in_za=nothing, arm_out_za=nothing, arm_inout_za=nothing, arm_preserves_za=nothing, section=nothing, unnamed_addr=nothing, alignment=nothing, vscale_range=nothing, frame_pointer=nothing, target_cpu=nothing, tune_cpu=nothing, reciprocal_estimates=nothing, prefer_vector_width=nothing, target_features=nothing, unsafe_fp_math=nothing, no_infs_fp_math=nothing, no_nans_fp_math=nothing, approx_func_fp_math=nothing, no_signed_zeros_fp_math=nothing, denormal_fp_math=nothing, denormal_fp_math_f32=nothing, fp_contract=nothing, instrument_function_entry=nothing, instrument_function_exit=nothing, no_inline=nothing, always_inline=nothing, no_unwind=nothing, will_return=nothing, optimize_none=nothing, vec_type_hint=nothing, work_group_size_hint=nothing, reqd_work_group_size=nothing, intel_reqd_sub_group_size=nothing, uwtable_kind=nothing, body::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), NamedAttribute("function_type", function_type), ]
    !isnothing(sym_visibility) && push!(attributes, NamedAttribute("sym_visibility", sym_visibility))
    !isnothing(linkage) && push!(attributes, NamedAttribute("linkage", linkage))
    !isnothing(dso_local) && push!(attributes, NamedAttribute("dso_local", dso_local))
    !isnothing(CConv) && push!(attributes, NamedAttribute("CConv", CConv))
    !isnothing(comdat) && push!(attributes, NamedAttribute("comdat", comdat))
    !isnothing(convergent) && push!(attributes, NamedAttribute("convergent", convergent))
    !isnothing(personality) && push!(attributes, NamedAttribute("personality", personality))
    !isnothing(garbageCollector) && push!(attributes, NamedAttribute("garbageCollector", garbageCollector))
    !isnothing(passthrough) && push!(attributes, NamedAttribute("passthrough", passthrough))
    !isnothing(arg_attrs) && push!(attributes, NamedAttribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, NamedAttribute("res_attrs", res_attrs))
    !isnothing(function_entry_count) && push!(attributes, NamedAttribute("function_entry_count", function_entry_count))
    !isnothing(memory_effects) && push!(attributes, NamedAttribute("memory_effects", memory_effects))
    !isnothing(visibility_) && push!(attributes, NamedAttribute("visibility_", visibility_))
    !isnothing(arm_streaming) && push!(attributes, NamedAttribute("arm_streaming", arm_streaming))
    !isnothing(arm_locally_streaming) && push!(attributes, NamedAttribute("arm_locally_streaming", arm_locally_streaming))
    !isnothing(arm_streaming_compatible) && push!(attributes, NamedAttribute("arm_streaming_compatible", arm_streaming_compatible))
    !isnothing(arm_new_za) && push!(attributes, NamedAttribute("arm_new_za", arm_new_za))
    !isnothing(arm_in_za) && push!(attributes, NamedAttribute("arm_in_za", arm_in_za))
    !isnothing(arm_out_za) && push!(attributes, NamedAttribute("arm_out_za", arm_out_za))
    !isnothing(arm_inout_za) && push!(attributes, NamedAttribute("arm_inout_za", arm_inout_za))
    !isnothing(arm_preserves_za) && push!(attributes, NamedAttribute("arm_preserves_za", arm_preserves_za))
    !isnothing(section) && push!(attributes, NamedAttribute("section", section))
    !isnothing(unnamed_addr) && push!(attributes, NamedAttribute("unnamed_addr", unnamed_addr))
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))
    !isnothing(vscale_range) && push!(attributes, NamedAttribute("vscale_range", vscale_range))
    !isnothing(frame_pointer) && push!(attributes, NamedAttribute("frame_pointer", frame_pointer))
    !isnothing(target_cpu) && push!(attributes, NamedAttribute("target_cpu", target_cpu))
    !isnothing(tune_cpu) && push!(attributes, NamedAttribute("tune_cpu", tune_cpu))
    !isnothing(reciprocal_estimates) && push!(attributes, NamedAttribute("reciprocal_estimates", reciprocal_estimates))
    !isnothing(prefer_vector_width) && push!(attributes, NamedAttribute("prefer_vector_width", prefer_vector_width))
    !isnothing(target_features) && push!(attributes, NamedAttribute("target_features", target_features))
    !isnothing(unsafe_fp_math) && push!(attributes, NamedAttribute("unsafe_fp_math", unsafe_fp_math))
    !isnothing(no_infs_fp_math) && push!(attributes, NamedAttribute("no_infs_fp_math", no_infs_fp_math))
    !isnothing(no_nans_fp_math) && push!(attributes, NamedAttribute("no_nans_fp_math", no_nans_fp_math))
    !isnothing(approx_func_fp_math) && push!(attributes, NamedAttribute("approx_func_fp_math", approx_func_fp_math))
    !isnothing(no_signed_zeros_fp_math) && push!(attributes, NamedAttribute("no_signed_zeros_fp_math", no_signed_zeros_fp_math))
    !isnothing(denormal_fp_math) && push!(attributes, NamedAttribute("denormal_fp_math", denormal_fp_math))
    !isnothing(denormal_fp_math_f32) && push!(attributes, NamedAttribute("denormal_fp_math_f32", denormal_fp_math_f32))
    !isnothing(fp_contract) && push!(attributes, NamedAttribute("fp_contract", fp_contract))
    !isnothing(instrument_function_entry) && push!(attributes, NamedAttribute("instrument_function_entry", instrument_function_entry))
    !isnothing(instrument_function_exit) && push!(attributes, NamedAttribute("instrument_function_exit", instrument_function_exit))
    !isnothing(no_inline) && push!(attributes, NamedAttribute("no_inline", no_inline))
    !isnothing(always_inline) && push!(attributes, NamedAttribute("always_inline", always_inline))
    !isnothing(no_unwind) && push!(attributes, NamedAttribute("no_unwind", no_unwind))
    !isnothing(will_return) && push!(attributes, NamedAttribute("will_return", will_return))
    !isnothing(optimize_none) && push!(attributes, NamedAttribute("optimize_none", optimize_none))
    !isnothing(vec_type_hint) && push!(attributes, NamedAttribute("vec_type_hint", vec_type_hint))
    !isnothing(work_group_size_hint) && push!(attributes, NamedAttribute("work_group_size_hint", work_group_size_hint))
    !isnothing(reqd_work_group_size) && push!(attributes, NamedAttribute("reqd_work_group_size", reqd_work_group_size))
    !isnothing(intel_reqd_sub_group_size) && push!(attributes, NamedAttribute("intel_reqd_sub_group_size", intel_reqd_sub_group_size))
    !isnothing(uwtable_kind) && push!(attributes, NamedAttribute("uwtable_kind", uwtable_kind))
    
    create_operation(
        "llvm.func", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function lshr(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, isExact=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(isExact) && push!(attributes, NamedAttribute("isExact", isExact))
    
    create_operation(
        "llvm.lshr", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function landingpad(operand_0::Vector{Value}; res::IR.Type, cleanup=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(cleanup) && push!(attributes, NamedAttribute("cleanup", cleanup))
    
    create_operation(
        "llvm.landingpad", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`linker_options`

Pass the given options to the linker when the resulting object file is linked.
This is used extensively on Windows to determine the C runtime that the object
files should link against.

Examples:
```mlir
// Link against the MSVC static threaded CRT.
llvm.linker_options [\"/DEFAULTLIB:\", \"libcmt\"]

// Link against aarch64 compiler-rt builtins
llvm.linker_options [\"-l\", \"clang_rt.builtins-aarch64\"]
```
"""
function linker_options(; options, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("options", options), ]
    
    create_operation(
        "llvm.linker_options", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`load`

The `load` operation is used to read from memory. A load may be marked as
atomic, volatile, and/or nontemporal, and takes a number of optional
attributes that specify aliasing information.

An atomic load only supports a limited set of pointer, integer, and
floating point types, and requires an explicit alignment.

Examples:
```mlir
// A volatile load of a float variable.
%0 = llvm.load volatile %ptr : !llvm.ptr -> f32

// A nontemporal load of a float variable.
%0 = llvm.load %ptr {nontemporal} : !llvm.ptr -> f32

// An atomic load of an integer variable.
%0 = llvm.load %ptr atomic monotonic {alignment = 8 : i64}
    : !llvm.ptr -> i64
```

See the following link for more details:
https://llvm.org/docs/LangRef.html#load-instruction
"""
function load(addr::Value; res::IR.Type, alignment=nothing, volatile_=nothing, nontemporal=nothing, invariant=nothing, invariantGroup=nothing, ordering=nothing, syncscope=nothing, dereferenceable=nothing, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, NamedAttribute("volatile_", volatile_))
    !isnothing(nontemporal) && push!(attributes, NamedAttribute("nontemporal", nontemporal))
    !isnothing(invariant) && push!(attributes, NamedAttribute("invariant", invariant))
    !isnothing(invariantGroup) && push!(attributes, NamedAttribute("invariantGroup", invariantGroup))
    !isnothing(ordering) && push!(attributes, NamedAttribute("ordering", ordering))
    !isnothing(syncscope) && push!(attributes, NamedAttribute("syncscope", syncscope))
    !isnothing(dereferenceable) && push!(attributes, NamedAttribute("dereferenceable", dereferenceable))
    !isnothing(access_groups) && push!(attributes, NamedAttribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "llvm.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`module_flags`

Represents the equivalent in MLIR for LLVM\'s `llvm.module.flags` metadata,
which requires a list of metadata triplets. Each triplet entry is described
by a `ModuleFlagAttr`.

# Example
```mlir
llvm.module.flags [
  #llvm.mlir.module_flag<error, \"wchar_size\", 4>,
  #llvm.mlir.module_flag<max, \"PIC Level\", 2>
]
```
"""
function module_flags(; flags, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("flags", flags), ]
    
    create_operation(
        "llvm.module_flags", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mul(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.mul", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`mlir_none`

Unlike LLVM IR, MLIR does not have first-class token values. They must be
explicitly created as SSA values using `llvm.mlir.none`. This operation has
no operands or attributes, and returns a none token value of a wrapped LLVM IR
pointer type.

Examples:

```mlir
%0 = llvm.mlir.none : !llvm.token
```
"""
function mlir_none(; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.mlir.none", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function or(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, isDisjoint=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(isDisjoint) && push!(attributes, NamedAttribute("isDisjoint", isDisjoint))
    
    create_operation(
        "llvm.or", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`mlir_poison`

Unlike LLVM IR, MLIR does not have first-class poison values. Such values
must be created as SSA values using `llvm.mlir.poison`. This operation has
no operands or attributes. It creates a poison value of the specified LLVM
IR dialect type.

# Example

```mlir
// Create a poison value for a structure with a 32-bit integer followed
// by a float.
%0 = llvm.mlir.poison : !llvm.struct<(i32, f32)>
```
"""
function mlir_poison(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.mlir.poison", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ptrtoint(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.ptrtoint", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function resume(value::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.resume", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function return_(arg=nothing::Union{Nothing, Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(arg) && push!(operands, arg)
    
    create_operation(
        "llvm.return", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function sdiv(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, isExact=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(isExact) && push!(attributes, NamedAttribute("isExact", isExact))
    
    create_operation(
        "llvm.sdiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function sext(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.sext", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function sitofp(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.sitofp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function srem(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.srem", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function select(condition::Value, trueValue::Value, falseValue::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[condition, trueValue, falseValue, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.select", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function shl(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.shl", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function shufflevector(v1::Value, v2::Value; res::IR.Type, mask, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[v1, v2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mask", mask), ]
    
    create_operation(
        "llvm.shufflevector", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`store`

The `store` operation is used to write to memory. A store may be marked as
atomic, volatile, and/or nontemporal, and takes a number of optional
attributes that specify aliasing information.

An atomic store only supports a limited set of pointer, integer, and
floating point types, and requires an explicit alignment.

Examples:
```mlir
// A volatile store of a float variable.
llvm.store volatile %val, %ptr : f32, !llvm.ptr

// A nontemporal store of a float variable.
llvm.store %val, %ptr {nontemporal} : f32, !llvm.ptr

// An atomic store of an integer variable.
llvm.store %val, %ptr atomic monotonic {alignment = 8 : i64}
    : i64, !llvm.ptr
```

See the following link for more details:
https://llvm.org/docs/LangRef.html#store-instruction
"""
function store(value::Value, addr::Value; alignment=nothing, volatile_=nothing, nontemporal=nothing, invariantGroup=nothing, ordering=nothing, syncscope=nothing, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, NamedAttribute("volatile_", volatile_))
    !isnothing(nontemporal) && push!(attributes, NamedAttribute("nontemporal", nontemporal))
    !isnothing(invariantGroup) && push!(attributes, NamedAttribute("invariantGroup", invariantGroup))
    !isnothing(ordering) && push!(attributes, NamedAttribute("ordering", ordering))
    !isnothing(syncscope) && push!(attributes, NamedAttribute("syncscope", syncscope))
    !isnothing(access_groups) && push!(attributes, NamedAttribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "llvm.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function sub(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.sub", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function switch(value::Value, defaultOperands::Vector{Value}, caseOperands::Vector{Value}; case_values=nothing, case_operand_segments, branch_weights=nothing, defaultDestination::Block, caseDestinations::Vector{Block}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, defaultOperands..., caseOperands..., ]
    owned_regions = Region[]
    successors = Block[defaultDestination, caseDestinations..., ]
    attributes = NamedAttribute[NamedAttribute("case_operand_segments", case_operand_segments), ]
    push!(attributes, operandsegmentsizes([1, length(defaultOperands), length(caseOperands), ]))
    !isnothing(case_values) && push!(attributes, NamedAttribute("case_values", case_values))
    !isnothing(branch_weights) && push!(attributes, NamedAttribute("branch_weights", branch_weights))
    
    create_operation(
        "llvm.switch", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function trunc(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.trunc", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function udiv(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, isExact=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(isExact) && push!(attributes, NamedAttribute("isExact", isExact))
    
    create_operation(
        "llvm.udiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function uitofp(arg::Value; res::IR.Type, nonNeg=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(nonNeg) && push!(attributes, NamedAttribute("nonNeg", nonNeg))
    
    create_operation(
        "llvm.uitofp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function urem(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.urem", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`mlir_undef`

Unlike LLVM IR, MLIR does not have first-class undefined values. Such values
must be created as SSA values using `llvm.mlir.undef`. This operation has no
operands or attributes. It creates an undefined value of the specified LLVM
IR dialect type.

# Example

```mlir
// Create a structure with a 32-bit integer followed by a float.
%0 = llvm.mlir.undef : !llvm.struct<(i32, f32)>
```
"""
function mlir_undef(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.mlir.undef", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function unreachable(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.unreachable", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function va_arg(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.va_arg", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function xor(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.xor", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function zext(arg::Value; res::IR.Type, nonNeg=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(nonNeg) && push!(attributes, NamedAttribute("nonNeg", nonNeg))
    
    create_operation(
        "llvm.zext", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mlir_zero`

Unlike LLVM IR, MLIR does not have first-class zero-initialized values.
Such values must be created as SSA values using `llvm.mlir.zero`. This
operation has no operands or attributes. It creates a zero-initialized
value of the specified LLVM IR dialect type.

# Example

```mlir
// Create a zero-initialized value for a structure with a 32-bit integer
// followed by a float.
%0 = llvm.mlir.zero : !llvm.struct<(i32, f32)>
```
"""
function mlir_zero(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.mlir.zero", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes
import ...API



function barrier0(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.barrier0", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`barrier_arrive`

Thread that executes this op announces their arrival at the barrier with 
given id and continue their execution.

The default barrier id is 0 that is similar to `nvvm.barrier` Op. When 
`barrierId` is not present, the default barrier id is used. 

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)
"""
function barrier_arrive(barrierId=nothing::Union{Nothing, Value}; numberOfThreads::Value, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[numberOfThreads, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(barrierId) && push!(operands, barrierId)
    
    create_operation(
        "nvvm.barrier.arrive", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function barrier(barrierId=nothing::Union{Nothing, Value}; numberOfThreads=nothing::Union{Nothing, Value}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(barrierId) && push!(operands, barrierId)
    !isnothing(numberOfThreads) && push!(operands, numberOfThreads)
    push!(attributes, operandsegmentsizes([Int(!isnothing(barrierId)), Int(!isnothing(numberOfThreads)), ]))
    
    create_operation(
        "nvvm.barrier", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_ntid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.ntid.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_ntid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.ntid.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_ntid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.ntid.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_ctaid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.ctaid.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_ctaid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.ctaid.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_ctaid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.ctaid.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_cluster_ctaid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.cluster.ctaid.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_cluster_ctaid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.cluster.ctaid.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_cluster_ctaid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.cluster.ctaid.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`breakpoint`

Breakpoint suspends execution of the program for debugging.
[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-brkpt)
"""
function breakpoint(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.breakpoint", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`st_bulk`

Initializes a region of shared memory at the address given by `addr`.
The `size` operand specifies the number of bytes to initialize and must be 
a multiple of 8.
The `initVal` operand specifies the value to initialize the memory to. The 
only supported value is 0.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-bulk)
"""
function st_bulk(addr::Value, size::Value; initVal=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, size, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(initVal) && push!(attributes, NamedAttribute("initVal", initVal))
    
    create_operation(
        "nvvm.st.bulk", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_clock64(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.clock64", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_clock(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.clock", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cluster_arrive`

The `cluster.arrive` can be used by the threads within the cluster for synchronization and
communication. The `cluster.arrive` instruction marks the warps\' arrival at the barrier
without causing the executing thread to wait for other participating threads.

The `aligned` attribute, when provided, generates the .aligned version of the PTX instruction.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)
"""
function cluster_arrive(; aligned=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(aligned) && push!(attributes, NamedAttribute("aligned", aligned))
    
    create_operation(
        "nvvm.cluster.arrive", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cluster_arrive_relaxed`

The `cluster.arrive` can be used by the threads within the cluster for synchronization and
communication. The `cluster.arrive` instruction marks the warps\' arrival at the barrier
without causing the executing thread to wait for other participating threads.

The `aligned` attribute, when provided, generates the .aligned version of the PTX instruction.
The .relaxed qualifier on `cluster.arrive` specifies that there are no memory
ordering and visibility guarantees provided for the memory accesses performed prior to
`cluster.arrive`.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)
"""
function cluster_arrive_relaxed(; aligned=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(aligned) && push!(attributes, NamedAttribute("aligned", aligned))
    
    create_operation(
        "nvvm.cluster.arrive.relaxed", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_cluster_nctarank(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.cluster.nctarank", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_cluster_nctaid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.cluster.nctaid.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_cluster_nctaid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.cluster.nctaid.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_cluster_nctaid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.cluster.nctaid.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_nclusterid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.nclusterid.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_nclusterid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.nclusterid.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_nclusterid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.nclusterid.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_cluster_ctarank(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.cluster.ctarank", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_clusterid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.clusterid.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_clusterid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.clusterid.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_clusterid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.clusterid.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cluster_wait`

The `cluster.wait` causes the executing thread to wait for all non-exited threads
of the cluster to perform `cluster.arrive`. The `aligned` attribute, when provided,
generates the .aligned version of the PTX instruction.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)
"""
function cluster_wait(; aligned=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(aligned) && push!(attributes, NamedAttribute("aligned", aligned))
    
    create_operation(
        "nvvm.cluster.wait", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`convert_bf16x2_to_f8x2`

This Op converts the given bf16 inputs in a bf16x2 vector to the specified 
f8 type.
The result `dst` is represented as an i16 type or as a vector
of two i8 types.
If `dst` is returned as an i16 type, the converted values from `a`
are packed such that the value converted from the first element of `a`
is stored in the upper 8 bits of `dst` and the value converted from the
second element of `a` is stored in the lower 8 bits of `dst`.
If `dst` is returned as a vector type, each converted value is stored as an 
i8 element in the vector.
The `rnd` and `sat` attributes specify the rounding and saturation modes 
respectively.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
"""
function convert_bf16x2_to_f8x2(a::Value; dst::IR.Type, type, rnd=nothing, sat=nothing, location=Location())
    op_ty_results = IR.Type[dst, ]
    operands = Value[a, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("type", type), ]
    !isnothing(rnd) && push!(attributes, NamedAttribute("rnd", rnd))
    !isnothing(sat) && push!(attributes, NamedAttribute("sat", sat))
    
    create_operation(
        "nvvm.convert.bf16x2.to.f8x2", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`convert_f16x2_to_f8x2`

This Op converts the given f16 inputs in an f16x2 vector to the specified 
f8 type.
The result `dst` is represented as an i16 type or as a vector
of two i8 types.
If `dst` is returned as an i16 type, the converted values from `a`
are packed such that the value converted from the first element of `a`
is stored in the upper 8 bits of `dst` and the value converted from the
second element of `a` is stored in the lower 8 bits of `dst`.
If `dst` is returned as a vector type, each converted value is stored as an 
i8 element in the vector.
The `relu` attribute, when set, lowers to the \'.relu\' variant of
the cvt instruction.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
"""
function convert_f16x2_to_f8x2(a::Value; dst::IR.Type, type, relu=nothing, location=Location())
    op_ty_results = IR.Type[dst, ]
    operands = Value[a, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("type", type), ]
    !isnothing(relu) && push!(attributes, NamedAttribute("relu", relu))
    
    create_operation(
        "nvvm.convert.f16x2.to.f8x2", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`convert_f32x2_to_f6x2`

This Op converts each of the given float inputs to the specified fp6 type.
The result `dst` is represented either as an i16 type or as a vector
of two i8 types.
If `dst` is returned as an i16 type, the converted values are packed such 
that the value converted from `a` is stored in the upper 8 bits of `dst` 
with 2 MSB bits padded with zeros and the value converted from `b` is 
stored in the lower 8 bits of `dst` with 2 MSB bits padded with zeros.
If `dst` is returned as a vector type, each converted value is stored as an 
i8 element in the vector.
The `relu` attribute, when set, lowers to the \'.relu\' variant of
the cvt instruction.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
"""
function convert_f32x2_to_f6x2(a::Value, b::Value; dst::IR.Type, type, relu=nothing, location=Location())
    op_ty_results = IR.Type[dst, ]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("type", type), ]
    !isnothing(relu) && push!(attributes, NamedAttribute("relu", relu))
    
    create_operation(
        "nvvm.convert.f32x2.to.f6x2", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`convert_f32x2_to_f8x2`

This Op converts each of the given float inputs to the specified fp8 type.
The result `dst` is represented as an i16 type or as a vector
of two i8 types.
If `dst` is returned as an i16 type, the converted values are packed such 
that the value converted from `a` is stored in the upper 8 bits of `dst` 
and the value converted from `b` is stored in the lower 8 bits of `dst`.
If `dst` is returned as a vector type, each converted value is stored as an 
i8 element in the vector.
The `rnd` and `sat` attributes specify the rounding and saturation modes respectively.
The `relu` attribute, when set, lowers to the \'.relu\' variant of
the cvt instruction.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
"""
function convert_f32x2_to_f8x2(a::Value, b::Value; dst::IR.Type, type, rnd=nothing, sat=nothing, relu=nothing, location=Location())
    op_ty_results = IR.Type[dst, ]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("type", type), ]
    !isnothing(rnd) && push!(attributes, NamedAttribute("rnd", rnd))
    !isnothing(sat) && push!(attributes, NamedAttribute("sat", sat))
    !isnothing(relu) && push!(attributes, NamedAttribute("relu", relu))
    
    create_operation(
        "nvvm.convert.f32x2.to.f8x2", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`convert_float_to_tf32`

This Op converts the given f32 input to tf32.
The result `res` is represented as an i32 type.
The `relu` attribute, when set, lowers to the \'.relu\' variant of
the cvt instruction. The `rnd` and `sat` attributes specify the
the rounding and saturation modes respectively.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
"""
function convert_float_to_tf32(src::Value; res::IR.Type, rnd=nothing, sat=nothing, relu=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(rnd) && push!(attributes, NamedAttribute("rnd", rnd))
    !isnothing(sat) && push!(attributes, NamedAttribute("sat", sat))
    !isnothing(relu) && push!(attributes, NamedAttribute("relu", relu))
    
    create_operation(
        "nvvm.convert.float.to.tf32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_bulk_commit_group`

This Op commits all prior initiated but uncommitted cp.async.bulk
instructions into a cp.async.bulk-group.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group)
"""
function cp_async_bulk_commit_group(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.cp.async.bulk.commit.group", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_bulk_shared_cluster_global`

Initiates an asynchronous copy operation from global memory to cluster\'s
shared memory.

The `multicastMask` operand is optional. When it is present, the Op copies
data from global memory to shared memory of multiple CTAs in the cluster.
Operand `multicastMask` specifies the destination CTAs in the cluster such
that each bit position in the 16-bit `multicastMask` operand corresponds to
the `nvvm.read.ptx.sreg.ctaid` of the destination CTA.

The `l2CacheHint` operand is optional, and it is used to specify cache
eviction policy that may be used during the memory access.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)
"""
function cp_async_bulk_shared_cluster_global(dstMem::Value, srcMem::Value, mbar::Value, size::Value, multicastMask=nothing::Union{Nothing, Value}; l2CacheHint=nothing::Union{Nothing, Value}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dstMem, srcMem, mbar, size, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(multicastMask) && push!(operands, multicastMask)
    !isnothing(l2CacheHint) && push!(operands, l2CacheHint)
    push!(attributes, operandsegmentsizes([1, 1, 1, 1, Int(!isnothing(multicastMask)), Int(!isnothing(l2CacheHint)), ]))
    
    create_operation(
        "nvvm.cp.async.bulk.shared.cluster.global", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_bulk_prefetch`

Initiates an asynchronous prefetch of data from the location
specified by `srcMem` to the L2 cache.

The `l2CacheHint` operand is optional, and it is used to specify cache
eviction policy that may be used during the memory access.

# Example
```mlir
  nvvm.cp.async.bulk.prefetch %src, %size : !llvm.ptr<1>

  // with l2_cache_hint
  nvvm.cp.async.bulk.prefetch %src, %size l2_cache_hint = %ch : !llvm.ptr<1>
```

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch)
"""
function cp_async_bulk_prefetch(srcMem::Value, size::Value, l2CacheHint=nothing::Union{Nothing, Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[srcMem, size, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(l2CacheHint) && push!(operands, l2CacheHint)
    
    create_operation(
        "nvvm.cp.async.bulk.prefetch", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_bulk_global_shared_cta`

Initiates an asynchronous copy operation from Shared CTA memory to
global memory. The 32-bit operand `size` specifies the amount of
memory to be copied, in terms of number of bytes. `size` must be a
multiple of 16. The `l2CacheHint` operand is optional, and it is used
to specify cache eviction policy that may be used during the memory
access. The `byteMask` operand is optional. The i-th bit in the 16-bit
wide `byteMask` specifies whether the i-th byte of each 16-byte wide
chunk of source data is copied to the destination. If the bit is set,
the byte is copied.

# Example
```mlir
  nvvm.cp.async.bulk.global.shared.cta %dst, %src, %size
      : !llvm.ptr<1>, !llvm.ptr<3>

  // with l2_cache_hint
  nvvm.cp.async.bulk.global.shared.cta %dst, %src, %size l2_cache_hint = %ch
      : !llvm.ptr<1>, !llvm.ptr<3>

  // with byte_mask
  nvvm.cp.async.bulk.global.shared.cta %dst, %src, %size byte_mask = %mask
      : !llvm.ptr<1>, !llvm.ptr<3>

  // with both l2_cache_hint and byte_mask
  nvvm.cp.async.bulk.global.shared.cta %dst, %src, %size l2_cache_hint = %ch byte_mask = %mask
      : !llvm.ptr<1>, !llvm.ptr<3>
```

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)
"""
function cp_async_bulk_global_shared_cta(dstMem::Value, srcMem::Value, size::Value, l2CacheHint=nothing::Union{Nothing, Value}; byteMask=nothing::Union{Nothing, Value}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dstMem, srcMem, size, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(l2CacheHint) && push!(operands, l2CacheHint)
    !isnothing(byteMask) && push!(operands, byteMask)
    push!(attributes, operandsegmentsizes([1, 1, 1, Int(!isnothing(l2CacheHint)), Int(!isnothing(byteMask)), ]))
    
    create_operation(
        "nvvm.cp.async.bulk.global.shared.cta", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_bulk_shared_cluster_shared_cta`

Initiates an asynchronous copy operation from Shared CTA memory to Shared
cluster memory.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)
"""
function cp_async_bulk_shared_cluster_shared_cta(dstMem::Value, srcMem::Value, mbar::Value, size::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dstMem, srcMem, mbar, size, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.cp.async.bulk.shared.cluster.shared.cta", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_bulk_tensor_shared_cluster_global`

Initiates an asynchronous copy operation on the tensor data from global 
memory to shared memory. 

The Op operates has two load modes:
1) Tiled Mode: It\'s the default mode. The source multi-dimensional tensor 
layout is preserved at the destination. 

2) Im2col Mode: This mode is used when `im2colOffsets` operands are present.
the elements in the Bounding Box of the source tensor are rearranged into
columns at the destination. In this mode, the tensor has to be at least 
3-dimensional. 

The `multicastMask` operand is optional. When it is present, the Op copies
data from global memory to shared memory of multiple CTAs in the cluster.
Operand `multicastMask` specifies the destination CTAs in the cluster such 
that each bit position in the 16-bit `multicastMask` operand corresponds to
the `nvvm.read.ptx.sreg.ctaid` of the destination CTA.     

The `l2CacheHint` operand is optional, and it is used to specify cache 
eviction policy that may be used during the memory access.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
"""
function cp_async_bulk_tensor_shared_cluster_global(dstMem::Value, tmaDescriptor::Value, coordinates::Vector{Value}, mbar::Value, im2colOffsets::Vector{Value}, multicastMask=nothing::Union{Nothing, Value}; l2CacheHint=nothing::Union{Nothing, Value}, predicate=nothing::Union{Nothing, Value}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dstMem, tmaDescriptor, coordinates..., mbar, im2colOffsets..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(multicastMask) && push!(operands, multicastMask)
    !isnothing(l2CacheHint) && push!(operands, l2CacheHint)
    !isnothing(predicate) && push!(operands, predicate)
    push!(attributes, operandsegmentsizes([1, 1, length(coordinates), 1, length(im2colOffsets), Int(!isnothing(multicastMask)), Int(!isnothing(l2CacheHint)), Int(!isnothing(predicate)), ]))
    
    create_operation(
        "nvvm.cp.async.bulk.tensor.shared.cluster.global", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_bulk_tensor_prefetch`

Initiates an asynchronous prefetch operation on the tensor data from global
memory to L2 cache.

The Op has two modes:
1) Tiled Mode: It\'s the default mode. The source multi-dimensional tensor
layout is preserved at the destination.

2) Im2col Mode: This mode is used when `im2colOffsets` operands are present.
the elements in the Bounding Box of the source tensor are rearranged into
columns at the destination. In this mode, the tensor has to be at least
3-dimensional.

The `l2CacheHint` operand is optional, and it is used to specify cache
eviction policy that may be used during the memory access.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor)
"""
function cp_async_bulk_tensor_prefetch(tmaDescriptor::Value, coordinates::Vector{Value}, im2colOffsets::Vector{Value}, l2CacheHint=nothing::Union{Nothing, Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[tmaDescriptor, coordinates..., im2colOffsets..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(l2CacheHint) && push!(operands, l2CacheHint)
    push!(attributes, operandsegmentsizes([1, length(coordinates), length(im2colOffsets), Int(!isnothing(l2CacheHint)), ]))
    
    create_operation(
        "nvvm.cp.async.bulk.tensor.prefetch", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_bulk_tensor_reduce`

Initiates an asynchronous reduction operation of tensor data in
global memory with tensor data in shared memory.

The `mode` attribute indicates whether the copy mode is tile or im2col.
The `redOp` attribute specifies the reduction operations applied.
The supported reduction operations are:
{add, min, max, inc, dec, and, or, xor}

The `l2CacheHint` operand is optional, and it is used to specify cache
eviction policy that may be used during the memory access.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor)
"""
function cp_async_bulk_tensor_reduce(tmaDescriptor::Value, srcMem::Value, coordinates::Vector{Value}, l2CacheHint=nothing::Union{Nothing, Value}; redKind, mode=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[tmaDescriptor, srcMem, coordinates..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("redKind", redKind), ]
    !isnothing(l2CacheHint) && push!(operands, l2CacheHint)
    push!(attributes, operandsegmentsizes([1, 1, length(coordinates), Int(!isnothing(l2CacheHint)), ]))
    !isnothing(mode) && push!(attributes, NamedAttribute("mode", mode))
    
    create_operation(
        "nvvm.cp.async.bulk.tensor.reduce", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function cp_async_bulk_tensor_global_shared_cta(tmaDescriptor::Value, srcMem::Value, coordinates::Vector{Value}, predicate=nothing::Union{Nothing, Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[tmaDescriptor, srcMem, coordinates..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(predicate) && push!(operands, predicate)
    push!(attributes, operandsegmentsizes([1, 1, length(coordinates), Int(!isnothing(predicate)), ]))
    
    create_operation(
        "nvvm.cp.async.bulk.tensor.global.shared.cta", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_bulk_wait_group`

Op waits for completion of the most recent bulk async-groups.

The `\$group` operand tells waiting has to be done until for \$group or fewer
of the most recent bulk async-groups. If `\$group` is 0, the op wait until 
all the most recent bulk async-groups have completed.

The `\$read` indicates that the waiting has to be done until all the bulk 
async operations in the specified bulk async-group have completed reading 
from their source locations.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)
"""
function cp_async_bulk_wait_group(; group, read=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("group", group), ]
    !isnothing(read) && push!(attributes, NamedAttribute("read", read))
    
    create_operation(
        "nvvm.cp.async.bulk.wait_group", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function cp_async_commit_group(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.cp.async.commit.group", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_mbarrier_arrive`

The `cp.async.mbarrier.arrive` Op makes the mbarrier object track
all prior cp.async operations initiated by the executing thread.
The `addr` operand specifies the address of the mbarrier object
in generic address space. The `noinc` attr impacts how the
mbarrier\'s state is updated.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive)
"""
function cp_async_mbarrier_arrive(addr::Value; noinc=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(noinc) && push!(attributes, NamedAttribute("noinc", noinc))
    
    create_operation(
        "nvvm.cp.async.mbarrier.arrive", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cp_async_mbarrier_arrive_shared`

The `cp.async.mbarrier.arrive.shared` Op makes the mbarrier object
track all prior cp.async operations initiated by the executing thread.
The `addr` operand specifies the address of the mbarrier object in
shared memory. The `noinc` attr impacts how the mbarrier\'s state
is updated. 

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive)
"""
function cp_async_mbarrier_arrive_shared(addr::Value; noinc=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(noinc) && push!(attributes, NamedAttribute("noinc", noinc))
    
    create_operation(
        "nvvm.cp.async.mbarrier.arrive.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function cp_async_shared_global(dst::Value, src::Value, cpSize=nothing::Union{Nothing, Value}; size, modifier, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dst, src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("size", size), NamedAttribute("modifier", modifier), ]
    !isnothing(cpSize) && push!(operands, cpSize)
    
    create_operation(
        "nvvm.cp.async.shared.global", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function cp_async_wait_group(; n, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("n", n), ]
    
    create_operation(
        "nvvm.cp.async.wait.group", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`dot_accumulate_2way`

Performs a two-way 16-bit to 8-bit dot-product which is accumulated in a 
32-bit result.
Operand `a` is a vector of two 16-bit elements and operand `b` a vector 
of four 8-bit elements between which the dot product is computed.

The `a_type` and `b_type` attributes specify the type of the elements in `a`
and `b` respectively.
If `a_type` or `b_type` is `s`, then the elements in the corresponding 
vector are sign-extended to 32-bit before the dot product is computed.
If `a_type` or `b_type` is `u`, then the elements in the corresponding 
vector are zero-extended to 32-bit instead.

The `b_hi` boolean attribute specifies which two bytes of `b` are used for 
the dot product. If `b_hi` is true, then the dot product is computed 
between  `a` and elements at indices 2 and 3 of `b`. If `b_hi` is false, 
then the dot product is computed between `a` and elements at indices 0 and 
1 of `b`.

Operand `c` is a 32-bit integer to which the result is accumulated. It is
treated as holding a signed integer if any of `a_type` or `b_type` is 
signed.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-dp2a)
"""
function dot_accumulate_2way(a::Value, b::Value, c::Value; res::IR.Type, a_type, b_type, b_hi, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[a, b, c, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("a_type", a_type), NamedAttribute("b_type", b_type), NamedAttribute("b_hi", b_hi), ]
    
    create_operation(
        "nvvm.dot.accumulate.2way", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`dot_accumulate_4way`

Performs a four-way byte dot-product which is accumulated in a 32-bit
result.
Operand `a` and `b` are vectors of 4 bytes between which the dot product is 
computed.

The `a_type` and `b_type` attributes specify the type of the elements in `a`
and `b` respectively.
If `a_type` or `b_type` is `signed`, then the elements in the corresponding 
vector are sign-extended to 32-bit before the dot product is computed.
If `a_type` or `b_type` is `unsigned`, then the elements in the 
corresponding vector are zero-extended to 32-bit instead.

Operand `c` is a 32-bit integer to which the result is accumulated. It is
treated as holding a signed integer if any of `a_type` or `b_type` is `s8`.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-dp4a)
"""
function dot_accumulate_4way(a::Value, b::Value, c::Value; res::IR.Type, a_type, b_type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[a, b, c, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("a_type", a_type), NamedAttribute("b_type", b_type), ]
    
    create_operation(
        "nvvm.dot.accumulate.4way", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`elect_sync`

The `elect.sync` instruction elects one predicated active leader
thread from among a set of threads specified in the `membermask`.
When the `membermask` is not provided explicitly, a default value
of `0xFFFFFFFF` is used. The predicate result is set to `True` for
the leader thread, and `False` for all other threads.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync)
"""
function elect_sync(membermask=nothing::Union{Nothing, Value}; pred::IR.Type, location=Location())
    op_ty_results = IR.Type[pred, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(membermask) && push!(operands, membermask)
    
    create_operation(
        "nvvm.elect.sync", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg0(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg0", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg1(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg1", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg2(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg2", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg3(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg3", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg4(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg5(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg5", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg6(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg6", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg7(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg7", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg8(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg9(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg9", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg10(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg10", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg11(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg11", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg12(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg12", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg13(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg13", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg14(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg14", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg15(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg15", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg16(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg17(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg17", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg18(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg18", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg19(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg19", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg20(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg20", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg21(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg21", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg22(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg22", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg23(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg23", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg24(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg24", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg25(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg25", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg26(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg26", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg27(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg27", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg28(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg28", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg29(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg29", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg30(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg30", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_envreg31(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.envreg31", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`exit`

Ends execution of a thread.
[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-exit)
"""
function exit(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.exit", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`fence_mbarrier_init`

Fence operation that applies on the prior nvvm.mbarrier.init

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
"""
function fence_mbarrier_init(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.fence.mbarrier.init", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`fence_proxy_acquire`

`fence.proxy.acquire` is a uni-directional fence used to establish ordering
between a prior memory access performed via the generic proxy and a
subsequent memory access performed via the tensormap proxy

The address operand `addr` and the operand `size` together specify the
memory range `[addr, addr+size)` on which the ordering guarantees on the
memory accesses across the proxies is to be provided. The only supported
value for the `size` operand is 128 and must be an immediate. Generic Addressing
is used unconditionally, and the address specified by the operand `addr` must
fall within the `.global` state space. Otherwise, the behavior is undefined

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
"""
function fence_proxy_acquire(addr::Value, size::Value; scope, fromProxy=nothing, toProxy=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, size, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("scope", scope), ]
    !isnothing(fromProxy) && push!(attributes, NamedAttribute("fromProxy", fromProxy))
    !isnothing(toProxy) && push!(attributes, NamedAttribute("toProxy", toProxy))
    
    create_operation(
        "nvvm.fence.proxy.acquire", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`fence_proxy`

Fence operation with proxy to establish an ordering between memory accesses
that may happen through different proxies.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
"""
function fence_proxy(; kind, space=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("kind", kind), ]
    !isnothing(space) && push!(attributes, NamedAttribute("space", space))
    
    create_operation(
        "nvvm.fence.proxy", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`fence_proxy_release`

`fence.proxy.release` is a uni-directional fence used to establish ordering
between a prior memory access performed via the generic proxy and a
subsequent memory access performed via the tensormap proxy. `fence.proxy.release`
operation can form a release sequence that synchronizes with an acquire
sequence that contains the fence.proxy.acquire proxy fence operation

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
"""
function fence_proxy_release(; scope, fromProxy=nothing, toProxy=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("scope", scope), ]
    !isnothing(fromProxy) && push!(attributes, NamedAttribute("fromProxy", fromProxy))
    !isnothing(toProxy) && push!(attributes, NamedAttribute("toProxy", toProxy))
    
    create_operation(
        "nvvm.fence.proxy.release", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function fence_sc_cluster(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.fence.sc.cluster", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_globaltimer(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.globaltimer", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_nctaid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.nctaid.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_nctaid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.nctaid.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_nctaid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.nctaid.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_gridid(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.gridid", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`griddepcontrol_launch_dependents`

Signals that specific dependents the runtime system designated to react to 
this instruction can be scheduled as soon as all other CTAs in the grid 
issue the same instruction or have completed.


[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol)
"""
function griddepcontrol_launch_dependents(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.griddepcontrol.launch.dependents", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`griddepcontrol_wait`

Causes the executing thread to wait until all prerequisite grids in flight 
have completed and all the memory operations from the prerequisite grids 
are performed and made visible to the current grid.


[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol)
"""
function griddepcontrol_wait(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.griddepcontrol.wait", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`inline_ptx`
This op allows using PTX directly within the NVVM 
    dialect, while greatly simplifying llvm.inline_asm generation. It 
    automatically handles register size selection and sets the correct 
    read/write access for each operand. The operation leverages the 
    `BasicPtxBuilderInterface` to abstract away low-level details of 
    PTX assembly formatting.

    The `predicate` attribute is used to specify a predicate for the 
    PTX instruction.

    Example 1: Read-only Parameters
    ```mlir
    nvvm.inline_ptx \"mbarrier.init.b64 [\$0], \$1;\" (%barrier_gen, %count) : !llvm.ptr, i32

    // Lowers to:
    llvm.inline_asm has_side_effects asm_dialect = att 
      \"mbarrier.init.b64 [\$0], \$1;\", \"l,r\" %arg0, %arg2 : (!llvm.ptr, i32) -> ()
    ```

    Example 2: Read-only and Write-only Parameters
    ```mlir
    %0 = nvvm.inline_ptx \"ex2.approx.ftz.f32 \$0, \$1;\" (%input) : f32 -> f32

    // Lowers to:
    %0 = llvm.inline_asm has_side_effects asm_dialect = att 
      \"ex2.approx.ftz.f32 \$0, \$1;\", \"=f,f\" %arg0 : (f32) -> f32
    ```

    Example 3: Predicate Usage
    ```mlir
    nvvm.inline_ptx \"mbarrier.init.b64 [\$0], \$1;\" (%barrier_gen, %count), 
      predicate = %pred : !llvm.ptr, i32, i1

    // Lowers to:
    llvm.inline_asm has_side_effects asm_dialect = att 
      \"@\$2 mbarrier.init.b64 [\$0], \$1;\", \"l,r,b\" %arg0, %arg2, %arg3 
      : (!llvm.ptr, i32, i1) -> ()
    ```
"""
function inline_ptx(readOnlyArgs::Vector{Value}, predicate=nothing::Union{Nothing, Value}; writeOnlyArgs::Vector{IR.Type}, ptxCode, location=Location())
    op_ty_results = IR.Type[writeOnlyArgs..., ]
    operands = Value[readOnlyArgs..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("ptxCode", ptxCode), ]
    !isnothing(predicate) && push!(operands, predicate)
    push!(attributes, operandsegmentsizes([length(readOnlyArgs), Int(!isnothing(predicate)), ]))
    
    create_operation(
        "nvvm.inline_ptx", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_laneid(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.laneid", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_lanemask_eq(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.lanemask.eq", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_lanemask_ge(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.lanemask.ge", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_lanemask_gt(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.lanemask.gt", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_lanemask_le(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.lanemask.le", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_lanemask_lt(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.read.ptx.sreg.lanemask.lt", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ldmatrix(ptr::Value; res::IR.Type, num, layout, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("num", num), NamedAttribute("layout", layout), ]
    
    create_operation(
        "nvvm.ldmatrix", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbarrier_arrive_expect_tx(addr::Value, txcount::Value, predicate=nothing::Union{Nothing, Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, txcount, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(predicate) && push!(operands, predicate)
    
    create_operation(
        "nvvm.mbarrier.arrive.expect_tx", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbarrier_arrive_expect_tx_shared(addr::Value, txcount::Value, predicate=nothing::Union{Nothing, Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, txcount, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(predicate) && push!(operands, predicate)
    
    create_operation(
        "nvvm.mbarrier.arrive.expect_tx.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbarrier_arrive_nocomplete(addr::Value, count::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[addr, count, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.mbarrier.arrive.nocomplete", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbarrier_arrive_nocomplete_shared(addr::Value, count::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[addr, count, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.mbarrier.arrive.nocomplete.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbarrier_arrive(addr::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.mbarrier.arrive", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbarrier_arrive_shared(addr::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.mbarrier.arrive.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbarrier_init(addr::Value, count::Value, predicate=nothing::Union{Nothing, Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, count, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(predicate) && push!(operands, predicate)
    
    create_operation(
        "nvvm.mbarrier.init", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbarrier_init_shared(addr::Value, count::Value, predicate=nothing::Union{Nothing, Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, count, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(predicate) && push!(operands, predicate)
    
    create_operation(
        "nvvm.mbarrier.init.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbarrier_inval(addr::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.mbarrier.inval", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbarrier_inval_shared(addr::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.mbarrier.inval.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbarrier_test_wait(addr::Value, state::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[addr, state, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.mbarrier.test.wait", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbarrier_test_wait_shared(addr::Value, state::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[addr, state, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.mbarrier.test.wait.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbarrier_try_wait_parity(addr::Value, phase::Value, ticks::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, phase, ticks, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.mbarrier.try_wait.parity", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbarrier_try_wait_parity_shared(addr::Value, phase::Value, ticks::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, phase, ticks, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.mbarrier.try_wait.parity.shared", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mapa(a::Value, b::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.mapa", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`match_sync`

The `match.sync` op performs broadcast and compare of operand `val` across 
all non-exited threads in `thread_mask` and returns a mask depending on the 
kind and an optional predicate.

The matching operation kinds are:
- `any`: Returns a mask corresponding to the non-exited threads in the 
`thread_mask` that have the same value of operand `val`.
- `all`: Returns a mask and a predicate. If all non-exited threads in the 
`thread_mask` have the same value of operand `val`, the predicate is set to 
true and the mask corresponds to the non-exited threads in the 
`thread_mask`. Otherwise, the predicate is set to false and the mask is 0.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-match-sync)
"""
function match_sync(thread_mask::Value, val::Value; res::IR.Type, kind, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[thread_mask, val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("kind", kind), ]
    
    create_operation(
        "nvvm.match.sync", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mma_sync`

The `nvvm.mma.sync` operation collectively performs the operation
`D = matmul(A, B) + C` using all threads in a warp.

All the threads in the warp must execute the same `mma.sync` operation.

For each possible multiplicand PTX data type, there are one or more possible
instruction shapes given as \"mMnNkK\". The below table describes the posssibilities
as well as the types required for the operands. Note that the data type for
C (the accumulator) and D (the result) can vary independently when there are
multiple possibilities in the \"C/D Type\" column.

When an optional attribute cannot be immediately inferred from the types of
the operands and the result during parsing or validation, an error will be
raised.

`b1Op` is only relevant when the binary (b1) type is given to
`multiplicandDataType`. It specifies how the multiply-and-acumulate is
performed and is either `xor_popc` or `and_poc`. The default is `xor_popc`.

`intOverflowBehavior` is only relevant when the `multiplicandType` attribute
is one of `u8, s8, u4, s4`, this attribute describes how overflow is handled
in the accumulator. When the attribute is `satfinite`, the accumulator values
are clamped in the int32 range on overflow. This is the default behavior.
Alternatively, accumulator behavior `wrapped` can also be specified, in
which case overflow wraps from one end of the range to the other.

`layoutA` and `layoutB` are required and should generally be set to
`#nvvm.mma_layout<row>` and `#nvvm.mma_layout<col>` respectively, but other
combinations are possible for certain layouts according to the table below.

```
| A/B Type | Shape     | ALayout | BLayout | A Type   | B Type   | C/D Type          |
|----------|-----------|---------|---------|----------|----------|-------------------|
| f64      | .m8n8k4   | row     | col     | 1x f64   | 1x f64   | 2x f64            |
| f16      | .m8n8k4   | row/col | row/col | 2x f16x2 | 2x f16x2 | 4x f16x2 or 8xf32 |
|          | .m16n8k8  | row     | col     | 2x f16x2 | 1x f16x2 | 2x f16x2 or 4 f32 |
|          | .m16n8k16 | row     | col     | 4x f16x2 | 2x f16x2 | 2x f16x2 or 4 f32 |
| bf16     | .m16n8k8  | row     | col     | 2x i32   | 1x i32   | 4x f32            |
|          | .m16n8k16 | row     | col     | 4x i32   | 2x i32   | 4x f32            |
| tf32     | .m16n8k4  | row     | col     | 2x i32   | 1x i32   | 4x f32            |
|          | .m16n8k8  | row     | col     | 4x i32   | 2x i32   | 2x f16x2 or 4 f32 |
| u8/s8    | .m8n8k16  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
|          | .m16n8k16 | row     | col     | 2x i32   | 1x i32   | 4x i32            |
|          | .m16n8k32 | row     | col     | 4x i32   | 2x i32   | 4x i32            |
| u4/s4    | .m8n8k32  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
|          | m16n8k32  | row     | col     | 2x i32   | 1x i32   | 4x i32            |
|          | m16n8k64  | row     | col     | 4x i32   | 2x i32   | 4x i32            |
| b1       | m8n8k128  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
|          | m16n8k128 | row     | col     | 2x i32   | 1x i32   | 4x i32            |
```


# Example
```mlir

%128 = nvvm.mma.sync A[%120, %121, %122, %123]
                     B[%124, %125]
                     C[%126, %127]
                     {layoutA = #nvvm.mma_layout<row>,
                      layoutB = #nvvm.mma_layout<col>,
                      shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}}
    : (vector<2xf16>, vector<2xf16>, vector<2xf16>)
       -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
```
"""
function mma_sync(operandA::Vector{Value}, operandB::Vector{Value}, operandC::Vector{Value}; res::IR.Type, shape, b1Op=nothing, intOverflowBehavior=nothing, layoutA, layoutB, multiplicandAPtxType=nothing, multiplicandBPtxType=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operandA..., operandB..., operandC..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("shape", shape), NamedAttribute("layoutA", layoutA), NamedAttribute("layoutB", layoutB), ]
    push!(attributes, operandsegmentsizes([length(operandA), length(operandB), length(operandC), ]))
    !isnothing(b1Op) && push!(attributes, NamedAttribute("b1Op", b1Op))
    !isnothing(intOverflowBehavior) && push!(attributes, NamedAttribute("intOverflowBehavior", intOverflowBehavior))
    !isnothing(multiplicandAPtxType) && push!(attributes, NamedAttribute("multiplicandAPtxType", multiplicandAPtxType))
    !isnothing(multiplicandBPtxType) && push!(attributes, NamedAttribute("multiplicandBPtxType", multiplicandBPtxType))
    
    create_operation(
        "nvvm.mma.sync", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`prefetch`

Operand `addr` can be a global, local or generic address pointer. No 
operation is performed if `addr` maps to a `shared` memory location.

The `cacheLevel` attribute specifies the cache level to which the cache line
containing the specified address is brought.

`uniform` can be specified after the `cacheLevel` to indicate that the 
prefetch is performed to the specified uniform cache level. If `uniform` is 
specified, `addr` must be a generic address pointer and no operation is 
performed if `addr` maps to a `const`, `local`, or `shared` memory location.

The `evictPriority` attribute is optional and specifies the cache eviction
priority when `cacheLevel` is L2.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prefetch-prefetchu)
"""
function prefetch(addr::Value; cacheLevel, uniform=nothing, evictPriority=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("cacheLevel", cacheLevel), ]
    !isnothing(uniform) && push!(attributes, NamedAttribute("uniform", uniform))
    !isnothing(evictPriority) && push!(attributes, NamedAttribute("evictPriority", evictPriority))
    
    create_operation(
        "nvvm.prefetch", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function prefetch_tensormap(tmaDescriptor::Value, predicate=nothing::Union{Nothing, Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[tmaDescriptor, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(predicate) && push!(operands, predicate)
    
    create_operation(
        "nvvm.prefetch.tensormap", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function rcp_approx_ftz_f(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.rcp.approx.ftz.f", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`redux_sync`

`redux.sync` performs a reduction operation `kind` of the 32 bit source 
register across all non-exited threads in the membermask.

The `abs` and `nan` attributes can be used in the case of f32 input type, 
where the `abs` attribute causes the absolute value of the input to be used 
in the reduction operation, and the `nan` attribute causes the reduction 
operation to return NaN if any of the inputs to participating threads are 
NaN.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-redux-sync)
"""
function redux_sync(val::Value, mask_and_clamp::Value; res::IR.Type, kind, abs=nothing, nan=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[val, mask_and_clamp, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("kind", kind), ]
    !isnothing(abs) && push!(attributes, NamedAttribute("abs", abs))
    !isnothing(nan) && push!(attributes, NamedAttribute("nan", nan))
    
    create_operation(
        "nvvm.redux.sync", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function setmaxregister(; regCount, action, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("regCount", regCount), NamedAttribute("action", action), ]
    
    create_operation(
        "nvvm.setmaxregister", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`shfl_sync`

The `shfl.sync` Op implements data shuffle within threads of a warp.
The `thread_mask` denotes the threads participating in the Op where
the bit position corresponds to a particular thread’s laneid.
The `offset` specifies a source lane or source lane offset
(depending on `kind`). The `val` is the input value to be copied from
the source. The `mask_and_clamp` contains two packed values specifying
a mask for logically splitting warps into sub-segments and an upper bound
for clamping the source lane index.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync)
"""
function shfl_sync(thread_mask::Value, val::Value, offset::Value, mask_and_clamp::Value; res::IR.Type, kind, return_value_and_is_valid=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[thread_mask, val, offset, mask_and_clamp, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("kind", kind), ]
    !isnothing(return_value_and_is_valid) && push!(attributes, NamedAttribute("return_value_and_is_valid", return_value_and_is_valid))
    
    create_operation(
        "nvvm.shfl.sync", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_nsmid(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.nsmid", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_smid(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.smid", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`stmatrix`

Collectively store one or more matrices across all threads in a warp to the
location indicated by the address operand \$ptr in shared memory.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix)
"""
function stmatrix(ptr::Value, sources::Vector{Value}; layout, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, sources..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("layout", layout), ]
    
    create_operation(
        "nvvm.stmatrix", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function bar_warp_sync(mask::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[mask, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.bar.warp.sync", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tcgen05_alloc`

The `tcgen05.alloc` Op allocates tensor core memory for
the amount specified by `nCols` and writes the destination
address to the `addr` argument. The `nCols` operand specifies the
number of columns to be allocated and it must be a power-of-two.
[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)
"""
function tcgen05_alloc(addr::Value, nCols::Value; group=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, nCols, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(group) && push!(attributes, NamedAttribute("group", group))
    
    create_operation(
        "nvvm.tcgen05.alloc", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tcgen05_commit`

The `tcgen05.commit` makes the mbarrier object, specified by
the operand `addr`, track the completion of all the prior
async-tcgen05 operations initiated by the executing thread.
The multicast variants allow signaling on the mbarrier objects
of multiple CTAs within the cluster. Operand `multicastMask`,
when present, specifies the destination CTAs in the cluster such
that each bit position in the 16-bit `multicastMask` operand
corresponds to the `nvvm.read.ptx.sreg.ctaid` of the destination CTA.
[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
"""
function tcgen05_commit(addr::Value, multicastMask=nothing::Union{Nothing, Value}; group=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(multicastMask) && push!(operands, multicastMask)
    !isnothing(group) && push!(attributes, NamedAttribute("group", group))
    
    create_operation(
        "nvvm.tcgen05.commit", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tcgen05_cp`

Instruction tcgen05.cp initiates an asynchronous copy operation from
shared memory to the location specified by the address operand `taddr`
in the Tensor Memory. The 64-bit register operand `smem_desc` specifies
the matrix descriptor representing the source matrix in the shared memory
that needs to be copied.

# Example
```mlir
  nvvm.tcgen05.cp %taddr, %smem_desc {
    group = #nvvm.tcgen05_group<cta_2>,
    shape = #nvvm.tcgen05_cp_shape<shape_64x128b>,
    multicast = #nvvm.tcgen05_cp_multicast<warpx2_01_23>,
    srcFormat = #nvvm.tcgen05_cp_src_fmt<b6x16_p32>
  }
```
[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-cp)
"""
function tcgen05_cp(taddr::Value, smem_desc::Value; shape, group=nothing, multicast=nothing, srcFormat=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[taddr, smem_desc, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("shape", shape), ]
    !isnothing(group) && push!(attributes, NamedAttribute("group", group))
    !isnothing(multicast) && push!(attributes, NamedAttribute("multicast", multicast))
    !isnothing(srcFormat) && push!(attributes, NamedAttribute("srcFormat", srcFormat))
    
    create_operation(
        "nvvm.tcgen05.cp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tcgen05_dealloc`

The `tcgen05.dealloc` Op de-allocates the tensor core memory
specified by `tmemAddr`, which must be from a previous tensor
memory allocation. The `nCols` operand specifies the number
of columns to be de-allocated, and it must be a power-of-two.
[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)
"""
function tcgen05_dealloc(taddr::Value, nCols::Value; group=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[taddr, nCols, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(group) && push!(attributes, NamedAttribute("group", group))
    
    create_operation(
        "nvvm.tcgen05.dealloc", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tcgen05_fence`

The `tcgen05.fence<before>` orders all prior async tcgen05 operations
with respect to the subsequent tcgen05 and execution ordering operations.
The `tcgen05.fence<after>` orders all subsequent async tcgen05 operations
with respect to the prior tcgen05 and execution ordering operations.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-fence)
"""
function tcgen05_fence(; kind, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("kind", kind), ]
    
    create_operation(
        "nvvm.tcgen05.fence", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tcgen05_ld`

Instruction `tcgen05.ld` asynchronously loads data from the Tensor Memory at
the location specified by the 32-bit address operand `tmemAddr` into the
destination register `res`, collectively across all threads of the warps.

The `shape` and the `num` attribute together determines the total
dimension of the data which is loaded from the Tensor Memory. The `shape`
attribute indicates the base dimension of data to be accessed as described
in the Data Movement Shape. The `num` attribute indicates the repeat
factor on the base dimension resulting in the total dimension of the data
that is accessed.

The shape `16x32bx2` performs two accesses into Tensor Memory of the shape
`16x32b`. The base address of the first access is specified by `tmemAddr`
and the base address of the second access is specified by
`tmemAddr + offset`, where `offset` is an immediate argument.

The unit attribute `pack` can be used to pack two 16-bit
elements from adjacent columns into a single 32-bit element during the load.

The following table describes the size of the vector for various combinations
of `num` and `shape` attributes:
```
|=====================================================================|
| num/shape      |     16x32bx2/16x64b/32x32b |  16x128b   | 16x256b  |
|=====================================================================|
| x1             |          1                 |    2       |    4     |
| x2             |          2                 |    4       |    8     |
| x4             |          4                 |    8       |    16    |
| x8             |          8                 |    16      |    32    |
| x16            |          16                |    32      |    64    |
| x32            |          32                |    64      |    128   |
| x64            |          64                |    128     |    NA    |
| x128           |          128               |    NA      |    NA    |
|=====================================================================|
```

# Example
```mlir
  nvvm.tcgen05.ld %tmemAddr, %offset pack {
    shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>,
  } : <2xi32>
```

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st)
"""
function tcgen05_ld(tmemAddr::Value, offset=nothing::Union{Nothing, Value}; res::IR.Type, pack=nothing, shape, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[tmemAddr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("shape", shape), ]
    !isnothing(offset) && push!(operands, offset)
    !isnothing(pack) && push!(attributes, NamedAttribute("pack", pack))
    
    create_operation(
        "nvvm.tcgen05.ld", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tcgen05_mma_smem_desc`

The `nvvm.tcgen05_mma_smem_desc` constructs a Shared Memory descriptor
for tcgen05.mma. This descriptor is a 64-bit value which describes the
properties of multiplicand matrix in shared memory including its location
in the shared memory of the current CTA.

```
+-----------+------+------------------------------------------------------+
| Bit-field | Size | Description                                          |
+-----------+------+------------------------------------------------------+
| 0-13      | 14   | Matrix start address                                 |
| 14-15     | 2    | Reserved                                             |
| 16-29     | 14   | Leading dim relative-offset (or) absolute-address    |
| 30-31     | 2    | Reserved                                             |
| 32-45     | 14   | Stride dimension byte offset                         |
| 46-48     | 3    | Fixed constant value of 0b001                        |
| 49-51     | 3    | Matrix base offset                                   |
| 52        | 1    | Leading dimension stride mode:                       |
|           |      |   0: byte offset relative                            |
|           |      |   1: byte address absolute                           |
| 53-60     | 8    | Fixed constant value of 0xb00000000                  |
| 61-63     | 3    | Swizzling mode:                                      |
|           |      |   0: No swizzling                                    |
|           |      |   1: 128-Byte with 32B atomic swizzling              |
|           |      |   2: 128-Byte swizzling                              |
|           |      |   4: 64-Byte swizzling                               |
|           |      |   6: 32-Byte swizzling                               |
|           |      |   (Values 3, 5 and 7 are invalid)                    |
+-----------+------+------------------------------------------------------+    
```

# Example
```mlir
  %desc = nvvm.tcgen05.mma_smem_desc (%startAddr, %leadingDimOffset, %strideDimOffset,
                                      %baseOffset, %leadingDimMode, %swizzleMode) : (i32, i32, i32, i8, i1, i8) -> i64
```
[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor)
"""
function tcgen05_mma_smem_desc(startAddr::Value, leadingDimOffset::Value, strideDimOffset::Value, baseOffset::Value, leadingDimMode::Value, swizzleMode::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[startAddr, leadingDimOffset, strideDimOffset, baseOffset, leadingDimMode, swizzleMode, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.tcgen05.mma_smem_desc", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tcgen05_relinquish_alloc_permit`

The `tcgen05.relinquish_alloc_permit` Op specifies that the CTA
of the executing thread is relinquishing the right to allocate
Tensor Memory. So, it is illegal for a CTA to perform `tcgen05.alloc`
after any of its constituent threads execute `tcgen05.relinquish_alloc_permit`.
[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)
"""
function tcgen05_relinquish_alloc_permit(; group=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(group) && push!(attributes, NamedAttribute("group", group))
    
    create_operation(
        "nvvm.tcgen05.relinquish_alloc_permit", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tcgen05_shift`

The `tcgen05.shift` is an asynchronous instruction which initiates
the shifting of 32-byte elements downwards across all the rows,
except the last, by one row. The operand `taddr` specifies the base
address of the matrix in Tensor Memory whose rows must be down shifted.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-shift)
"""
function tcgen05_shift(taddr::Value; group=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[taddr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(group) && push!(attributes, NamedAttribute("group", group))
    
    create_operation(
        "nvvm.tcgen05.shift", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tcgen05_st`

Instruction `tcgen05.st` asynchronously stores data from the source register `r`
into the Tensor Memory at the location specified by the 32-bit address operand
`tmemAddr`, collectively across all threads of the warps.

The `shape` and the `num` attribute together determines the total dimension of
the data which is stored to the Tensor Memory. The `shape` indicates the base
dimension of data to be accessed. The `num` attribute indicates the repeat
factor on the base dimension resulting in the total dimension of the data that
is accessed.

The shape `16x32bx2` performs two accesses into Tensor Memory of the shape
`16x32b`. The base address of the first access is specified by `tmemAddr`
and the base address of the second access is specified by
`tmemAddr + offset`, where `offset` is an immediate argument.

The unit attribute `unpack` can be used to unpack a 32-bit element
in the register into two 16-bit elements and store them in adjacent columns.

The following table describes the size of the vector for various combinations
of `num` and `shape` attributes:
```
|=====================================================================|
| num/shape      |     16x32bx2/16x64b/32x32b |  16x128b   | 16x256b  |
|=====================================================================|
| x1             |          1                 |    2       |    4     |
| x2             |          2                 |    4       |    8     |
| x4             |          4                 |    8       |    16    |
| x8             |          8                 |    16      |    32    |
| x16            |          16                |    32      |    64    |
| x32            |          32                |    64      |    128   |
| x64            |          64                |    128     |    NA    |
| x128           |          128               |    NA      |    NA    |
|=====================================================================|
```

# Example
```mlir
  nvvm.tcgen05.st %tmemAddr, %val, %offset unpack {
    shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>,
  } : <2xi32>
```

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st)
"""
function tcgen05_st(tmemAddr::Value, val::Value, offset=nothing::Union{Nothing, Value}; unpack=nothing, shape, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[tmemAddr, val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("shape", shape), ]
    !isnothing(offset) && push!(operands, offset)
    !isnothing(unpack) && push!(attributes, NamedAttribute("unpack", unpack))
    
    create_operation(
        "nvvm.tcgen05.st", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`tcgen05_wait`

The `tcgen05.wait<load>` causes the executing thread to block until
all prior `tcgen05.ld` operations issued by the executing thread
have completed. Similarly, the `tcgen05.wait<store>` causes the executing
thread to block until all prior `tcgen05.st` operations issued by the
executing thread have completed.
[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-wait)
"""
function tcgen05_wait(; kind, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("kind", kind), ]
    
    create_operation(
        "nvvm.tcgen05.wait", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_tid_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.tid.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_tid_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.tid.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_tid_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.tid.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`vote_sync`

The `vote.sync` op will cause executing thread to wait until all non-exited
threads corresponding to membermask have executed `vote.sync` with the same
qualifiers and same membermask value before resuming execution.

The vote operation kinds are:
- `any`: True if source predicate is True for some thread in membermask.
- `all`: True if source predicate is True for all non-exited threads in
  membermask. 
- `uni`: True if source predicate has the same value in all non-exited
  threads in membermask.
- `ballot`: In the ballot form, the destination result is a 32 bit integer.
  In this form, the predicate from each thread in membermask are copied into
  the corresponding bit position of the result, where the bit position
  corresponds to the thread’s lane id.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync)
"""
function vote_sync(mask::Value, pred::Value; res::IR.Type, kind, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[mask, pred, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("kind", kind), ]
    
    create_operation(
        "nvvm.vote.sync", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_load(ptr::Value, stride::Value; res::IR.Type, m, n, k, layout, eltype, frag, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, stride, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("m", m), NamedAttribute("n", n), NamedAttribute("k", k), NamedAttribute("layout", layout), NamedAttribute("eltype", eltype), NamedAttribute("frag", frag), ]
    
    create_operation(
        "nvvm.wmma.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_mma(args::Vector{Value}; res::IR.Type, m, n, k, layoutA, layoutB, eltypeA, eltypeB, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("m", m), NamedAttribute("n", n), NamedAttribute("k", k), NamedAttribute("layoutA", layoutA), NamedAttribute("layoutB", layoutB), NamedAttribute("eltypeA", eltypeA), NamedAttribute("eltypeB", eltypeB), ]
    
    create_operation(
        "nvvm.wmma.mma", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_store(ptr::Value, args::Vector{Value}, stride::Value; m, n, k, layout, eltype, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, args..., stride, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("m", m), NamedAttribute("n", n), NamedAttribute("k", k), NamedAttribute("layout", layout), NamedAttribute("eltype", eltype), ]
    
    create_operation(
        "nvvm.wmma.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_nwarpid(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.nwarpid", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_warpid(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.warpid", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function read_ptx_sreg_warpsize(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "nvvm.read.ptx.sreg.warpsize", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`wgmma_fence_aligned`

Enforce an ordering of register accesses between warpgroup level matrix 
multiplication and other operations. 

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence)
"""
function wgmma_fence_aligned(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.wgmma.fence.aligned", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`wgmma_commit_group_sync_aligned`

Commits all prior uncommitted warpgroup level matrix multiplication operations.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group)
"""
function wgmma_commit_group_sync_aligned(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "nvvm.wgmma.commit.group.sync.aligned", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`wgmma_mma_async`

The warpgroup (128 threads) level matrix multiply and accumulate operation 
has either of the following forms, where matrix D is called accumulator:
  D = A * B + D
  D = A * B, where the input from accumulator D is disabled.

Supported shapes:  
```
|--------------|--------------|------------|--------------|---------------|
|              |              |            |              |f16+=e4m3*e4m3 |
|              |              |            |              |f16+=e5m2*e5m2 |
|f32+=tf32*tf32|f16+=f16 *f16 | s32+=s8*s8 |s32 += b1 * b1|f16+=e5m2*e4m3 |
|              |f32+=f16 *f16 | s32+=u8*u8 |              |f16+=e4m3*e5m2 |
|              |f32+=bf16*bf16| s32+=u8*u8 |              |f16+=e4m3*e5m2 |
|              |f32+=bf16*bf16| s32+=s8*u8 |              |f32+=e4m3*e4m3 |
|              |              | s32+=u8*s8 |              |f32+=e5m2*e5m2 |
|              |              |            |              |f32+=e4m3*e5m2 |
|              |              |            |              |f32+=e4m3*e5m2 |
|--------------|--------------|------------|--------------|---------------|
|   .m64n8k8   |  .m64n8k16   | .m64n8k32  | .m64n8k256   | .m64n8k32     |
|   .m64n16k8  |  .m64n16k16  | .m64n16k32 | .m64n16k256  | .m64n16k32    |
|   .m64n24k8  |  .m64n24k16  | .m64n24k32 | .m64n24k256  | .m64n24k32    |
|   .m64n32k8  |  .m64n32k16  | .m64n32k32 | .m64n32k256  | .m64n32k32    |
|   .m64n40k8  |  .m64n40k16  | .m64n48k32 | .m64n48k256  | .m64n40k32    |
|   .m64n48k8  |  .m64n48k16  | .m64n64k32 | .m64n64k256  | .m64n48k32    |
|   .m64n56k8  |  .m64n56k16  | .m64n80k32 | .m64n80k256  | .m64n56k32    |
|   .m64n64k8  |  .m64n64k16  | .m64n96k32 | .m64n96k256  | .m64n64k32    |
|   .m64n72k8  |  .m64n72k16  | .m64n112k32| .m64n112k256 | .m64n72k32    |
|   .m64n80k8  |  .m64n80k16  | .m64n128k32| .m64n128k256 | .m64n80k32    |
|   .m64n88k8  |  .m64n88k16  | .m64n144k32| .m64n144k256 | .m64n88k32    |
|   .m64n96k8  |  .m64n96k16  | .m64n160k32| .m64n160k256 | .m64n96k32    |
|   .m64n104k8 |  .m64n104k16 | .m64n176k32| .m64n176k256 | .m64n104k32   |
|   .m64n112k8 |  .m64n112k16 | .m64n192k32| .m64n192k256 | .m64n112k32   |
|   .m64n120k8 |  .m64n120k16 | .m64n208k32| .m64n208k256 | .m64n120k32   |
|   .m64n128k8 |  .m64n128k16 | .m64n224k32| .m64n224k256 | .m64n128k32   |
|   .m64n136k8 |  .m64n136k16 | .m64n240k32| .m64n240k256 | .m64n136k32   |
|   .m64n144k8 |  .m64n144k16 | .m64n256k32| .m64n256k256 | .m64n144k32   |
|   .m64n152k8 |  .m64n152k16 |            |              | .m64n152k32   |
|   .m64n160k8 |  .m64n160k16 |            |              | .m64n160k32   |
|   .m64n168k8 |  .m64n168k16 |            |              | .m64n168k32   |
|   .m64n176k8 |  .m64n176k16 |            |              | .m64n176k32   |
|   .m64n184k8 |  .m64n184k16 |            |              | .m64n184k32   |
|   .m64n192k8 |  .m64n192k16 |            |              | .m64n192k32   |
|   .m64n200k8 |  .m64n200k16 |            |              | .m64n200k32   |
|   .m64n208k8 |  .m64n208k16 |            |              | .m64n208k32   |
|   .m64n216k8 |  .m64n216k16 |            |              | .m64n216k32   |
|   .m64n224k8 |  .m64n224k16 |            |              | .m64n224k32   |
|   .m64n232k8 |  .m64n232k16 |            |              | .m64n232k32   |
|   .m64n240k8 |  .m64n240k16 |            |              | .m64n240k32   |
|   .m64n248k8 |  .m64n248k16 |            |              | .m64n248k32   |
|   .m64n256k8 |  .m64n256k16 |            |              | .m64n256k32   |
|--------------|--------------|------------|--------------|---------------|
```


[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions)
"""
function wgmma_mma_async(inouts::Value, descriptorA::Value, descriptorB::Value; results::IR.Type, shape, typeA, typeB, typeD, scaleD, scaleA, scaleB, layoutA, layoutB, satfinite=nothing, location=Location())
    op_ty_results = IR.Type[results, ]
    operands = Value[inouts, descriptorA, descriptorB, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("shape", shape), NamedAttribute("typeA", typeA), NamedAttribute("typeB", typeB), NamedAttribute("typeD", typeD), NamedAttribute("scaleD", scaleD), NamedAttribute("scaleA", scaleA), NamedAttribute("scaleB", scaleB), NamedAttribute("layoutA", layoutA), NamedAttribute("layoutB", layoutB), ]
    !isnothing(satfinite) && push!(attributes, NamedAttribute("satfinite", satfinite))
    
    create_operation(
        "nvvm.wgmma.mma_async", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`wgmma_wait_group_sync_aligned`

Signal the completion of a preceding warpgroup operation.

[For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)
"""
function wgmma_wait_group_sync_aligned(; group, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("group", group), ]
    
    create_operation(
        "nvvm.wgmma.wait.group.sync.aligned", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes
import ...API


"""
`ballot`

Ballot provides a bit mask containing the 1-bit predicate value from each lane.
The nth bit of the result contains the 1 bit contributed by the nth warp lane.
"""
function ballot(pred::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[pred, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.ballot", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function barrier(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.barrier", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function s_barrier_signal(; id, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("id", id), ]
    
    create_operation(
        "rocdl.s.barrier.signal", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function s_barrier_wait(; id, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("id", id), ]
    
    create_operation(
        "rocdl.s.barrier.wait", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workgroup_dim_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "rocdl.workgroup.dim.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workgroup_dim_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "rocdl.workgroup.dim.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workgroup_dim_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "rocdl.workgroup.dim.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workgroup_id_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "rocdl.workgroup.id.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workgroup_id_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "rocdl.workgroup.id.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workgroup_id_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "rocdl.workgroup.id.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_f32_bf8`

Convert 8-bit bf8 value from the `byteSel`th bit of `srcA` to fp32.
"""
function cvt_f32_bf8(srcA::Value; res::IR.Type, byteSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("byteSel", byteSel), ]
    
    create_operation(
        "rocdl.cvt.f32.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_f32_fp8`

Convert 8-bit fp8 value from the `byteSel`th bit of `srcA` to fp32.
"""
function cvt_f32_fp8(srcA::Value; res::IR.Type, byteSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("byteSel", byteSel), ]
    
    create_operation(
        "rocdl.cvt.f32.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_pk_bf8_f32`

Convert `srcA` and `srcB` to bf8 and store into the low/high word of
`old`, preserving the other word.
"""
function cvt_pk_bf8_f32(srcA::Value, srcB::Value, old::Value; res::IR.Type, wordSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, srcB, old, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("wordSel", wordSel), ]
    
    create_operation(
        "rocdl.cvt.pk.bf8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_pk_f32_bf8`

Convert `src` based on \$wordSel to packed fp32,
"""
function cvt_pk_f32_bf8(src::Value; res::IR.Type, wordSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("wordSel", wordSel), ]
    
    create_operation(
        "rocdl.cvt.pk.f32.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_pk_f32_fp8`

Convert `src` based on \$wordSel to packed fp32.
"""
function cvt_pk_f32_fp8(src::Value; res::IR.Type, wordSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("wordSel", wordSel), ]
    
    create_operation(
        "rocdl.cvt.pk.f32.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_pk_fp8_f32`

Convert `srcA` and `srcB` to fp8 and store into the low/high word of
`old`, preserving the other word.
"""
function cvt_pk_fp8_f32(srcA::Value, srcB::Value, old::Value; res::IR.Type, wordSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, srcB, old, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("wordSel", wordSel), ]
    
    create_operation(
        "rocdl.cvt.pk.fp8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_pkrtz`

Convert two f32 values into a packed vector<2xf16>.
"""
function cvt_pkrtz(srcA::Value, srcB::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, srcB, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.pkrtz", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_f16_bf8`

Convert a bf8 byte from `src`, selected by
`srcSelIndex`, to f16 while multiplying it by the expontent of `scale`,
and place it into the `dstLoHiSel`th bit
of `oldVdst` preserving the other element of that vector in
the return value.

The bytes are stored as an `i32` and not a `<4 x i8>`.
"""
function cvt_scalef32_f16_bf8(oldVdst::Value, src::Value, scale::Value; res::IR.Type, srcSelIndex, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcSelIndex", srcSelIndex), NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    create_operation(
        "rocdl.cvt.scalef32.f16.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_f16_fp8`

Convert a fp8 byte from `src`, selected by
`srcSelIndex`, to f16 while multiplying it by the expontent of `scale`,
and place it into the `dstLoHiSel`th bit
of `oldVdst` preserving the other element of that vector in
the return value.

The bytes are stored as an `i32` and not a `<4 x i8>`.
"""
function cvt_scalef32_f16_fp8(oldVdst::Value, src::Value, scale::Value; res::IR.Type, srcSelIndex, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcSelIndex", srcSelIndex), NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    create_operation(
        "rocdl.cvt.scalef32.f16.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_f32_bf8`

Convert a bf8 byte from `src`, selected by
`srcSelIndex`, to f32, multiplying it by the exponent of `scale`.

The bytes are stored in an `i32`, not a `<4 x i8>`.
"""
function cvt_scalef32_f32_bf8(src::Value, scale::Value; res::IR.Type, srcSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcSelIndex", srcSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.f32.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_f32_fp8`

Convert a fp8 byte from `src`, selected by
`srcSelIndex`, to f32, multiplying it by the exponent of `scale`.

The bytes are stored in an `i32`, not a `<4 x i8>`.
"""
function cvt_scalef32_f32_fp8(src::Value, scale::Value; res::IR.Type, srcSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcSelIndex", srcSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.f32.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_bf6_bf16`

Convert 32 packed bf16 values to packed bf6, dividing by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_bf6_bf16(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.pk32.bf6.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_bf6_f16`

Convert 32 packed f16 values to packed bf6, dividing by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_bf6_f16(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.pk32.bf6.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_bf16_bf6`

Convert 32 packed bf6 values to packed bf16, multiplying by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_bf16_bf6(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.pk32.bf16.bf6", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_bf16_fp6`

Convert 32 packed fp6 values to packed bf16, multiplying by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_bf16_fp6(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.pk32.bf16.fp6", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_f16_bf6`

Convert 32 packed bf6 values to packed f16, multiplying by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_f16_bf6(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.pk32.f16.bf6", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_f16_fp6`

Convert 32 packed fp6 values to packed f16, multiplying by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_f16_fp6(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.pk32.f16.fp6", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_f32_bf6`

Convert 32 packed bf6 values to packed f32, multiplying by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_f32_bf6(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.pk32.f32.bf6", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_f32_fp6`

Convert 32 packed fp6 values to packed f32, multiplying by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_f32_fp6(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.pk32.f32.fp6", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_fp6_bf16`

Convert 32 packed bf16 values to packed fp6, dividing by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_fp6_bf16(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.pk32.fp6.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk32_fp6_f16`

Convert 32 packed f16 values to packed fp6, dividing by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_pk32_fp6_f16(src::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.pk32.fp6.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_bf8_bf16`

Convert two bf16 values in `src0` to two bf8 bytes, dividing by the exponent in `scale`. The bytes are
packed into a 16-bit value which is inserted into `oldVdst` at the
`dstLoHiSel` position, with the entire updated vector being returned.
"""
function cvt_scalef32_pk_bf8_bf16(oldVdst::Value, src0::Value, scale::Value; res::IR.Type, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.bf8.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_bf8_f16`

Convert two f16 values in `src0` to two bf8 bytes, dividing by the exponent in `scale`. The bytes are
packed into a 16-bit value which is inserted into `oldVdst` at the
`dstLoHiSel` position, with the entire updated vector being returned.
"""
function cvt_scalef32_pk_bf8_f16(oldVdst::Value, src0::Value, scale::Value; res::IR.Type, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.bf8.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_bf8_f32`

Convert two f32 values in `src0` and `src1` to two bf8 bytes,
dividing by the exponent in `scale`. The bytes are packed into
a 16-bit value which is inserted into `oldVdst` at the `dstLoHiSel`
position, with the entire updated vector being returned.
"""
function cvt_scalef32_pk_bf8_f32(oldVdst::Value, src0::Value, src1::Value, scale::Value; res::IR.Type, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, src1, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.bf8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_bf16_bf8`

Convert two packed bf8 values in `src0` to two bf16 values, multiplying by the exponent in `scale`.
The two values to be converted are selected from the low or high half
of `src` (a packed vector represented as an `i32`)
on the basis of `srcLoHiSel`.
"""
function cvt_scalef32_pk_bf16_bf8(src::Value, scale::Value; res::IR.Type, srcLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcLoHiSel", srcLoHiSel), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.bf16.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_bf16_fp4`

Convert two packed fp4 (f4E2M1) values  stored as one byte of a 32-bit integer
to packed bf16, multiplying by the exponent part of `scale`
before doing so.

The byte to convert is chosen by `srcSelIndex`.
"""
function cvt_scalef32_pk_bf16_fp4(src::Value, scale::Value; res::IR.Type, srcSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcSelIndex", srcSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.bf16.fp4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_bf16_fp8`

Convert two packed fp8 values in `src0` to two bf16 values, multiplying by the exponent in `scale`.
The two values to be converted are selected from the low or high half
of `src` (a packed vector represented as an `i32`)
on the basis of `srcLoHiSel`.
"""
function cvt_scalef32_pk_bf16_fp8(src::Value, scale::Value; res::IR.Type, srcLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcLoHiSel", srcLoHiSel), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.bf16.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_f16_bf8`

Convert two packed bf8 values in `src0` to two f16 values, multiplying by the exponent in `scale`.
The two values to be converted are selected from the low or high half
of `src` (a packed vector represented as an `i32`)
on the basis of `srcLoHiSel`.
"""
function cvt_scalef32_pk_f16_bf8(src::Value, scale::Value; res::IR.Type, srcLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcLoHiSel", srcLoHiSel), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.f16.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_f16_fp4`

Convert two packed fp4 (f4E2M1) values  stored as one byte of a 32-bit integer
to packed f16, multiplying by the exponent part of `scale`
before doing so.

The byte to convert is chosen by `srcSelIndex`.
"""
function cvt_scalef32_pk_f16_fp4(src::Value, scale::Value; res::IR.Type, srcSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcSelIndex", srcSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.f16.fp4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_f16_fp8`

Convert two packed fp8 values in `src0` to two f16 values, multiplying by the exponent in `scale`.
The two values to be converted are selected from the low or high half
of `src` (a packed vector represented as an `i32`)
on the basis of `srcLoHiSel`.
"""
function cvt_scalef32_pk_f16_fp8(src::Value, scale::Value; res::IR.Type, srcLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcLoHiSel", srcLoHiSel), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.f16.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_f32_bf8`

Convert two packed bf8 values in `src0` to two f32 values, multiplying by the exponent in `scale`.
The two values to be converted are selected from the low or high half
of `src` (a packed vector represented as an `i32`)
on the basis of `srcLoHiSel`.
"""
function cvt_scalef32_pk_f32_bf8(src::Value, scale::Value; res::IR.Type, srcLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcLoHiSel", srcLoHiSel), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.f32.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_f32_fp4`

Convert two packed fp4 (f4E2M1) values  stored as one byte of a 32-bit integer
to packed f32, multiplying by the exponent part of `scale`
before doing so.

The byte to convert is chosen by `srcSelIndex`.
"""
function cvt_scalef32_pk_f32_fp4(src::Value, scale::Value; res::IR.Type, srcSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcSelIndex", srcSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.f32.fp4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_f32_fp8`

Convert two packed fp8 values in `src0` to two f32 values, multiplying by the exponent in `scale`.
The two values to be converted are selected from the low or high half
of `src` (a packed vector represented as an `i32`)
on the basis of `srcLoHiSel`.
"""
function cvt_scalef32_pk_f32_fp8(src::Value, scale::Value; res::IR.Type, srcLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("srcLoHiSel", srcLoHiSel), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.f32.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_fp4_bf16`

Convert two packed bf16 values to packed
fp4, dividing by the exponent part of `scale`
before doing so.

The two scaled values are packed  into a byte.
That byte is used to update the `dstSelIndex`th
byte of `oldVdst`, which is returned in its entirity.
"""
function cvt_scalef32_pk_fp4_bf16(oldVdst::Value, src::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.fp4.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_fp4_f16`

Convert two packed f16 values to packed
fp4, dividing by the exponent part of `scale`
before doing so.

The two scaled values are packed  into a byte.
That byte is used to update the `dstSelIndex`th
byte of `oldVdst`, which is returned in its entirity.
"""
function cvt_scalef32_pk_fp4_f16(oldVdst::Value, src::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.fp4.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_fp4_f32`

Convert two single-precision float values, passed in `src0` and `src1`
into two fp4 values, dividing them by the expontent part of `scale`
before doing so.

The two scaled values are packed  into a byte.
That byte is used to update the `dstSelIndex`th
byte of `oldVdst`, which is returned in its entirity.
"""
function cvt_scalef32_pk_fp4_f32(oldVdst::Value, src0::Value, src1::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, src1, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.fp4.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_fp8_bf16`

Convert two bf16 values in `src0` to two fp8 bytes, dividing by the exponent in `scale`. The bytes are
packed into a 16-bit value which is inserted into `oldVdst` at the
`dstLoHiSel` position, with the entire updated vector being returned.
"""
function cvt_scalef32_pk_fp8_bf16(oldVdst::Value, src0::Value, scale::Value; res::IR.Type, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.fp8.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_fp8_f16`

Convert two f16 values in `src0` to two fp8 bytes, dividing by the exponent in `scale`. The bytes are
packed into a 16-bit value which is inserted into `oldVdst` at the
`dstLoHiSel` position, with the entire updated vector being returned.
"""
function cvt_scalef32_pk_fp8_f16(oldVdst::Value, src0::Value, scale::Value; res::IR.Type, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.fp8.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_pk_fp8_f32`

Convert two f32 values in `src0` and `src1` to two fp8 bytes,
dividing by the exponent in `scale`. The bytes are packed into
a 16-bit value which is inserted into `oldVdst` at the `dstLoHiSel`
position, with the entire updated vector being returned.
"""
function cvt_scalef32_pk_fp8_f32(oldVdst::Value, src0::Value, src1::Value, scale::Value; res::IR.Type, dstLoHiSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, src1, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstLoHiSel", dstLoHiSel), ]
    
    create_operation(
        "rocdl.cvt.scalef32.pk.fp8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_bf8_bf16`

Convert a bf16 value in `src0` to a bf8 bytes, dividing by the exponent in `scale` and using `seed`
for stochiastic rounding. Place the resulting byte in the
`dstSelIndex`th bit of `oldVdst` and return the entire packed vector,
which is stored as an `i32`.
"""
function cvt_scalef32_sr_bf8_bf16(oldVdst::Value, src0::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.bf8.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_bf8_f16`

Convert a f16 value in `src0` to a bf8 bytes, dividing by the exponent in `scale` and using `seed`
for stochiastic rounding. Place the resulting byte in the
`dstSelIndex`th bit of `oldVdst` and return the entire packed vector,
which is stored as an `i32`.
"""
function cvt_scalef32_sr_bf8_f16(oldVdst::Value, src0::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.bf8.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_bf8_f32`

Convert a f32 value in `src0` to a bf8 bytes, dividing by the exponent in `scale` and using `seed`
for stochiastic rounding. Place the resulting byte in the
`dstSelIndex`th bit of `oldVdst` and return the entire packed vector,
which is stored as an `i32`.
"""
function cvt_scalef32_sr_bf8_f32(oldVdst::Value, src0::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.bf8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_fp8_bf16`

Convert a bf16 value in `src0` to a fp8 bytes, dividing by the exponent in `scale` and using `seed`
for stochiastic rounding. Place the resulting byte in the
`dstSelIndex`th bit of `oldVdst` and return the entire packed vector,
which is stored as an `i32`.
"""
function cvt_scalef32_sr_fp8_bf16(oldVdst::Value, src0::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.fp8.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_fp8_f16`

Convert a f16 value in `src0` to a fp8 bytes, dividing by the exponent in `scale` and using `seed`
for stochiastic rounding. Place the resulting byte in the
`dstSelIndex`th bit of `oldVdst` and return the entire packed vector,
which is stored as an `i32`.
"""
function cvt_scalef32_sr_fp8_f16(oldVdst::Value, src0::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.fp8.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_fp8_f32`

Convert a f32 value in `src0` to a fp8 bytes, dividing by the exponent in `scale` and using `seed`
for stochiastic rounding. Place the resulting byte in the
`dstSelIndex`th bit of `oldVdst` and return the entire packed vector,
which is stored as an `i32`.
"""
function cvt_scalef32_sr_fp8_f32(oldVdst::Value, src0::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src0, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.fp8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk32_bf6_bf16`

Convert 32 packed bf16 values to packed bf6, dividing by the exponent part of `scale`
before doing so and applying random rounding derived from
`seed`.
"""
function cvt_scalef32_sr_pk32_bf6_bf16(src::Value, seed::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.pk32.bf6.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk32_bf6_f16`

Convert 32 packed f16 values to packed bf6, dividing by the exponent part of `scale`
before doing so and applying random rounding derived from
`seed`.
"""
function cvt_scalef32_sr_pk32_bf6_f16(src::Value, seed::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.pk32.bf6.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk32_bf6_f32`

Convert 32 packed f32 values to packed bf6, dividing by the exponent part of `scale`
before doing so and applying random rounding derived from
`seed`.
"""
function cvt_scalef32_sr_pk32_bf6_f32(src::Value, seed::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.pk32.bf6.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk32_fp6_bf16`

Convert 32 packed bf16 values to packed fp6, dividing by the exponent part of `scale`
before doing so and applying random rounding derived from
`seed`.
"""
function cvt_scalef32_sr_pk32_fp6_bf16(src::Value, seed::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.pk32.fp6.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk32_fp6_f16`

Convert 32 packed f16 values to packed fp6, dividing by the exponent part of `scale`
before doing so and applying random rounding derived from
`seed`.
"""
function cvt_scalef32_sr_pk32_fp6_f16(src::Value, seed::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.pk32.fp6.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk32_fp6_f32`

Convert 32 packed f32 values to packed fp6, dividing by the exponent part of `scale`
before doing so and applying random rounding derived from
`seed`.
"""
function cvt_scalef32_sr_pk32_fp6_f32(src::Value, seed::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.pk32.fp6.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk_fp4_bf16`

Convert two packed bf16 values to packed
fp4, dividing by the exponent part of `scale`
before doing so and using `seed` as the random seed for
stochiastic rounding.

The two scaled values are packed (little-endian)
into a byte. That byte is used to update the `dstSelIndex`th
byte of `oldVdst`, which is returned in its entirity.
"""
function cvt_scalef32_sr_pk_fp4_bf16(oldVdst::Value, src::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.pk.fp4.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk_fp4_f16`

Convert two packed f16 values to packed
fp4, dividing by the exponent part of `scale`
before doing so and using `seed` as the random seed for
stochiastic rounding.

The two scaled values are packed (little-endian)
into a byte. That byte is used to update the `dstSelIndex`th
byte of `oldVdst`, which is returned in its entirity.
"""
function cvt_scalef32_sr_pk_fp4_f16(oldVdst::Value, src::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.pk.fp4.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_sr_pk_fp4_f32`

Convert two packed f32 values to packed
fp4, dividing by the exponent part of `scale`
before doing so and using `seed` as the random seed for
stochiastic rounding.

The two scaled values are packed (little-endian)
into a byte. That byte is used to update the `dstSelIndex`th
byte of `oldVdst`, which is returned in its entirity.
"""
function cvt_scalef32_sr_pk_fp4_f32(oldVdst::Value, src::Value, seed::Value, scale::Value; res::IR.Type, dstSelIndex, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[oldVdst, src, seed, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dstSelIndex", dstSelIndex), ]
    
    create_operation(
        "rocdl.cvt.scalef32.sr.pk.fp4.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_2xpk16_bf6_f32`

Convert 32 single-precision float values, packed into two length-16
vectors that will be logically concanenated, to packed bf6, dividing by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_2xpk16_bf6_f32(src0::Value, src1::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src0, src1, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.2xpk16.bf6.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_scalef32_2xpk16_fp6_f32`

Convert 32 single-precision float values, packed into two length-16
vectors that will be logically concanenated, to packed fp6, dividing by the exponent part of `scale`
before doing so.
"""
function cvt_scalef32_2xpk16_fp6_f32(src0::Value, src1::Value, scale::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src0, src1, scale, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.cvt.scalef32.2xpk16.fp6.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_sr_bf8_f32`

Convert `srcA` to bf8, adding the rounding factor from `srcB`,
and store into the `byteSel`th byte of `old`, preserving the others.
"""
function cvt_sr_bf8_f32(srcA::Value, srcB::Value, old::Value; res::IR.Type, byteSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, srcB, old, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("byteSel", byteSel), ]
    
    create_operation(
        "rocdl.cvt.sr.bf8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cvt_sr_fp8_f32`

Convert `srcA` to fp8, adding the rounding factor from `srcB`,
and store into the `byteSel`th byte of `old`, preserving the others.
"""
function cvt_sr_fp8_f32(srcA::Value, srcB::Value, old::Value; res::IR.Type, byteSel, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcA, srcB, old, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("byteSel", byteSel), ]
    
    create_operation(
        "rocdl.cvt.sr.fp8.f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function update_dpp(old::Value, src::Value; res::IR.Type, dppCtrl, rowMask, bankMask, boundCtrl, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[old, src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dppCtrl", dppCtrl), NamedAttribute("rowMask", rowMask), NamedAttribute("bankMask", bankMask), NamedAttribute("boundCtrl", boundCtrl), ]
    
    create_operation(
        "rocdl.update.dpp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_bpermute(index::Value, src::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[index, src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.ds_bpermute", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_swizzle(src::Value, offset::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, offset, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.ds_swizzle", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function global_load_lds(globalPtr::Value, ldsPtr::Value; size, offset, aux, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[globalPtr, ldsPtr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("size", size), NamedAttribute("offset", offset), NamedAttribute("aux", aux), ]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "rocdl.global.load.lds", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function grid_dim_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "rocdl.grid.dim.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function grid_dim_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "rocdl.grid.dim.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function grid_dim_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "rocdl.grid.dim.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function iglp_opt(; variant, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("variant", variant), ]
    
    create_operation(
        "rocdl.iglp.opt", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function load_to_lds(globalPtr::Value, ldsPtr::Value; size, offset, aux, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[globalPtr, ldsPtr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("size", size), NamedAttribute("offset", offset), NamedAttribute("aux", aux), ]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "rocdl.load.to.lds", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function make_buffer_rsrc(base::Value, stride::Value, numRecords::Value, flags::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[base, stride, numRecords, flags, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.make.buffer.rsrc", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbcnt_hi(in0::Value, in1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in0, in1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mbcnt.hi", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mbcnt_lo(in0::Value, in1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in0, in1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mbcnt.lo", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`permlanex16`

Performs a `permlanex16` operation with the given operands, applying the
permutation specified by \$fi to the provided inputs.
"""
function permlanex16(old::Value, src0::Value, src1::Value, src2::Value; res::IR.Type, fi, boundControl, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[old, src0, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("fi", fi), NamedAttribute("boundControl", boundControl), ]
    
    create_operation(
        "rocdl.permlanex16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_buffer_atomic_cmpswap(src::Value, cmp::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, cmp, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.raw.buffer.atomic.cmpswap", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_buffer_atomic_fadd(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.raw.buffer.atomic.fadd", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_buffer_atomic_fmax(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.raw.buffer.atomic.fmax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_buffer_atomic_smax(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.raw.buffer.atomic.smax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_buffer_atomic_umin(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.raw.buffer.atomic.umin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_buffer_load(rsrc::Value, offset::Value, soffset::Value, aux::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.raw.buffer.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_buffer_store(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.raw.buffer.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_atomic_cmpswap(src::Value, cmp::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; res::IR.Type, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, cmp, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "rocdl.raw.ptr.buffer.atomic.cmpswap", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_atomic_fadd(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "rocdl.raw.ptr.buffer.atomic.fadd", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_atomic_fmax(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "rocdl.raw.ptr.buffer.atomic.fmax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_atomic_smax(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "rocdl.raw.ptr.buffer.atomic.smax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_atomic_umin(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "rocdl.raw.ptr.buffer.atomic.umin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_load_lds(rsrc::Value, ldsPtr::Value, size::Value, voffset::Value, soffset::Value, offset::Value, aux::Value; alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[rsrc, ldsPtr, size, voffset, soffset, offset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "rocdl.raw.ptr.buffer.load.lds", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_load(rsrc::Value, offset::Value, soffset::Value, aux::Value; res::IR.Type, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "rocdl.raw.ptr.buffer.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function raw_ptr_buffer_store(vdata::Value, rsrc::Value, offset::Value, soffset::Value, aux::Value; alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vdata, rsrc, offset, soffset, aux, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "rocdl.raw.ptr.buffer.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`readlane`

Get the value in lane `src1` from input `src0`.
"""
function readlane(src0::Value, src1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src0, src1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.readlane", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function s_barrier(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.s.barrier", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function s_sleep(; count, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("count", count), ]
    
    create_operation(
        "rocdl.s.sleep", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function s_waitcnt(; bitfield, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("bitfield", bitfield), ]
    
    create_operation(
        "rocdl.s.waitcnt", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function sched_barrier(; mask, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mask", mask), ]
    
    create_operation(
        "rocdl.sched.barrier", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function sched_group_barrier(; mask, size, groupId, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mask", mask), NamedAttribute("size", size), NamedAttribute("groupId", groupId), ]
    
    create_operation(
        "rocdl.sched.group.barrier", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function s_setprio(; priority, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("priority", priority), ]
    
    create_operation(
        "rocdl.s.setprio", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workitem_id_x(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "rocdl.workitem.id.x", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workitem_id_y(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "rocdl.workitem.id.y", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function workitem_id_z(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "rocdl.workitem.id.z", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function s_wait_dscnt(; id, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("id", id), ]
    
    create_operation(
        "rocdl.s.wait.dscnt", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wavefrontsize(; res::IR.Type, range=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(range) && push!(attributes, NamedAttribute("range", range))
    
    create_operation(
        "rocdl.wavefrontsize", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_read_tr4_b64(ptr::Value; res::IR.Type, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "rocdl.ds.read.tr4.b64", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_read_tr6_b96(ptr::Value; res::IR.Type, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "rocdl.ds.read.tr6.b96", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_read_tr8_b64(ptr::Value; res::IR.Type, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "rocdl.ds.read.tr8.b64", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function ds_read_tr16_b64(ptr::Value; res::IR.Type, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "rocdl.ds.read.tr16.b64", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_4x4x1f32(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.4x4x1f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_4x4x2bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.4x4x2bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_4x4x4bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.4x4x4bf16.1k", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_4x4x4f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.4x4x4f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x1f32(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x1f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x2bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x2bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x4bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x4bf16.1k", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x4f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x4f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x4f32(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x4f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x8_xf32(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x8.xf32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x8bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x8bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x16bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x16bf16.1k", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x16f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x16f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x32_bf8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x32.bf8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x32_bf8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x32.bf8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x32_bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x32.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x32_f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x32.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x32_fp8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x32.fp8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_16x16x32_fp8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.16x16x32.fp8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x1f32(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x1f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x2bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x2bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x2f32(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x2f32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x4_xf32(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x4.xf32", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x4bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x4bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x4bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x4bf16.1k", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x4f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x4f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x8bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x8bf16.1k", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x8f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x8f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x16_bf8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x16.bf8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x16_bf8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x16.bf8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x16_bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x16.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x16_f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x16.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x16_fp8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x16.fp8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f32_32x32x16_fp8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f32.32x32x16.fp8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f64_4x4x4f64(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f64.4x4x4f64", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_f64_16x16x4f64(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.f64.16x16x4f64", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_4x4x4i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.i32.4x4x4i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_16x16x4i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.i32.16x16x4i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_16x16x16i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.i32.16x16x16i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_16x16x32_i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.i32.16x16x32.i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_16x16x64_i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.i32.16x16x64.i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_32x32x4i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.i32.32x32x4i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_32x32x8i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.i32.32x32x8i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_32x32x16_i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.i32.32x32x16.i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_i32_32x32x32_i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.i32.32x32x32.i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_scale_f32_16x16x128_f8f6f4(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.scale.f32.16x16x128.f8f6f4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mfma_scale_f32_32x32x64_f8f6f4(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.mfma.scale.f32.32x32x64.f8f6f4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_16x16x32_bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.smfmac.f32.16x16x32.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_16x16x32_f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.smfmac.f32.16x16x32.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_16x16x64_bf8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.smfmac.f32.16x16x64.bf8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_16x16x64_bf8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.smfmac.f32.16x16x64.bf8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_16x16x64_fp8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.smfmac.f32.16x16x64.fp8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_16x16x64_fp8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.smfmac.f32.16x16x64.fp8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_32x32x16_bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.smfmac.f32.32x32x16.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_32x32x16_f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.smfmac.f32.32x32x16.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_32x32x32_bf8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.smfmac.f32.32x32x32.bf8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_32x32x32_bf8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.smfmac.f32.32x32x32.bf8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_32x32x32_fp8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.smfmac.f32.32x32x32.fp8.bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_f32_32x32x32_fp8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.smfmac.f32.32x32x32.fp8.fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_i32_16x16x64_i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.smfmac.i32.16x16x64.i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function smfmac_i32_32x32x32_i8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.smfmac.i32.32x32x32.i8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_bf16_16x16x16_bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.wmma.bf16.16x16x16.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_f16_16x16x16_f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.wmma.f16.16x16x16.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_f32_16x16x16_bf8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.wmma.f32.16x16x16.bf8_bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_f32_16x16x16_bf8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.wmma.f32.16x16x16.bf8_fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_f32_16x16x16_bf16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.wmma.f32.16x16x16.bf16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_f32_16x16x16_f16(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.wmma.f32.16x16x16.f16", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_f32_16x16x16_fp8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.wmma.f32.16x16x16.fp8_bf8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_f32_16x16x16_fp8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.wmma.f32.16x16x16.fp8_fp8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_i32_16x16x16_iu4(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.wmma.i32.16x16x16.iu4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_i32_16x16x16_iu8(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.wmma.i32.16x16x16.iu8", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function wmma_i32_16x16x32_iu4(args::Vector{Value}; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "rocdl.wmma.i32.16x16x32.iu4", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes
import ...API



function intr_acos(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.acos", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_asin(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.asin", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_atan2(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.atan2", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_atan(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.atan", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_abs(in::Value; res::IR.Type, is_int_min_poison, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("is_int_min_poison", is_int_min_poison), ]
    
    create_operation(
        "llvm.intr.abs", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_annotation(integer::Value, annotation::Value, fileName::Value, line::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[integer, annotation, fileName, line, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.annotation", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_assume(cond::Value, op_bundle_operands::Vector{Value}; op_bundle_sizes, op_bundle_tags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[cond, op_bundle_operands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("op_bundle_sizes", op_bundle_sizes), ]
    !isnothing(op_bundle_tags) && push!(attributes, NamedAttribute("op_bundle_tags", op_bundle_tags))
    
    create_operation(
        "llvm.intr.assume", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_bitreverse(in::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.bitreverse", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_bswap(in::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.bswap", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_experimental_constrained_fpext(arg_0::Value; res::IR.Type, fpExceptionBehavior, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg_0, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("fpExceptionBehavior", fpExceptionBehavior), ]
    
    create_operation(
        "llvm.intr.experimental.constrained.fpext", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_experimental_constrained_fptrunc(arg_0::Value; res::IR.Type, roundingmode, fpExceptionBehavior, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg_0, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("roundingmode", roundingmode), NamedAttribute("fpExceptionBehavior", fpExceptionBehavior), ]
    
    create_operation(
        "llvm.intr.experimental.constrained.fptrunc", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_experimental_constrained_sitofp(arg_0::Value; res::IR.Type, roundingmode, fpExceptionBehavior, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg_0, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("roundingmode", roundingmode), NamedAttribute("fpExceptionBehavior", fpExceptionBehavior), ]
    
    create_operation(
        "llvm.intr.experimental.constrained.sitofp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_experimental_constrained_uitofp(arg_0::Value; res::IR.Type, roundingmode, fpExceptionBehavior, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg_0, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("roundingmode", roundingmode), NamedAttribute("fpExceptionBehavior", fpExceptionBehavior), ]
    
    create_operation(
        "llvm.intr.experimental.constrained.uitofp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_copysign(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.copysign", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_coro_align(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.coro.align", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_coro_begin(token::Value, mem::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[token, mem, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.coro.begin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_coro_end(handle::Value, unwind::Value, retvals::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[handle, unwind, retvals, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.coro.end", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_coro_free(id::Value, handle::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[id, handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.coro.free", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_coro_id(align::Value, promise::Value, coroaddr::Value, fnaddrs::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[align, promise, coroaddr, fnaddrs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.coro.id", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_coro_promise(handle::Value, align::Value, from::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[handle, align, from, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.coro.promise", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_coro_resume(handle::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.coro.resume", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_coro_save(handle::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.coro.save", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_coro_size(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.coro.size", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_coro_suspend(save::Value, final::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[save, final, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.coro.suspend", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_cos(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.cos", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_cosh(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.cosh", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_ctlz(in::Value; res=nothing::Union{Nothing, IR.Type}, is_zero_poison, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("is_zero_poison", is_zero_poison), ]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.ctlz", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_cttz(in::Value; res=nothing::Union{Nothing, IR.Type}, is_zero_poison, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("is_zero_poison", is_zero_poison), ]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.cttz", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_ctpop(in::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.ctpop", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_dbg_declare(addr::Value; varInfo, locationExpr=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varInfo", varInfo), ]
    !isnothing(locationExpr) && push!(attributes, NamedAttribute("locationExpr", locationExpr))
    
    create_operation(
        "llvm.intr.dbg.declare", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_dbg_label(; label, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("label", label), ]
    
    create_operation(
        "llvm.intr.dbg.label", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_dbg_value(value::Value; varInfo, locationExpr=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varInfo", varInfo), ]
    !isnothing(locationExpr) && push!(attributes, NamedAttribute("locationExpr", locationExpr))
    
    create_operation(
        "llvm.intr.dbg.value", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_debugtrap(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.debugtrap", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_eh_typeid_for(type_info::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[type_info, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.eh.typeid.for", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_exp2(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.exp2", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_exp10(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.exp10", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_exp(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.exp", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_expect(val::Value, expected::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val, expected, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.expect", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_expect_with_probability(val::Value, expected::Value; res=nothing::Union{Nothing, IR.Type}, prob, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val, expected, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("prob", prob), ]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.expect.with.probability", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_fabs(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.fabs", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_ceil(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.ceil", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_floor(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.floor", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_fma(a::Value, b::Value, c::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, c, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.fma", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_fmuladd(a::Value, b::Value, c::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, c, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.fmuladd", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_trunc(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.trunc", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_frexp(val::Value; res::IR.Type, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.frexp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_fshl(a::Value, b::Value, c::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, c, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.fshl", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_fshr(a::Value, b::Value, c::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, c, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.fshr", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_get_active_lane_mask(base::Value, n::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[base, n, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.get.active.lane.mask", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_invariant_end(start::Value, ptr::Value; size, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[start, ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("size", size), ]
    
    create_operation(
        "llvm.intr.invariant.end", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_invariant_start(ptr::Value; res=nothing::Union{Nothing, IR.Type}, size, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("size", size), ]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.invariant.start", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_is_constant(val::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.is.constant", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_is_fpclass(in::Value; res::IR.Type, bit, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("bit", bit), ]
    
    create_operation(
        "llvm.intr.is.fpclass", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_launder_invariant_group(ptr::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.launder.invariant.group", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_lifetime_end(ptr::Value; size, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("size", size), ]
    
    create_operation(
        "llvm.intr.lifetime.end", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_lifetime_start(ptr::Value; size, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("size", size), ]
    
    create_operation(
        "llvm.intr.lifetime.start", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_llrint(val::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.llrint", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_llround(val::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.llround", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_ldexp(val::Value, power::Value; res::IR.Type, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[val, power, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.ldexp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_log2(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.log2", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_log10(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.log10", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_log(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.log", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_lrint(val::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.lrint", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_lround(val::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.lround", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_masked_load(data::Value, mask::Value, pass_thru::Vector{Value}; res::IR.Type, alignment, nontemporal=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[data, mask, pass_thru..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("alignment", alignment), ]
    !isnothing(nontemporal) && push!(attributes, NamedAttribute("nontemporal", nontemporal))
    
    create_operation(
        "llvm.intr.masked.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_masked_store(value::Value, data::Value, mask::Value; alignment, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, data, mask, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("alignment", alignment), ]
    
    create_operation(
        "llvm.intr.masked.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_matrix_column_major_load(data::Value, stride::Value; res::IR.Type, isVolatile, rows, columns, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[data, stride, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("isVolatile", isVolatile), NamedAttribute("rows", rows), NamedAttribute("columns", columns), ]
    
    create_operation(
        "llvm.intr.matrix.column.major.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_matrix_column_major_store(matrix::Value, data::Value, stride::Value; isVolatile, rows, columns, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[matrix, data, stride, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("isVolatile", isVolatile), NamedAttribute("rows", rows), NamedAttribute("columns", columns), ]
    
    create_operation(
        "llvm.intr.matrix.column.major.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_matrix_multiply(lhs::Value, rhs::Value; res::IR.Type, lhs_rows, lhs_columns, rhs_columns, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("lhs_rows", lhs_rows), NamedAttribute("lhs_columns", lhs_columns), NamedAttribute("rhs_columns", rhs_columns), ]
    
    create_operation(
        "llvm.intr.matrix.multiply", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_matrix_transpose(matrix::Value; res::IR.Type, rows, columns, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[matrix, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("rows", rows), NamedAttribute("columns", columns), ]
    
    create_operation(
        "llvm.intr.matrix.transpose", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_maxnum(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.maxnum", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_maximum(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.maximum", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_memcpy_inline(dst::Value, src::Value; len, isVolatile, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dst, src, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("len", len), NamedAttribute("isVolatile", isVolatile), ]
    !isnothing(access_groups) && push!(attributes, NamedAttribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "llvm.intr.memcpy.inline", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_memcpy(dst::Value, src::Value, len::Value; isVolatile, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dst, src, len, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("isVolatile", isVolatile), ]
    !isnothing(access_groups) && push!(attributes, NamedAttribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "llvm.intr.memcpy", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_memmove(dst::Value, src::Value, len::Value; isVolatile, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dst, src, len, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("isVolatile", isVolatile), ]
    !isnothing(access_groups) && push!(attributes, NamedAttribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "llvm.intr.memmove", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_memset_inline(dst::Value, val::Value; len, isVolatile, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dst, val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("len", len), NamedAttribute("isVolatile", isVolatile), ]
    !isnothing(access_groups) && push!(attributes, NamedAttribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "llvm.intr.memset.inline", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_memset(dst::Value, val::Value, len::Value; isVolatile, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dst, val, len, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("isVolatile", isVolatile), ]
    !isnothing(access_groups) && push!(attributes, NamedAttribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    create_operation(
        "llvm.intr.memset", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_minnum(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.minnum", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_minimum(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.minimum", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_nearbyint(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.nearbyint", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_experimental_noalias_scope_decl(; scope, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("scope", scope), ]
    
    create_operation(
        "llvm.intr.experimental.noalias.scope.decl", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_powi(val::Value, power::Value; res::IR.Type, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[val, power, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.powi", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_pow(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.pow", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_prefetch(addr::Value; rw, hint, cache, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("rw", rw), NamedAttribute("hint", hint), NamedAttribute("cache", cache), ]
    
    create_operation(
        "llvm.intr.prefetch", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_ptr_annotation(ptr::Value, annotation::Value, fileName::Value, line::Value, attr::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, annotation, fileName, line, attr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.ptr.annotation", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_ptrmask(ptr::Value, mask::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, mask, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.ptrmask", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_rint(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.rint", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_roundeven(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.roundeven", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_round(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.round", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_sadd_sat(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.sadd.sat", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_sadd_with_overflow(operand_0::Value, operand_1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.sadd.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_smax(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.smax", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_smin(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.smin", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_smul_with_overflow(operand_0::Value, operand_1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.smul.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_ssa_copy(operand::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.ssa.copy", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_sshl_sat(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.sshl.sat", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_ssub_sat(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.ssub.sat", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_ssub_with_overflow(operand_0::Value, operand_1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.ssub.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_sin(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.sin", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_sinh(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.sinh", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_sqrt(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.sqrt", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_stackrestore(ptr::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.stackrestore", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_stacksave(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.stacksave", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_stepvector(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.stepvector", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_strip_invariant_group(ptr::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.strip.invariant.group", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_tan(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.tan", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_tanh(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.tanh", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_threadlocal_address(global_::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[global_, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.threadlocal.address", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_trap(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.trap", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_uadd_sat(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.uadd.sat", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_uadd_with_overflow(operand_0::Value, operand_1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.uadd.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_ubsantrap(; failureKind, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("failureKind", failureKind), ]
    
    create_operation(
        "llvm.intr.ubsantrap", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_umax(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.umax", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_umin(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.umin", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_umul_with_overflow(operand_0::Value, operand_1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.umul.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_ushl_sat(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.ushl.sat", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_usub_sat(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.usub.sat", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_usub_with_overflow(operand_0::Value, operand_1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.usub.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_ashr(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.ashr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_add(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.add", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_and(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.and", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_fadd(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.fadd", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_fdiv(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.fdiv", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_fmuladd(op1::Value, op2::Value, op3::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[op1, op2, op3, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.fmuladd", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_fmul(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.fmul", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_fneg(op::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[op, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.fneg", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_fpext(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.fpext", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_fptosi(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.fptosi", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_fptoui(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.fptoui", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_fptrunc(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.fptrunc", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_frem(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.frem", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_fsub(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.fsub", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_fma(op1::Value, op2::Value, op3::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[op1, op2, op3, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.fma", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_inttoptr(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.inttoptr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_lshr(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.lshr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_load(ptr::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_merge(cond::Value, true_val::Value, false_val::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[cond, true_val, false_val, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.merge", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_mul(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.mul", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_or(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.or", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_ptrtoint(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.ptrtoint", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_reduce_add(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.reduce.add", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_reduce_and(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.reduce.and", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_reduce_fadd(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.reduce.fadd", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_reduce_fmax(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.reduce.fmax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_reduce_fmin(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.reduce.fmin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_reduce_fmul(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.reduce.fmul", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_reduce_mul(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.reduce.mul", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_reduce_or(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.reduce.or", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_reduce_smax(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.reduce.smax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_reduce_smin(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.reduce.smin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_reduce_umax(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.reduce.umax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_reduce_umin(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.reduce.umin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_reduce_xor(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.reduce.xor", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_sdiv(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.sdiv", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_sext(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.sext", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_sitofp(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.sitofp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_smax(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.smax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_smin(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.smin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_srem(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.srem", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_select(cond::Value, true_val::Value, false_val::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[cond, true_val, false_val, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.select", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_shl(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.shl", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_store(val::Value, ptr::Value, mask::Value, evl::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val, ptr, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_experimental_vp_strided_load(ptr::Value, stride::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, stride, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.experimental.vp.strided.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_experimental_vp_strided_store(val::Value, ptr::Value, stride::Value, mask::Value, evl::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val, ptr, stride, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.experimental.vp.strided.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_sub(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.sub", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_trunc(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.trunc", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_udiv(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.udiv", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_uitofp(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.uitofp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_umax(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.umax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_umin(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.umin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_urem(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.urem", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_xor(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.xor", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vp_zext(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vp.zext", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vacopy(dest_list::Value, src_list::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dest_list, src_list, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vacopy", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vaend(arg_list::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[arg_list, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vaend", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vastart(arg_list::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[arg_list, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vastart", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_var_annotation(val::Value, annotation::Value, fileName::Value, line::Value, attr::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val, annotation, fileName, line, attr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.var.annotation", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_masked_compressstore(operand_0::Value, operand_1::Value, operand_2::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.masked.compressstore", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_masked_expandload(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.masked.expandload", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_masked_gather(ptrs::Value, mask::Value, pass_thru::Vector{Value}; res::IR.Type, alignment, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptrs, mask, pass_thru..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("alignment", alignment), ]
    
    create_operation(
        "llvm.intr.masked.gather", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_masked_scatter(value::Value, ptrs::Value, mask::Value; alignment, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, ptrs, mask, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("alignment", alignment), ]
    
    create_operation(
        "llvm.intr.masked.scatter", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_deinterleave2(vec::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[vec, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vector.deinterleave2", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_extract(srcvec::Value; res::IR.Type, pos, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcvec, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("pos", pos), ]
    
    create_operation(
        "llvm.intr.vector.extract", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_insert(dstvec::Value, srcvec::Value; res=nothing::Union{Nothing, IR.Type}, pos, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dstvec, srcvec, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("pos", pos), ]
    !isnothing(res) && push!(op_ty_results, res)
    
    create_operation(
        "llvm.intr.vector.insert", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end


function intr_vector_interleave2(vec1::Value, vec2::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[vec1, vec2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vector.interleave2", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_add(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vector.reduce.add", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_and(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vector.reduce.and", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_fadd(start_value::Value, input::Value; res::IR.Type, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[start_value, input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.vector.reduce.fadd", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_fmax(in::Value; res::IR.Type, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.vector.reduce.fmax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_fmaximum(in::Value; res::IR.Type, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.vector.reduce.fmaximum", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_fmin(in::Value; res::IR.Type, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.vector.reduce.fmin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_fminimum(in::Value; res::IR.Type, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.vector.reduce.fminimum", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_fmul(start_value::Value, input::Value; res::IR.Type, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[start_value, input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.intr.vector.reduce.fmul", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_mul(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vector.reduce.mul", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_or(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vector.reduce.or", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_smax(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vector.reduce.smax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_smin(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vector.reduce.smin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_umax(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vector.reduce.umax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_umin(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vector.reduce.umin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vector_reduce_xor(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vector.reduce.xor", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function intr_vscale(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.intr.vscale", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes
import ...API


"""
`v_iv`

Binary VCIX operation with an immediate second operand.

Correponds to:
|Mnemonic|funct6|vm|rs2|rs1|funct3|rd|Destination|Sources|
|--|--|--|--|--|--|--|--|--|
|sf.vc.v.iv|0010--|0|vs2|simm|011|vd|vector vd| simm[4:0]  vector vs2|
"""
function v_iv(vs2::Value, vl=nothing::Union{Nothing, Value}; res::IR.Type, opcode, imm, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[vs2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("opcode", opcode), NamedAttribute("imm", imm), ]
    !isnothing(vl) && push!(operands, vl)
    
    create_operation(
        "vcix.v.iv", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`v_sv`

Binary VCIX operation with an integer scalar, or floating pointer scalar or
vector second operand.

Correponds to:
|Mnemonic|funct6|vm|rs2|rs1|funct3|rd|Destination| Sources|
|--|--|--|--|--|--|--|--|--|--|
|sf.vc.v.vv|0010--|0|vs2|vs1|000|vd|vector vd|vector vs1, vector vs|
|sf.vc.v.xv|0010--|0|vs2|xs1|100|vd|vector vd|scalar xs1, vector vs2|
|sf.vc.v.fv|0010--|0|vs2|fs1|101|vd|vector vd|scalar fs1, vector vs2|
"""
function v_sv(vs2::Value, op::Value, vl=nothing::Union{Nothing, Value}; res::IR.Type, opcode, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[vs2, op, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("opcode", opcode), ]
    !isnothing(vl) && push!(operands, vl)
    
    create_operation(
        "vcix.v.sv", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes
import ...API


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
    
    create_operation(
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
    
    create_operation(
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
    
    create_operation(
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
    
    create_operation(
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
    
    create_operation(
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
    
    create_operation(
        "xevm.prefetch", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

end # llvm

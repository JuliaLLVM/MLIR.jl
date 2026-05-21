module llvm

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes

"""
`ashr`

"""
function ashr(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.ashr", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`add`

"""
function add(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.add", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`addrspacecast`

"""
function addrspacecast(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.addrspacecast", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mlir_addressof`

Creates an SSA value containing a pointer to a global variable or constant
defined by `llvm.mlir.global`. The global value can be defined after its
first referenced. If the global value is a constant, storing into it is not
allowed.

Examples:

```mlir
func @foo() {
  // Get the address of a global variable.
  %0 = llvm.mlir.addressof @const : !llvm.ptr<i32>

  // Use it as a regular pointer.
  %1 = llvm.load %0 : !llvm.ptr<i32>

  // Get the address of a function.
  %2 = llvm.mlir.addressof @foo : !llvm.ptr<func<void ()>>

  // The function address can be used for indirect calls.
  llvm.call %2() : () -> ()
}

// Define the global.
llvm.mlir.global @const(42 : i32) : i32
```
"""
function mlir_addressof(; res::IR.Type, global_name, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("global_name", global_name), ]
    
    IR.create_operation(
        "llvm.mlir.addressof", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`alloca`

"""
function alloca(arraySize::Value; res::IR.Type, alignment=nothing, elem_type=nothing, inalloca=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arraySize, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))
    !isnothing(elem_type) && push!(attributes, NamedAttribute("elem_type", elem_type))
    !isnothing(inalloca) && push!(attributes, NamedAttribute("inalloca", inalloca))
    
    IR.create_operation(
        "llvm.alloca", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`and`

"""
function and(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.and", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`cmpxchg`

"""
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
    
    IR.create_operation(
        "llvm.cmpxchg", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`atomicrmw`

"""
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
    
    IR.create_operation(
        "llvm.atomicrmw", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`bitcast`

"""
function bitcast(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.bitcast", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`br`

"""
function br(destOperands::Vector{Value}; loop_annotation=nothing, dest::Block, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[destOperands..., ]
    owned_regions = Region[]
    successors = Block[dest, ]
    attributes = NamedAttribute[]
    !isnothing(loop_annotation) && push!(attributes, NamedAttribute("loop_annotation", loop_annotation))
    
    IR.create_operation(
        "llvm.br", location;
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
function attribute `callee`. The trailing type list contains the optional
indirect callee type and the MLIR function type, which differs from the
LLVM function type that uses a explicit void type to model functions that do
not return a value.

Examples:

```mlir
// Direct call without arguments and with one result.
%0 = llvm.call @foo() : () -> (f32)

// Direct call with arguments and without a result.
llvm.call @bar(%0) : (f32) -> ()

// Indirect call with an argument and without a result.
llvm.call %1(%0) : !llvm.ptr, (f32) -> ()
```
"""
function call(operand_0::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, callee=nothing, fastmathFlags=nothing, branch_weights=nothing, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand_0..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(callee) && push!(attributes, NamedAttribute("callee", callee))
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    !isnothing(branch_weights) && push!(attributes, NamedAttribute("branch_weights", branch_weights))
    !isnothing(access_groups) && push!(attributes, NamedAttribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
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
    
    IR.create_operation(
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
    
    IR.create_operation(
        "llvm.comdat_selector", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`cond_br`

"""
function cond_br(condition::Value, trueDestOperands::Vector{Value}, falseDestOperands::Vector{Value}; branch_weights=nothing, loop_annotation=nothing, trueDest::Block, falseDest::Block, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[condition, trueDestOperands..., falseDestOperands..., ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([1, length(trueDestOperands), length(falseDestOperands), ]))
    !isnothing(branch_weights) && push!(attributes, NamedAttribute("branch_weights", branch_weights))
    !isnothing(loop_annotation) && push!(attributes, NamedAttribute("loop_annotation", loop_annotation))
    
    IR.create_operation(
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
operations. `llvm.mlir.constant` creates such values for scalars and
vectors. It has a mandatory `value` attribute, which may be an integer,
floating point attribute; dense or sparse attribute containing integers or
floats. The type of the attribute is one of the corresponding MLIR builtin
types. It may be omitted for `i64` and `f64` types that are implied. The
operation produces a new SSA value of the specified LLVM IR dialect type.
The type of that value _must_ correspond to the attribute type converted to
LLVM IR.

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
    
    IR.create_operation(
        "llvm.mlir.constant", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`extractelement`

"""
function extractelement(vector::Value, position::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vector, position, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.extractelement", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`extractvalue`

"""
function extractvalue(container::Value; res::IR.Type, position, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[container, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("position", position), ]
    
    IR.create_operation(
        "llvm.extractvalue", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`fadd`

"""
function fadd(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.fadd", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`fcmp`

"""
function fcmp(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, predicate, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("predicate", predicate), ]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.fcmp", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`fdiv`

"""
function fdiv(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.fdiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`fmul`

"""
function fmul(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.fmul", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`fneg`

"""
function fneg(operand::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.fneg", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`fpext`

"""
function fpext(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.fpext", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`fptosi`

"""
function fptosi(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.fptosi", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`fptoui`

"""
function fptoui(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.fptoui", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`fptrunc`

"""
function fptrunc(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.fptrunc", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`frem`

"""
function frem(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.frem", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`fsub`

"""
function fsub(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.fsub", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`fence`

"""
function fence(; ordering, syncscope=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("ordering", ordering), ]
    !isnothing(syncscope) && push!(attributes, NamedAttribute("syncscope", syncscope))
    
    IR.create_operation(
        "llvm.fence", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`freeze`

"""
function freeze(val::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
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

An optional \'inbounds\' attribute specifies the low-level pointer arithmetic
overflow behavior that LLVM uses after lowering the operation to LLVM IR.

Examples:

```mlir
// GEP with an SSA value offset
%0 = llvm.getelementptr %1[%2] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>

// GEP with a constant offset and the inbounds attribute set
%0 = llvm.getelementptr inbounds %1[3] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>

// GEP with constant offsets into a structure
%0 = llvm.getelementptr %1[0, 1]
   : (!llvm.ptr<struct(i32, f32)>) -> !llvm.ptr<f32>
```
"""
function getelementptr(base::Value, dynamicIndices::Vector{Value}; res::IR.Type, rawConstantIndices, elem_type=nothing, inbounds=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[base, dynamicIndices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("rawConstantIndices", rawConstantIndices), ]
    !isnothing(elem_type) && push!(attributes, NamedAttribute("elem_type", elem_type))
    !isnothing(inbounds) && push!(attributes, NamedAttribute("inbounds", inbounds))
    
    IR.create_operation(
        "llvm.getelementptr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mlir_global_ctors`

Specifies a list of constructor functions and priorities. The functions
referenced by this array will be called in ascending order of priority (i.e.
lowest first) when the module is loaded. The order of functions with the
same priority is not defined. This operation is translated to LLVM\'s
global_ctors global variable. The initializer functions are run at load
time. The `data` field present in LLVM\'s global_ctors variable is not
modeled here.

Examples:

```mlir
llvm.mlir.global_ctors {@ctor}

llvm.func @ctor() {
  ...
  llvm.return
}
```
"""
function mlir_global_ctors(; ctors, priorities, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("ctors", ctors), NamedAttribute("priorities", priorities), ]
    
    IR.create_operation(
        "llvm.mlir.global_ctors", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mlir_global_dtors`

Specifies a list of destructor functions and priorities. The functions
referenced by this array will be called in descending order of priority (i.e.
highest first) when the module is unloaded. The order of functions with the
same priority is not defined. This operation is translated to LLVM\'s
global_dtors global variable. The `data` field present in LLVM\'s
global_dtors variable is not modeled here.

Examples:

```mlir
llvm.func @dtor() {
  llvm.return
}
llvm.mlir.global_dtors {@dtor}
```
"""
function mlir_global_dtors(; dtors, priorities, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("dtors", dtors), NamedAttribute("priorities", priorities), ]
    
    IR.create_operation(
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
llvm.mlir.global constant @int_gep() : !llvm.ptr<i32> {
  %0 = llvm.mlir.addressof @g2 : !llvm.ptr<i32>
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.getelementptr %0[%1]
     : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
  // The initializer region must end with `llvm.return`.
  llvm.return %2 : !llvm.ptr<i32>
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
llvm.mlir.global constant @int_gep() : !llvm.ptr<i32> {
  %0 = llvm.mlir.addressof @g2 : !llvm.ptr<i32>
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.getelementptr %0[%1]
     : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
  llvm.return %2 : !llvm.ptr<i32>
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
function mlir_global(; global_type, constant=nothing, sym_name, linkage, dso_local=nothing, thread_local_=nothing, value=nothing, alignment=nothing, addr_space=nothing, unnamed_addr=nothing, section=nothing, comdat=nothing, visibility_=nothing, initializer::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[initializer, ]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("global_type", global_type), NamedAttribute("sym_name", sym_name), NamedAttribute("linkage", linkage), ]
    !isnothing(constant) && push!(attributes, NamedAttribute("constant", constant))
    !isnothing(dso_local) && push!(attributes, NamedAttribute("dso_local", dso_local))
    !isnothing(thread_local_) && push!(attributes, NamedAttribute("thread_local_", thread_local_))
    !isnothing(value) && push!(attributes, NamedAttribute("value", value))
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))
    !isnothing(addr_space) && push!(attributes, NamedAttribute("addr_space", addr_space))
    !isnothing(unnamed_addr) && push!(attributes, NamedAttribute("unnamed_addr", unnamed_addr))
    !isnothing(section) && push!(attributes, NamedAttribute("section", section))
    !isnothing(comdat) && push!(attributes, NamedAttribute("comdat", comdat))
    !isnothing(visibility_) && push!(attributes, NamedAttribute("visibility_", visibility_))
    
    IR.create_operation(
        "llvm.mlir.global", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`icmp`

"""
function icmp(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, predicate, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("predicate", predicate), ]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.icmp", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
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
"""
function inline_asm(operands::Vector{Value}; res=nothing::Union{Nothing, IR.Type}, asm_string, constraints, has_side_effects=nothing, is_align_stack=nothing, asm_dialect=nothing, operand_attrs=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("asm_string", asm_string), NamedAttribute("constraints", constraints), ]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(has_side_effects) && push!(attributes, NamedAttribute("has_side_effects", has_side_effects))
    !isnothing(is_align_stack) && push!(attributes, NamedAttribute("is_align_stack", is_align_stack))
    !isnothing(asm_dialect) && push!(attributes, NamedAttribute("asm_dialect", asm_dialect))
    !isnothing(operand_attrs) && push!(attributes, NamedAttribute("operand_attrs", operand_attrs))
    
    IR.create_operation(
        "llvm.inline_asm", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`insertelement`

"""
function insertelement(vector::Value, value::Value, position::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[vector, value, position, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.insertelement", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`insertvalue`

"""
function insertvalue(container::Value, value::Value; res=nothing::Union{Nothing, IR.Type}, position, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[container, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("position", position), ]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.insertvalue", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`inttoptr`

"""
function inttoptr(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.inttoptr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`invoke`

"""
function invoke(callee_operands::Vector{Value}, normalDestOperands::Vector{Value}, unwindDestOperands::Vector{Value}; result_0::Vector{IR.Type}, callee=nothing, branch_weights=nothing, normalDest::Block, unwindDest::Block, location=Location())
    op_ty_results = IR.Type[result_0..., ]
    operands = Value[callee_operands..., normalDestOperands..., unwindDestOperands..., ]
    owned_regions = Region[]
    successors = Block[normalDest, unwindDest, ]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(callee_operands), length(normalDestOperands), length(unwindDestOperands), ]))
    !isnothing(callee) && push!(attributes, NamedAttribute("callee", callee))
    !isnothing(branch_weights) && push!(attributes, NamedAttribute("branch_weights", branch_weights))
    
    IR.create_operation(
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
function func(; sym_name, function_type, linkage=nothing, dso_local=nothing, CConv=nothing, comdat=nothing, personality=nothing, garbageCollector=nothing, passthrough=nothing, arg_attrs=nothing, res_attrs=nothing, function_entry_count=nothing, memory=nothing, visibility_=nothing, arm_streaming=nothing, arm_locally_streaming=nothing, section=nothing, unnamed_addr=nothing, alignment=nothing, body::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), NamedAttribute("function_type", function_type), ]
    !isnothing(linkage) && push!(attributes, NamedAttribute("linkage", linkage))
    !isnothing(dso_local) && push!(attributes, NamedAttribute("dso_local", dso_local))
    !isnothing(CConv) && push!(attributes, NamedAttribute("CConv", CConv))
    !isnothing(comdat) && push!(attributes, NamedAttribute("comdat", comdat))
    !isnothing(personality) && push!(attributes, NamedAttribute("personality", personality))
    !isnothing(garbageCollector) && push!(attributes, NamedAttribute("garbageCollector", garbageCollector))
    !isnothing(passthrough) && push!(attributes, NamedAttribute("passthrough", passthrough))
    !isnothing(arg_attrs) && push!(attributes, NamedAttribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, NamedAttribute("res_attrs", res_attrs))
    !isnothing(function_entry_count) && push!(attributes, NamedAttribute("function_entry_count", function_entry_count))
    !isnothing(memory) && push!(attributes, NamedAttribute("memory", memory))
    !isnothing(visibility_) && push!(attributes, NamedAttribute("visibility_", visibility_))
    !isnothing(arm_streaming) && push!(attributes, NamedAttribute("arm_streaming", arm_streaming))
    !isnothing(arm_locally_streaming) && push!(attributes, NamedAttribute("arm_locally_streaming", arm_locally_streaming))
    !isnothing(section) && push!(attributes, NamedAttribute("section", section))
    !isnothing(unnamed_addr) && push!(attributes, NamedAttribute("unnamed_addr", unnamed_addr))
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))
    
    IR.create_operation(
        "llvm.func", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`lshr`

"""
function lshr(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.lshr", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`landingpad`

"""
function landingpad(operand_0::Vector{Value}; res::IR.Type, cleanup=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(cleanup) && push!(attributes, NamedAttribute("cleanup", cleanup))
    
    IR.create_operation(
        "llvm.landingpad", location;
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
function load(addr::Value; res::IR.Type, alignment=nothing, volatile_=nothing, nontemporal=nothing, ordering=nothing, syncscope=nothing, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, NamedAttribute("volatile_", volatile_))
    !isnothing(nontemporal) && push!(attributes, NamedAttribute("nontemporal", nontemporal))
    !isnothing(ordering) && push!(attributes, NamedAttribute("ordering", ordering))
    !isnothing(syncscope) && push!(attributes, NamedAttribute("syncscope", syncscope))
    !isnothing(access_groups) && push!(attributes, NamedAttribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
        "llvm.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`metadata`

llvm.metadata op defines one or more metadata nodes.

# Example
```mlir
llvm.metadata @metadata {
  llvm.access_group @group1
  llvm.access_group @group2
}
```
"""
function metadata(; sym_name, body::Region, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), ]
    
    IR.create_operation(
        "llvm.metadata", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mul`

"""
function mul(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.mul", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`mlir_null`

Unlike LLVM IR, MLIR does not have first-class null pointers. They must be
explicitly created as SSA values using `llvm.mlir.null`. This operation has
no operands or attributes, and returns a null value of a wrapped LLVM IR
pointer type.

Examples:

```mlir
// Null pointer to i8.
%0 = llvm.mlir.null : !llvm.ptr<i8>

// Null pointer to a function with signature void().
%1 = llvm.mlir.null : !llvm.ptr<func<void ()>>
```
"""
function mlir_null(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.mlir.null", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`or`

"""
function or(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
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
    
    IR.create_operation(
        "llvm.mlir.poison", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`ptrtoint`

"""
function ptrtoint(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.ptrtoint", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`resume`

"""
function resume(value::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.resume", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`return_`

"""
function return_(arg=nothing::Union{Nothing, Value}; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(arg) && push!(operands, arg)
    
    IR.create_operation(
        "llvm.return", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`sdiv`

"""
function sdiv(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.sdiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`sext`

"""
function sext(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.sext", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`sitofp`

"""
function sitofp(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.sitofp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`srem`

"""
function srem(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.srem", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`select`

"""
function select(condition::Value, trueValue::Value, falseValue::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[condition, trueValue, falseValue, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.select", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`shl`

"""
function shl(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.shl", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`shufflevector`

"""
function shufflevector(v1::Value, v2::Value; res::IR.Type, mask, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[v1, v2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mask", mask), ]
    
    IR.create_operation(
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
function store(value::Value, addr::Value; alignment=nothing, volatile_=nothing, nontemporal=nothing, ordering=nothing, syncscope=nothing, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, tbaa=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alignment) && push!(attributes, NamedAttribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, NamedAttribute("volatile_", volatile_))
    !isnothing(nontemporal) && push!(attributes, NamedAttribute("nontemporal", nontemporal))
    !isnothing(ordering) && push!(attributes, NamedAttribute("ordering", ordering))
    !isnothing(syncscope) && push!(attributes, NamedAttribute("syncscope", syncscope))
    !isnothing(access_groups) && push!(attributes, NamedAttribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, NamedAttribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, NamedAttribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, NamedAttribute("tbaa", tbaa))
    
    IR.create_operation(
        "llvm.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`sub`

"""
function sub(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.sub", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`switch`

"""
function switch(value::Value, defaultOperands::Vector{Value}, caseOperands::Vector{Value}; case_values=nothing, case_operand_segments, branch_weights=nothing, defaultDestination::Block, caseDestinations::Vector{Block}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, defaultOperands..., caseOperands..., ]
    owned_regions = Region[]
    successors = Block[defaultDestination, caseDestinations..., ]
    attributes = NamedAttribute[NamedAttribute("case_operand_segments", case_operand_segments), ]
    push!(attributes, operandsegmentsizes([1, length(defaultOperands), length(caseOperands), ]))
    !isnothing(case_values) && push!(attributes, NamedAttribute("case_values", case_values))
    !isnothing(branch_weights) && push!(attributes, NamedAttribute("branch_weights", branch_weights))
    
    IR.create_operation(
        "llvm.switch", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`trunc`

"""
function trunc(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.trunc", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`udiv`

"""
function udiv(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.udiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`uitofp`

"""
function uitofp(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.uitofp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`urem`

"""
function urem(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
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
    
    IR.create_operation(
        "llvm.mlir.undef", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`unreachable`

"""
function unreachable(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.unreachable", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`xor`

"""
function xor(lhs::Value, rhs::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.xor", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`zext`

"""
function zext(arg::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.zext", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end
import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes

"""
`intr_abs`

"""
function intr_abs(in::Value; res::IR.Type, is_int_min_poison, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("is_int_min_poison", is_int_min_poison), ]
    
    IR.create_operation(
        "llvm.intr.abs", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_annotation`

"""
function intr_annotation(integer::Value, annotation::Value, fileName::Value, line::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[integer, annotation, fileName, line, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.annotation", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_assume`

"""
function intr_assume(cond::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[cond, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.assume", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_bitreverse`

"""
function intr_bitreverse(in::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.bitreverse", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_bswap`

"""
function intr_bswap(in::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.bswap", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`call_intrinsic`

Call the specified llvm intrinsic. If the intrinsic is overloaded, use
the MLIR function type of this op to determine which intrinsic to call.
"""
function call_intrinsic(args::Vector{Value}; results::Vector{IR.Type}, intrin, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[results..., ]
    operands = Value[args..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("intrin", intrin), ]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.call_intrinsic", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_copysign`

"""
function intr_copysign(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.copysign", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_coro_align`

"""
function intr_coro_align(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.align", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_coro_begin`

"""
function intr_coro_begin(token::Value, mem::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[token, mem, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.begin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_coro_end`

"""
function intr_coro_end(handle::Value, unwind::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[handle, unwind, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.end", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_coro_free`

"""
function intr_coro_free(id::Value, handle::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[id, handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.free", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_coro_id`

"""
function intr_coro_id(align::Value, promise::Value, coroaddr::Value, fnaddrs::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[align, promise, coroaddr, fnaddrs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.id", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_coro_resume`

"""
function intr_coro_resume(handle::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.resume", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_coro_save`

"""
function intr_coro_save(handle::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[handle, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.save", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_coro_size`

"""
function intr_coro_size(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.size", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_coro_suspend`

"""
function intr_coro_suspend(save::Value, final::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[save, final, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.suspend", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_cos`

"""
function intr_cos(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.cos", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_ctlz`

"""
function intr_ctlz(in::Value; res::IR.Type, is_zero_poison, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("is_zero_poison", is_zero_poison), ]
    
    IR.create_operation(
        "llvm.intr.ctlz", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_cttz`

"""
function intr_cttz(in::Value; res::IR.Type, is_zero_poison, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("is_zero_poison", is_zero_poison), ]
    
    IR.create_operation(
        "llvm.intr.cttz", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_ctpop`

"""
function intr_ctpop(in::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.ctpop", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_dbg_declare`

"""
function intr_dbg_declare(addr::Value; varInfo, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varInfo", varInfo), ]
    
    IR.create_operation(
        "llvm.intr.dbg.declare", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_dbg_label`

"""
function intr_dbg_label(; label, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("label", label), ]
    
    IR.create_operation(
        "llvm.intr.dbg.label", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_dbg_value`

"""
function intr_dbg_value(value::Value; varInfo, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("varInfo", varInfo), ]
    
    IR.create_operation(
        "llvm.intr.dbg.value", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_debugtrap`

"""
function intr_debugtrap(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.debugtrap", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_eh_typeid_for`

"""
function intr_eh_typeid_for(type_info::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[type_info, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.eh.typeid.for", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_exp2`

"""
function intr_exp2(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.exp2", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_exp`

"""
function intr_exp(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.exp", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_expect`

"""
function intr_expect(val::Value, expected::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val, expected, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.expect", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_expect_with_probability`

"""
function intr_expect_with_probability(val::Value, expected::Value; res=nothing::Union{Nothing, IR.Type}, prob, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val, expected, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("prob", prob), ]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.expect.with.probability", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_fabs`

"""
function intr_fabs(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.fabs", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_ceil`

"""
function intr_ceil(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.ceil", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_floor`

"""
function intr_floor(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.floor", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_fma`

"""
function intr_fma(a::Value, b::Value, c::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, c, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.fma", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_fmuladd`

"""
function intr_fmuladd(a::Value, b::Value, c::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, c, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.fmuladd", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_trunc`

"""
function intr_trunc(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.trunc", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_fshl`

"""
function intr_fshl(a::Value, b::Value, c::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, c, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.fshl", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_fshr`

"""
function intr_fshr(a::Value, b::Value, c::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, c, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.fshr", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_get_active_lane_mask`

"""
function intr_get_active_lane_mask(base::Value, n::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[base, n, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.get.active.lane.mask", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_is_constant`

"""
function intr_is_constant(val::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.is.constant", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_is_fpclass`

"""
function intr_is_fpclass(in::Value; res::IR.Type, bit, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("bit", bit), ]
    
    IR.create_operation(
        "llvm.intr.is.fpclass", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_lifetime_end`

"""
function intr_lifetime_end(ptr::Value; size, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("size", size), ]
    
    IR.create_operation(
        "llvm.intr.lifetime.end", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_lifetime_start`

"""
function intr_lifetime_start(ptr::Value; size, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("size", size), ]
    
    IR.create_operation(
        "llvm.intr.lifetime.start", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_llrint`

"""
function intr_llrint(val::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.llrint", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_llround`

"""
function intr_llround(val::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.llround", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_log2`

"""
function intr_log2(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.log2", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_log10`

"""
function intr_log10(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.log10", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_log`

"""
function intr_log(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.log", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_lrint`

"""
function intr_lrint(val::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.lrint", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_lround`

"""
function intr_lround(val::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.lround", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_masked_load`

"""
function intr_masked_load(data::Value, mask::Value, pass_thru::Vector{Value}; res::IR.Type, alignment, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[data, mask, pass_thru..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("alignment", alignment), ]
    
    IR.create_operation(
        "llvm.intr.masked.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_masked_store`

"""
function intr_masked_store(value::Value, data::Value, mask::Value; alignment, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, data, mask, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("alignment", alignment), ]
    
    IR.create_operation(
        "llvm.intr.masked.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_matrix_column_major_load`

"""
function intr_matrix_column_major_load(data::Value, stride::Value; res::IR.Type, isVolatile, rows, columns, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[data, stride, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("isVolatile", isVolatile), NamedAttribute("rows", rows), NamedAttribute("columns", columns), ]
    
    IR.create_operation(
        "llvm.intr.matrix.column.major.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_matrix_column_major_store`

"""
function intr_matrix_column_major_store(matrix::Value, data::Value, stride::Value; isVolatile, rows, columns, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[matrix, data, stride, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("isVolatile", isVolatile), NamedAttribute("rows", rows), NamedAttribute("columns", columns), ]
    
    IR.create_operation(
        "llvm.intr.matrix.column.major.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_matrix_multiply`

"""
function intr_matrix_multiply(lhs::Value, rhs::Value; res::IR.Type, lhs_rows, lhs_columns, rhs_columns, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("lhs_rows", lhs_rows), NamedAttribute("lhs_columns", lhs_columns), NamedAttribute("rhs_columns", rhs_columns), ]
    
    IR.create_operation(
        "llvm.intr.matrix.multiply", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_matrix_transpose`

"""
function intr_matrix_transpose(matrix::Value; res::IR.Type, rows, columns, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[matrix, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("rows", rows), NamedAttribute("columns", columns), ]
    
    IR.create_operation(
        "llvm.intr.matrix.transpose", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_maxnum`

"""
function intr_maxnum(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.maxnum", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_maximum`

"""
function intr_maximum(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.maximum", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_memcpy_inline`

"""
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
    
    IR.create_operation(
        "llvm.intr.memcpy.inline", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_memcpy`

"""
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
    
    IR.create_operation(
        "llvm.intr.memcpy", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_memmove`

"""
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
    
    IR.create_operation(
        "llvm.intr.memmove", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_memset`

"""
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
    
    IR.create_operation(
        "llvm.intr.memset", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_minnum`

"""
function intr_minnum(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.minnum", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_minimum`

"""
function intr_minimum(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.minimum", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_nearbyint`

"""
function intr_nearbyint(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.nearbyint", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_experimental_noalias_scope_decl`

"""
function intr_experimental_noalias_scope_decl(; scope, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("scope", scope), ]
    
    IR.create_operation(
        "llvm.intr.experimental.noalias.scope.decl", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_powi`

"""
function intr_powi(val::Value, power::Value; res::IR.Type, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[val, power, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.powi", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_pow`

"""
function intr_pow(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.pow", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_prefetch`

"""
function intr_prefetch(addr::Value; rw, hint, cache, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("rw", rw), NamedAttribute("hint", hint), NamedAttribute("cache", cache), ]
    
    IR.create_operation(
        "llvm.intr.prefetch", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_ptr_annotation`

"""
function intr_ptr_annotation(ptr::Value, annotation::Value, fileName::Value, line::Value, attr::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, annotation, fileName, line, attr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.ptr.annotation", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_rint`

"""
function intr_rint(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.rint", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_roundeven`

"""
function intr_roundeven(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.roundeven", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_round`

"""
function intr_round(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.round", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_sadd_sat`

"""
function intr_sadd_sat(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.sadd.sat", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_sadd_with_overflow`

"""
function intr_sadd_with_overflow(operand_0::Value, operand_1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.sadd.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_smax`

"""
function intr_smax(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.smax", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_smin`

"""
function intr_smin(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.smin", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_smul_with_overflow`

"""
function intr_smul_with_overflow(operand_0::Value, operand_1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.smul.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_ssa_copy`

"""
function intr_ssa_copy(operand::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.ssa.copy", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_sshl_sat`

"""
function intr_sshl_sat(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.sshl.sat", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_ssub_sat`

"""
function intr_ssub_sat(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.ssub.sat", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_ssub_with_overflow`

"""
function intr_ssub_with_overflow(operand_0::Value, operand_1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.ssub.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_sin`

"""
function intr_sin(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.sin", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_sqrt`

"""
function intr_sqrt(in::Value; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.sqrt", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_stackrestore`

"""
function intr_stackrestore(ptr::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[ptr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.stackrestore", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_stacksave`

"""
function intr_stacksave(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.stacksave", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_experimental_stepvector`

"""
function intr_experimental_stepvector(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.experimental.stepvector", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_threadlocal_address`

"""
function intr_threadlocal_address(global_::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[global_, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.threadlocal.address", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_trap`

"""
function intr_trap(; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.trap", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_uadd_sat`

"""
function intr_uadd_sat(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.uadd.sat", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_uadd_with_overflow`

"""
function intr_uadd_with_overflow(operand_0::Value, operand_1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.uadd.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_ubsantrap`

"""
function intr_ubsantrap(; failureKind, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("failureKind", failureKind), ]
    
    IR.create_operation(
        "llvm.intr.ubsantrap", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_umax`

"""
function intr_umax(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.umax", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_umin`

"""
function intr_umin(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.umin", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_umul_with_overflow`

"""
function intr_umul_with_overflow(operand_0::Value, operand_1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.umul.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_ushl_sat`

"""
function intr_ushl_sat(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.ushl.sat", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_usub_sat`

"""
function intr_usub_sat(a::Value, b::Value; res=nothing::Union{Nothing, IR.Type}, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[a, b, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.usub.sat", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_usub_with_overflow`

"""
function intr_usub_with_overflow(operand_0::Value, operand_1::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.usub.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_ashr`

"""
function intr_vp_ashr(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.ashr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_add`

"""
function intr_vp_add(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.add", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_and`

"""
function intr_vp_and(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.and", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_fadd`

"""
function intr_vp_fadd(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.fadd", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_fdiv`

"""
function intr_vp_fdiv(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.fdiv", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_fmuladd`

"""
function intr_vp_fmuladd(op1::Value, op2::Value, op3::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[op1, op2, op3, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.fmuladd", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_fmul`

"""
function intr_vp_fmul(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.fmul", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_fneg`

"""
function intr_vp_fneg(op::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[op, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.fneg", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_fpext`

"""
function intr_vp_fpext(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.fpext", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_fptosi`

"""
function intr_vp_fptosi(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.fptosi", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_fptoui`

"""
function intr_vp_fptoui(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.fptoui", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_fptrunc`

"""
function intr_vp_fptrunc(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.fptrunc", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_frem`

"""
function intr_vp_frem(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.frem", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_fsub`

"""
function intr_vp_fsub(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.fsub", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_fma`

"""
function intr_vp_fma(op1::Value, op2::Value, op3::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[op1, op2, op3, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.fma", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_inttoptr`

"""
function intr_vp_inttoptr(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.inttoptr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_lshr`

"""
function intr_vp_lshr(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.lshr", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_load`

"""
function intr_vp_load(ptr::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_merge`

"""
function intr_vp_merge(cond::Value, true_val::Value, false_val::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[cond, true_val, false_val, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.merge", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_mul`

"""
function intr_vp_mul(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.mul", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_or`

"""
function intr_vp_or(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.or", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_ptrtoint`

"""
function intr_vp_ptrtoint(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.ptrtoint", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_reduce_add`

"""
function intr_vp_reduce_add(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.reduce.add", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_reduce_and`

"""
function intr_vp_reduce_and(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.reduce.and", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_reduce_fadd`

"""
function intr_vp_reduce_fadd(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.reduce.fadd", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_reduce_fmax`

"""
function intr_vp_reduce_fmax(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.reduce.fmax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_reduce_fmin`

"""
function intr_vp_reduce_fmin(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.reduce.fmin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_reduce_fmul`

"""
function intr_vp_reduce_fmul(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.reduce.fmul", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_reduce_mul`

"""
function intr_vp_reduce_mul(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.reduce.mul", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_reduce_or`

"""
function intr_vp_reduce_or(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.reduce.or", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_reduce_smax`

"""
function intr_vp_reduce_smax(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.reduce.smax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_reduce_smin`

"""
function intr_vp_reduce_smin(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.reduce.smin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_reduce_umax`

"""
function intr_vp_reduce_umax(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.reduce.umax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_reduce_umin`

"""
function intr_vp_reduce_umin(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.reduce.umin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_reduce_xor`

"""
function intr_vp_reduce_xor(satrt_value::Value, val::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[satrt_value, val, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.reduce.xor", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_sdiv`

"""
function intr_vp_sdiv(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.sdiv", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_sext`

"""
function intr_vp_sext(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.sext", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_sitofp`

"""
function intr_vp_sitofp(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.sitofp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_srem`

"""
function intr_vp_srem(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.srem", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_select`

"""
function intr_vp_select(cond::Value, true_val::Value, false_val::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[cond, true_val, false_val, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.select", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_shl`

"""
function intr_vp_shl(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.shl", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_store`

"""
function intr_vp_store(val::Value, ptr::Value, mask::Value, evl::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val, ptr, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_experimental_vp_strided_load`

"""
function intr_experimental_vp_strided_load(ptr::Value, stride::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptr, stride, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.experimental.vp.strided.load", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_experimental_vp_strided_store`

"""
function intr_experimental_vp_strided_store(val::Value, ptr::Value, stride::Value, mask::Value, evl::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val, ptr, stride, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.experimental.vp.strided.store", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_sub`

"""
function intr_vp_sub(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.sub", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_trunc`

"""
function intr_vp_trunc(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.trunc", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_udiv`

"""
function intr_vp_udiv(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.udiv", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_uitofp`

"""
function intr_vp_uitofp(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.uitofp", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_urem`

"""
function intr_vp_urem(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.urem", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_xor`

"""
function intr_vp_xor(lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[lhs, rhs, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.xor", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vp_zext`

"""
function intr_vp_zext(src::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[src, mask, evl, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vp.zext", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vacopy`

"""
function intr_vacopy(dest_list::Value, src_list::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dest_list, src_list, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vacopy", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vaend`

"""
function intr_vaend(arg_list::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[arg_list, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vaend", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vastart`

"""
function intr_vastart(arg_list::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[arg_list, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vastart", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_var_annotation`

"""
function intr_var_annotation(val::Value, annotation::Value, fileName::Value, line::Value, attr::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[val, annotation, fileName, line, attr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.var.annotation", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_masked_compressstore`

"""
function intr_masked_compressstore(operand_0::Value, operand_1::Value, operand_2::Value; location=Location())
    op_ty_results = IR.Type[]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.masked.compressstore", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_masked_expandload`

"""
function intr_masked_expandload(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.masked.expandload", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_masked_gather`

"""
function intr_masked_gather(ptrs::Value, mask::Value, pass_thru::Vector{Value}; res::IR.Type, alignment, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[ptrs, mask, pass_thru..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("alignment", alignment), ]
    
    IR.create_operation(
        "llvm.intr.masked.gather", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_masked_scatter`

"""
function intr_masked_scatter(value::Value, ptrs::Value, mask::Value; alignment, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[value, ptrs, mask, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("alignment", alignment), ]
    
    IR.create_operation(
        "llvm.intr.masked.scatter", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_extract`

"""
function intr_vector_extract(srcvec::Value; res::IR.Type, pos, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[srcvec, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("pos", pos), ]
    
    IR.create_operation(
        "llvm.intr.vector.extract", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_insert`

"""
function intr_vector_insert(srcvec::Value, dstvec::Value; res=nothing::Union{Nothing, IR.Type}, pos, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[srcvec, dstvec, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("pos", pos), ]
    !isnothing(res) && push!(op_ty_results, res)
    
    IR.create_operation(
        "llvm.intr.vector.insert", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`intr_vector_reduce_add`

"""
function intr_vector_reduce_add(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.add", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_and`

"""
function intr_vector_reduce_and(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.and", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_fadd`

"""
function intr_vector_reduce_fadd(start_value::Value, input::Value; res::IR.Type, reassoc=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[start_value, input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(reassoc) && push!(attributes, NamedAttribute("reassoc", reassoc))
    
    IR.create_operation(
        "llvm.intr.vector.reduce.fadd", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_fmax`

"""
function intr_vector_reduce_fmax(in::Value; res::IR.Type, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.vector.reduce.fmax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_fmaximum`

"""
function intr_vector_reduce_fmaximum(in::Value; res::IR.Type, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.vector.reduce.fmaximum", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_fmin`

"""
function intr_vector_reduce_fmin(in::Value; res::IR.Type, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.vector.reduce.fmin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_fminimum`

"""
function intr_vector_reduce_fminimum(in::Value; res::IR.Type, fastmathFlags=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(fastmathFlags) && push!(attributes, NamedAttribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.intr.vector.reduce.fminimum", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_fmul`

"""
function intr_vector_reduce_fmul(start_value::Value, input::Value; res::IR.Type, reassoc=nothing, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[start_value, input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(reassoc) && push!(attributes, NamedAttribute("reassoc", reassoc))
    
    IR.create_operation(
        "llvm.intr.vector.reduce.fmul", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_mul`

"""
function intr_vector_reduce_mul(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.mul", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_or`

"""
function intr_vector_reduce_or(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.or", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_smax`

"""
function intr_vector_reduce_smax(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.smax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_smin`

"""
function intr_vector_reduce_smin(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.smin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_umax`

"""
function intr_vector_reduce_umax(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.umax", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_umin`

"""
function intr_vector_reduce_umin(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.umin", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_xor`

"""
function intr_vector_reduce_xor(in::Value; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[in, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.xor", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`intr_vscale`

"""
function intr_vscale(; res::IR.Type, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vscale", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end
end # llvm

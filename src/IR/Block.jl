mutable struct Block
    block::API.MlirBlock
    @atomic owned::Bool

    Block(block::API.MlirBlock, owned::Bool=true) = begin
        @assert !mlirBlockIsNull(block) "cannot create Block with null MlirBlock"
        finalizer(new(block, owned)) do block
            if block.owned
                API.mlirBlockDestroy(block.block)
            end
        end
    end
end

Block() = Block(Type[], Location[])
function Block(args::Vector{Type}, locs::Vector{Location})
    @assert length(args) == length(locs) "there should be one args for each locs (got $(length(args)) & $(length(locs)))"
    Block(API.mlirBlockCreate(length(args), args, locs))
end

Base.:(==)(a::Block, b::Block) = API.mlirBlockEqual(a, b)
Base.cconvert(::Core.Type{API.MlirBlock}, block::Block) = block
Base.unsafe_convert(::Core.Type{API.MlirBlock}, block::Block) = block.block

parent_op(block::Block) = Operation(API.mlirBlockGetParentOperation(block))
parent_region(block::Block) = Region(API.mlirBlockGetParentRegion(block))
Base.parent(block::Block) = parent_region(block)

function next_in_region(block::Block)
    block = API.mlirBlockGetNextInRegion(block)
    mlirBlockIsNull(block) && return nothing

    Block(block)
end

num_args(block::Block) = API.mlirBlockGetNumArguments(block)
function get_argument(block::Block, i)
    i ∉ 1:num_arguments(block) && throw(BoundsError(block, i))
    Value(API.mlirBlockGetArgument(block, i - 1))
end
push_argument!(block::Block, type, loc) = Value(API.mlirBlockAddArgument(block, type, loc))

first_op(block::Block) = Operation(API.mlirBlockGetFirstOperation(block))
function terminator(block::Block)
    op = API.mlirBlockGetTerminator(block)
    mlirOperationIsNull(op) && return nothing
    Operation(op)
end

function Base.push!(block::Block, op::Operation)
    API.mlirBlockAppendOwnedOperation(block, lose_ownership!(op))
    op
end

function Base.insert!(block::Block, pos, op::Operation)
    API.mlirBlockInsertOwnedOperation(block, pos - 1, lose_ownership!(op))
    op
end

function Base.pushfirst!(block::Block, op::Operation)
    insert!(block, 1, op)
    op
end

function insert_after!(block::Block, reference::Operation, op::Operation)
    API.mlirBlockInsertOwnedOperationAfter(block, reference, lose_ownership!(op))
    op
end

function insert_before!(block::Block, reference::Operation, op::Operation)
    API.mlirBlockInsertOwnedOperationBefore(block, reference, lose_ownership!(op))
    op
end

function lose_ownership!(block::Block)
    @assert block.owned
    API.mlirBlockDetach(block)
    @atomic block.owned = false
    block
end

function Base.show(io::IO, block::Block)
    c_print_callback = @cfunction(print_callback, Cvoid, (MlirStringRef, Any))
    ref = Ref(io)
    API.mlirBlockPrint(block, c_print_callback, ref)
end

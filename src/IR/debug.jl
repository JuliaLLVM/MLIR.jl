# inspired by LLVM.jl's memory tracking utilities, but with an extra `mark_dispose` function to track when objects are given another object as owner
using Preferences

const MEMCHECK_ENABLED = parse(Bool, @load_preference("memcheck", "false"))

# object => (alloc_bt, dispose_bt)
const tracked_objects = Dict{Any,Any}()

# the most basic check is asserting that we don't use a null pointer
@inline function refcheck(::Type, ref::Ptr)
    ref==C_NULL && throw(UndefRefError())
end

function mark_alloc(obj; allow_overwrite::Bool=false)
    @static if MEMCHECK_ENABLED
        io = Core.stdout
        new_alloc_bt = backtrace()[2:end]

        if haskey(tracked_objects, obj) && !allow_overwrite
            old_alloc_bt, dispose_bt = tracked_objects[obj]
            if dispose_bt == nothing
                print("\nWARNING: An instance of $(typeof(obj)) was not properly disposed of, and a new allocation will overwrite it.")
                print("\nThe original allocation was at:")
                Base.show_backtrace(io, old_alloc_bt)
                print("\nThe new allocation is at:")
                Base.show_backtrace(io, new_alloc_bt)
                println(io)
            end
        end

        tracked_objects[obj] = (new_alloc_bt, nothing)
    end
    return obj
end

function mark_use(obj::Any)
    @static if MEMCHECK_ENABLED
        io = Core.stdout

        if !haskey(tracked_objects, obj)
            # we have to ignore unknown objects, as they may originate externally.
            # for example, a Julia-created Type we call `context` on.
            return obj
        end

        alloc_bt, dispose_bt = tracked_objects[obj]
        if dispose_bt !== nothing
            print("\nWARNING: An instance of $(typeof(obj)) is being used after it was disposed of.")
            print("\nThe object was allocated at:")
            Base.show_backtrace(io, alloc_bt)
            print("\nThe object was disposed of at:")
            Base.show_backtrace(io, dispose_bt)
            print("\nThe object is being used at:")
            Base.show_backtrace(io, backtrace()[2:end])
            println(io)
        end
    end
    return obj
end

mark_dispose(obj) = mark_dispose(Returns(nothing), obj)

function mark_dispose(f, obj)
    data = @static if MEMCHECK_ENABLED
        io = Core.stdout
        new_dispose_bt = backtrace()[2:end]

        if !haskey(tracked_objects, obj)
            print(io, "\nWARNING: An unknown instance of $(typeof(obj)) is being disposed of.")
            Base.show_backtrace(io, new_dispose_bt)
            nothing
        else
            alloc_bt, old_dispose_bt = tracked_objects[obj]
            if old_dispose_bt !== nothing
                print("\nWARNING: An instance of $(typeof(obj)) is being disposed of twice.")
                print("\nThe object was allocated at:")
                Base.show_backtrace(io, alloc_bt)
                print("\nThe object was already disposed of at:")
                Base.show_backtrace(io, old_dispose_bt)
                print("\nThe object is being disposed of again at:")
                Base.show_backtrace(io, new_dispose_bt)
                println(io)
            end

            (alloc_bt, new_dispose_bt)
        end
    end
    ret = f(obj)
    @static if MEMCHECK_ENABLED
        if data !== nothing
            tracked_objects[obj] = data
        end
    end
    return
end

# we could potentially track ownership here
mark_donate(new_owner, obj) = mark_dispose(obj)

# MLIR.API types
for AT in [
    :MlirDialect,
    :MlirDialectHandle,
    :MlirDialectRegistry,
    :MlirContext,
    :MlirLocation,
    :MlirType,
    :MlirTypeID,
    :MlirTypeIDAllocator,
    :MlirModule,
    :MlirOperation,
    :MlirOpOperand,
    :MlirBlock,
    :MlirRegion,
    :MlirValue,
    # :MlirLogicalResult,
    :MlirAffineExpr,
    :MlirAffineMap,
    # :MlirAttribute,
    # :MlirNamedAttribute,
    :MlirIntegerSet,
    :MlirIdentifier,
    :MlirSymbolTable,
    :MlirExecutionEngine,
    :MlirPassManager,
    :MlirOpPassManager,
]
    @eval refcheck(T::Core.Type, ref::API.$AT) = refcheck(T, ref.ptr)
end

function report_leaks(code=0)
    # if we errored, we can't trust the memory state
    if code != 0
        return
    end

    @static if MEMCHECK_ENABLED
        io = Core.stdout
        for (obj, (alloc_bt, dispose_bt)) in tracked_objects
            if dispose_bt === nothing
                print(io, "\nWARNING: An instance of $(typeof(obj)) was not properly disposed of.")
                print("\nThe object was allocated at:")
                Base.show_backtrace(io, alloc_bt)
                println(io)
            end
        end
    end
end

# macro that adds an inner constructor to a type definition,
# calling `refcheck` on the ref field argument
macro checked(typedef)
    # decode structure definition
    if Meta.isexpr(typedef, :struct)
        structure = typedef.args[2]
        body = typedef.args[3]
    else
        error("argument is not a structure definition")
    end
    if isa(structure, Symbol)
        # basic type definition
        typename = structure
    elseif Meta.isexpr(structure, :<:)
        # typename <: parentname
        all(e->isa(e,Symbol), structure.args) ||
            error("typedef should consist of plain types, ie. not parametric ones")
        typename = structure.args[1]
    else
        error("malformed type definition: cannot decode type name")
    end

    # decode fields
    field_names = Symbol[]
    field_defs = Union{Symbol,Expr}[]
    for arg in body.args
        if isa(arg, LineNumberNode)
            continue
        elseif isa(arg, Symbol)
            push!(field_names, arg)
            push!(field_defs, arg)
        elseif Meta.isexpr(arg, :(::))
            push!(field_names, arg.args[1])
            push!(field_defs, arg)
        end
    end
    :ref in field_names || error("structure definition should contain 'ref' field")

    # insert checked constructor
    push!(body.args, :(
        $typename($(field_defs...)) = ($refcheck($typename, ref); new($(field_names...)))
    ))

    return esc(typedef)
end

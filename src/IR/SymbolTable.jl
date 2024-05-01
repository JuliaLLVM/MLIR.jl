struct SymbolTable
    st::API.MlirSymbolTable

    function SymbolTable(st)
        @assert !mlirIsNull(st) "cannot create SymbolTable with null MlirSymbolTable"
        finalizer(API.Dispatcher.mlirSymbolTableDestroy, new(st))
    end
end

"""
    mlirSymbolTableCreate(operation)

Creates a symbol table for the given operation. If the operation does not have the SymbolTable trait, returns a null symbol table.
"""
SymbolTable(op::Operation) = SymbolTable(API.Dispatcher.mlirSymbolTableCreate(op))

Base.convert(::Core.Type{API.MlirSymbolTable}, st::SymbolTable) = st.st

# TODO mlirSymbolTableGetSymbolAttributeName
# TODO mlirSymbolTableGetVisibilityAttributeName

"""
    lookup(symboltable, name)

Looks up a symbol with the given name in the given symbol table and returns the operation that corresponds to the symbol.
If the symbol cannot be found, returns a null operation.
"""
lookup(st::SymbolTable, name::AbstractString) = Operation(API.Dispatcher.mlirSymbolTableLookup(st, name))
Base.getindex(st::SymbolTable, name::AbstractString) = lookup(st, name)

"""
    push!(symboltable, operation)

Inserts the given operation into the given symbol table. The operation must have the symbol trait.
If the symbol table already has a symbol with the same name, renames the symbol being inserted to ensure name uniqueness.
Note that this does not move the operation itself into the block of the symbol table operation, this should be done separately.
Returns the name of the symbol after insertion.
"""
push!(st::SymbolTable, op::Operation) = Attribute(API.Dispatcher.mlirSymbolTableInsert(st, op))

"""
    delete!(symboltable, operation)

Removes the given operation from the symbol table and erases it.
"""
delete!(st::SymbolTable, op::Operation) = API.Dispatcher.mlirSymbolTableErase(st, op)

# TODO mlirSymbolTableReplaceAllSymbolUses
# TODO mlirSymbolTableWalkSymbolTables

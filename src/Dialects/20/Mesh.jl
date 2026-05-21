module mesh

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes

"""
`all_gather`

Gathers along the `gather_axis` tensor axis.

# Example
```mlir
mesh.mesh @mesh0(shape = 2x2)
...
%1 = mesh.all_gather %0 on @mesh0 mesh_axes = [1] gather_axis = 1
  : tensor<2x2xi8> -> tensor<2x4xi8>
```
Input:
```
                 +-------+-------+
device (0, 0) -> |  1  2 |  5  6 | <- device (0, 1)
                 |  3  4 |  7  8 |
                 +-------+-------+
device (1, 0) -> |  9 10 | 13 14 | <- device (1, 1)
                 | 11 12 | 15 16 |
                 +-------+-------+
```
Result:
```
gather tensor
axis 1
------------>
+-------------+
|  1  2  5  6 | <- devices (0, 0) and (0, 1)
|  3  4  7  8 |
+-------------+
|  9 10 13 14 | <- devices (1, 0) and (1, 1)
| 11 12 15 16 |
+-------------+
```
"""
function all_gather(input::Value; result::IR.Type, mesh, mesh_axes=nothing, gather_axis, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), NamedAttribute("gather_axis", gather_axis), ]
    !isnothing(mesh_axes) && push!(attributes, NamedAttribute("mesh_axes", mesh_axes))
    
    IR.create_operation(
        "mesh.all_gather", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`all_reduce`

The accumulation element type is specified by the result type and
it does not need to match the input element type.
The input element is converted to the result element type before
performing the reduction.

Attributes:
`reduction`: Indicates the reduction method.

# Example
```
%1 = mesh.all_reduce %0 on @mesh0 mesh_axes = [1, 0] reduction = <max>
  : tensor<3x4xf32> -> tensor<3x4xf64>
```
"""
function all_reduce(input::Value; result::IR.Type, mesh, mesh_axes=nothing, reduction=nothing, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), ]
    !isnothing(mesh_axes) && push!(attributes, NamedAttribute("mesh_axes", mesh_axes))
    !isnothing(reduction) && push!(attributes, NamedAttribute("reduction", reduction))
    
    IR.create_operation(
        "mesh.all_reduce", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`all_slice`

Slice along the `slice_axis` tensor axis.
This operation can be thought of as the inverse of all-gather.
Technically, it is not required that all processes have the same input tensor.
Each process will slice a piece of its local tensor based on its in-group device index.
The operation does not communicate data between devices. 

# Example
```mlir
mesh.mesh @mesh0(shape = 2x2)
...
%1 = mesh.all_slice %0 on @mesh0 mesh_axes = [1] slice_axis = 1
  : tensor<2x4xi8> -> tensor<2x2xi8>
```
Input:
```
+-------------+
|  1  2  5  6 | <- devices (0, 0) and (0, 1)
|  3  4  7  8 |
+-------------+
|  9 10 13 14 | <- devices (1, 0) and (1, 1)
| 11 12 15 16 |
+-------------+
```
Result:
```
gather tensor
axis 1
------------>
                 +-------+-------+
device (0, 0) -> |  1  2 |  5  6 | <- device (0, 1)
                 |  3  4 |  7  8 |
                 +-------+-------+
device (1, 0) -> |  9 10 | 13 14 | <- device (1, 1)
                 | 11 12 | 15 16 |
                 +-------+-------+
```
"""
function all_slice(input::Value; result::IR.Type, mesh, mesh_axes=nothing, slice_axis, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), NamedAttribute("slice_axis", slice_axis), ]
    !isnothing(mesh_axes) && push!(attributes, NamedAttribute("mesh_axes", mesh_axes))
    
    IR.create_operation(
        "mesh.all_slice", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`all_to_all`

Performs an all-to-all on tensor pieces split along `split_axis`.
The resulting pieces are concatenated along `concat_axis` on ech device.

# Example
```
mesh.mesh @mesh0(shape = 3)
...
%1 = mesh.all_to_all %0 on @mesh0 mesh_axes = [0]
  split_axis = 0 concat_axis = 0
  : tensor<3x2xi8> -> tensor<3x2xi8>
```
Input:
```
 device  device  device
 (0)     (1)     (2)
+-------+-------+-------+  | split and concat along
| 11 12 | 21 22 | 31 32 |  | tensor axis 0
| 13 14 | 23 24 | 33 34 |  ↓
| 15 16 | 25 26 | 35 36 |
+-------+-------+-------+
```
Result:
```
 device  device  device
 (0)     (1)     (2)
+-------+-------+-------+
| 11 12 | 13 14 | 15 16 |
| 21 22 | 23 24 | 25 26 |
| 31 32 | 33 34 | 35 36 |
+-------+-------+-------+
```
"""
function all_to_all(input::Value; result::IR.Type, mesh, mesh_axes=nothing, split_axis, concat_axis, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), NamedAttribute("split_axis", split_axis), NamedAttribute("concat_axis", concat_axis), ]
    !isnothing(mesh_axes) && push!(attributes, NamedAttribute("mesh_axes", mesh_axes))
    
    IR.create_operation(
        "mesh.all_to_all", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`broadcast`

Broadcast the tensor on `root` to all devices in each respective group.
The operation broadcasts along mesh axes `mesh_axes`.
The `root` device specifies the in-group multi-index that is broadcast to
all other devices in the group.

# Example
```
mesh.mesh @mesh0(shape = 2x2)

%1 = mesh.broadcast %0 on @mesh0
  mesh_axes = [0]
  root = [0]
  : (tensor<2xi8>) -> tensor<2xi8>
```

Input:
```
                 +-------+-------+                   | broadcast
device (0, 0) -> |  1  2 |  3  4 | <- device (0, 1)  | along axis 0
                 +-------+-------+                   ↓
device (1, 0) -> |       |       | <- device (1, 1) 
                 +-------+-------+
```

Output:
```
                 +-------+-------+
device (0, 0) -> |  1  2 |  3  4 | <- device (0, 1)
                 +-------+-------+
device (1, 0) -> |  1  2 |  3  4 | <- device (1, 1)
                 +-------+-------+
```
"""
function broadcast(input::Value, root_dynamic::Vector{Value}; result::IR.Type, mesh, mesh_axes=nothing, root, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, root_dynamic..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), NamedAttribute("root", root), ]
    !isnothing(mesh_axes) && push!(attributes, NamedAttribute("mesh_axes", mesh_axes))
    
    IR.create_operation(
        "mesh.broadcast", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`gather`

Gathers on device `root` along the `gather_axis` tensor axis.
`root` specifies the coordinates of a device along `mesh_axes`.
It uniquely identifies the root device for each device group.
The result tensor on non-root devices is undefined.
Using it will result in undefined behavior.

# Example
```mlir
mesh.mesh @mesh0(shape = 2x2)
...
%1 = mesh.gather %0 on @mesh0 mesh_axes = [1]
  gather_axis = 1 root = [1]
  : (tensor<2x2xi8>) -> tensor<2x4xi8>
```
Input:
```
                  gather tensor
                  axis 1
                  ------------>
                 +-------+-------+
device (0, 0) -> |  1  2 |  5  6 | <- device (0, 1)
                 |  3  4 |  7  8 |
                 +-------+-------+
device (1, 0) -> |  9 10 | 13 14 | <- device (1, 1)
                 | 11 12 | 15 16 |
                 +-------+-------+
```
Result:
```
+-------------+
|  1  2  5  6 | <- devices (0, 1)
|  3  4  7  8 |
+-------------+
|  9 10 13 14 | <- devices (1, 1)
| 11 12 15 16 |
+-------------+
```
Devices `(0, 0)` and `(1, 0)` have undefined result.
"""
function gather(input::Value, root_dynamic::Vector{Value}; result::IR.Type, mesh, mesh_axes=nothing, gather_axis, root, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, root_dynamic..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), NamedAttribute("gather_axis", gather_axis), NamedAttribute("root", root), ]
    !isnothing(mesh_axes) && push!(attributes, NamedAttribute("mesh_axes", mesh_axes))
    
    IR.create_operation(
        "mesh.gather", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`mesh_`

The mesh.mesh operation is a symbol operation that identifies a specific
mesh. The operation has three attributes:

1. `sym_name`: This attribute uniquely identifies the name of the mesh.
This name serves as a symbolic reference to the mesh throughout
the MLIR module, allowing for consistent referencing and easier debugging.

2. `shape`: This attribute represents the shape of the device mesh.
It uses the same notation as a tensor shape. Also allowing for dynamic
dimensions.
This flexibility allows for dynamic device assignment or configurations
where the exact number of devices might not be determined during compile
time.
For example `2x?x4`.

# Example
```
// A device mesh with 3 axes, the total device number is 4 * 8 * 12
// The dimension sizes are 4, 8, 12 
mesh.mesh @mesh0(shape = 4x8x12)

// A device mesh with 2 axes, the total device number is unknown
// The first dimension size is 4 and the second is unknown
mesh.mesh @mesh1(shape = 4x?)

// A device mesh with 2 axes, the total device number is unknown
// The first dimension size is unknown and the second is 4
mesh.mesh @mesh2(shape = ?x4)

// A device mesh with 2 axes, the number of devices along both axes
// is unknown
mesh.mesh @mesh3(shape = ?x?)
```
"""
function mesh_(; sym_name, shape, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("sym_name", sym_name), NamedAttribute("shape", shape), ]
    
    IR.create_operation(
        "mesh.mesh", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end


function mesh_shape(; result::Vector{IR.Type}, mesh, axes=nothing, location=Location())
    op_ty_results = IR.Type[result..., ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), ]
    !isnothing(axes) && push!(attributes, NamedAttribute("axes", axes))
    
    IR.create_operation(
        "mesh.mesh_shape", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`neighbors_linear_indices`

# Example
```
mesh.mesh @mesh0(shape = 10x20x30)
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c3 = arith.constant 3 : index
%idx = mesh.neighbors_linear_indices on @mesh[%c1, %c2, %c3] split_axes = [1] : index
```
The above returns two indices, `633` and `693`, which correspond to the
index of the previous process `(1, 1, 3)`, and the next process 
`(1, 3, 3) along the split axis `1`.

A negative value is returned if there is no neighbor in the respective
direction along the given `split_axes`.
"""
function neighbors_linear_indices(device::Vector{Value}; neighbor_down=nothing::Union{Nothing, IR.Type}, neighbor_up=nothing::Union{Nothing, IR.Type}, mesh, split_axes, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[device..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), NamedAttribute("split_axes", split_axes), ]
    !isnothing(neighbor_down) && push!(op_ty_results, neighbor_down)
    !isnothing(neighbor_up) && push!(op_ty_results, neighbor_up)
    
    IR.create_operation(
        "mesh.neighbors_linear_indices", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`process_linear_index`

# Example
```
%idx = mesh.process_linear_index on @mesh : index
```
if `@mesh` has shape `(10, 20, 30)`, a device with multi
index `(1, 2, 3)` will have linear index `3 + 30*2 + 20*30*1`.
"""
function process_linear_index(; result=nothing::Union{Nothing, IR.Type}, mesh, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), ]
    !isnothing(result) && push!(op_ty_results, result)
    
    IR.create_operation(
        "mesh.process_linear_index", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`process_multi_index`

It is used in the SPMD format of IR.
The `axes` mush be non-negative and less than the total number of mesh axes.
If the axes are empty then get the index along all axes.
"""
function process_multi_index(; result::Vector{IR.Type}, mesh, axes=nothing, location=Location())
    op_ty_results = IR.Type[result..., ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), ]
    !isnothing(axes) && push!(attributes, NamedAttribute("axes", axes))
    
    IR.create_operation(
        "mesh.process_multi_index", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`recv`

Receive from a device within a device group.
"""
function recv(input::Value, source_dynamic::Vector{Value}; result::IR.Type, mesh, mesh_axes=nothing, source=nothing, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, source_dynamic..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), ]
    !isnothing(mesh_axes) && push!(attributes, NamedAttribute("mesh_axes", mesh_axes))
    !isnothing(source) && push!(attributes, NamedAttribute("source", source))
    
    IR.create_operation(
        "mesh.recv", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`reduce`

Reduces on device `root` within each device group.
`root` specifies the coordinates of a device along `mesh_axes`.
It uniquely identifies the root device within its device group.
The accumulation element type is specified by the result type and
it does not need to match the input element type.
The input element is converted to the result element type before
performing the reduction.

Attributes:
`reduction`: Indicates the reduction method.

# Example
```
%1 = mesh.reduce %0 on @mesh0 mesh_axes = [1, 0]
  reduction = <max> root = [2, 3]
  : (tensor<3x4xf32>) -> tensor<3x4xf64>
```
"""
function reduce(input::Value, root_dynamic::Vector{Value}; result::IR.Type, mesh, mesh_axes=nothing, reduction=nothing, root, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, root_dynamic..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), NamedAttribute("root", root), ]
    !isnothing(mesh_axes) && push!(attributes, NamedAttribute("mesh_axes", mesh_axes))
    !isnothing(reduction) && push!(attributes, NamedAttribute("reduction", reduction))
    
    IR.create_operation(
        "mesh.reduce", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`reduce_scatter`

After the reduction, the result is scattered within each device group.
The tensor is split along `scatter_axis` and the pieces distributed
across the device group.
# Example
```
mesh.mesh @mesh0(shape = 2x2)
...
%1 = mesh.reduce_scatter %0 on @mesh0 mesh_axes = [1]
  reduction = <max> scatter_axis = 0
  : tensor<3x4xf32> -> tensor<1x4xf64>
```
Input:
```
                          device
                          (0, 1)
                             ↓
                 +-------+-------+  | scatter tensor
device (0, 0) -> |  1  2 |  5  6 |  | axis 0
                 |  3  4 |  7  8 |  ↓
                 +-------+-------+
device (1, 0) -> |  9 10 | 13 14 |
                 | 11 12 | 15 16 |
                 +-------+-------+
                            ↑
                          device
                          (1, 1)
```
Result:
```
+-------+
|  6  8 | <- devices (0, 0)
+-------+
| 10 12 | <- devices (0, 1)
+-------+
| 22 24 | <- devices (1, 0)
+-------+
| 26 28 | <- devices (1, 1)
+-------+
```
"""
function reduce_scatter(input::Value; result::IR.Type, mesh, mesh_axes=nothing, reduction=nothing, scatter_axis, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), NamedAttribute("scatter_axis", scatter_axis), ]
    !isnothing(mesh_axes) && push!(attributes, NamedAttribute("mesh_axes", mesh_axes))
    !isnothing(reduction) && push!(attributes, NamedAttribute("reduction", reduction))
    
    IR.create_operation(
        "mesh.reduce_scatter", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`scatter`

For each device group split the input tensor on the `root` device along
axis `scatter_axis` and scatter the parts across the group devices.

# Example
```
mesh.mesh @mesh0(shape = 2x2)
%1 = mesh.scatter %0 on @mesh0 mesh_axes = [0]
  scatter_axis = 0
  root = [1]
  : (tensor<2x2xi8>) -> tensor<1x2xi8>
```

Input:
```
                          device
                          (0, 1)
                             ↓
                 +-------+-------+  | scatter tensor
device (0, 0) -> |       |       |  | axis 0
                 |       |       |  ↓
                 +-------+-------+
device (1, 0) -> |  1  2 |  5  6 |
                 |  3  4 |  7  8 |
                 +-------+-------+
                            ↑
                          device
                          (1, 1)
```

Result:
```
                          device
                          (0, 1)
                             ↓
                 +-------+-------+
device (0, 0) -> |  1  2 |  5  6 |
                 +-------+-------+ 
device (1, 0) -> |  3  4 |  7  8 |
                 +-------+-------+
                            ↑
                          device
                          (1, 1)
```
"""
function scatter(input::Value, root_dynamic::Vector{Value}; result::IR.Type, mesh, mesh_axes=nothing, scatter_axis, root, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, root_dynamic..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), NamedAttribute("scatter_axis", scatter_axis), NamedAttribute("root", root), ]
    !isnothing(mesh_axes) && push!(attributes, NamedAttribute("mesh_axes", mesh_axes))
    
    IR.create_operation(
        "mesh.scatter", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`send`

Send from one device to another within a device group.
"""
function send(input::Value, destination_dynamic::Vector{Value}; result::IR.Type, mesh, mesh_axes=nothing, destination, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, destination_dynamic..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), NamedAttribute("destination", destination), ]
    !isnothing(mesh_axes) && push!(attributes, NamedAttribute("mesh_axes", mesh_axes))
    
    IR.create_operation(
        "mesh.send", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`shard`

The mesh.shard operation is designed to specify and guide the sharding
behavior of a tensor value across a mesh topology. This operation has two
operands and two optional attributes:

1. `input`: This operand represents the tensor value that needs to be
annotated for sharding.

2. `sharding`: This attribute is type of `MeshShardingType`, which is the core data
structure to represent distribution of a tensor on a mesh. it is typically defiend
by an `mesh.sharding` operation.

3. `annotate_for_users`: A unit attribute addressing the scenario when a
tensor\'s sharding annotation differs based on its context of use (either as
a result or an operand). If specified, the sharding pertains to specific
users of the tensor value, indicating how it should be considered when used
as an operand in subsequent operations. If not, the sharding applies to the
operation that defines the tensor value.

# Example
```
func.func @only_result_annotated(%arg0 : tensor<4x8xf32>) -> () {
  %sharding = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %sharding : tensor<4x8xf32>
  ...
}

func.func @only_operand_annotated(%arg0 : tensor<4x8xf32>) -> () {
  %sharding = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %sharding annotate_for_users : tensor<4x8xf32>
  ...
}

func.func @two_operands_annotated(%arg0 : tensor<4x8xf32>, %arg1 : tensor<16x8xf32>) -> () {
  %sharding = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %sharding annotate_for_users : tensor<4x8xf32>
  %1 = mesh.shard %arg1 to %sharding annotate_for_users : tensor<16x8xf32>
  ...
}

// The first mesh.shard op applies to %arg0, the second mesh.shard op
// applies for the operand of op0, the third mesh.shard op applies for the
// operand of op2
func.func @both_result_and_multi_operands_annotated(
    %arg0 : tensor<4x8xf32>) -> () {
  %sharding = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %sharding : tensor<4x8xf32>
  %sharding1 = mesh.sharding @mesh0 split_axes = [[1]] : !mesh.sharding
  %1 = mesh.shard %0 to %sharding1 annotate_for_users : tensor<4x8xf32>
  %sharding2 = mesh.sharding @mesh0 split_axes = [[2]] : !mesh.sharding
  %2 = mesh.shard %0 to %sharding2 annotate_for_users : tensor<4x8xf32>
  \"op0\"(%1) : ...
  \"op1\"(%2) : ...
  ...
}
```

The following usages are undefined:
```
func.func @annotate_on_same_result_with_different_sharding(
    %arg0 : tensor<4x8xf32>) -> () {
  %sharding1 = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
  %sharding2 = mesh.sharding @mesh0 split_axes = [[1]] : !mesh.sharding
  %0 = mesh.shard %arg0 to \$sharding1 : tensor<4x8xf32>
  %1 = mesh.shard %0 to sharding2 : tensor<4x8xf32>
  ...
}

func.func @annotate_on_same_result_same_value_with_different_sharding(
    %arg0 : tensor<4x8xf32>) -> () {
  %sharding1 = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
  %sharding2 = mesh.sharding @mesh0 split_axes = [[1]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %sharding1 : tensor<4x8xf32>
  %1 = mesh.shard %arg0 to %sharding2 : tensor<4x8xf32>
  ...
}

func.func @annotate_on_same_operand_with_different_sharding(
    %arg0 : tensor<4x8xf32>) -> () {
  %sharding1 = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
  %sharding2 = mesh.sharding @mesh0 split_axes = [[1]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %sharding1 annotate_for_users : tensor<4x8xf32>
  %1 = mesh.shard %0 to %sharding2 annotate_for_users : tensor<4x8xf32>
  ...
}

func.func @result_annotated_after_operand(
    %arg0 : tensor<4x8xf32>) -> () {
  %sharding1 = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
  %sharding2 = mesh.sharding @mesh0 split_axes = [[1]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %sharding1 annotate_for_users : tensor<4x8xf32>
  %1 = mesh.shard %0 to %sharding2 : tensor<4x8xf32>
  ...
}
```
"""
function shard(src::Value, sharding::Value; result=nothing::Union{Nothing, IR.Type}, annotate_for_users=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[src, sharding, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(annotate_for_users) && push!(attributes, NamedAttribute("annotate_for_users", annotate_for_users))
    
    IR.create_operation(
        "mesh.shard", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`shard_shape`

The device/process id is a linearized id of the device/process in the mesh.
This operation might be used during spmdization when the shard shape depends
on (non-constant) values used in `mesh.sharding`.
"""
function shard_shape(sharding::Value, device::Value; result::Vector{IR.Type}, shape, location=Location())
    op_ty_results = IR.Type[result..., ]
    operands = Value[sharding, device, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("shape", shape), ]
    
    IR.create_operation(
        "mesh.shard_shape", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`sharding`

The MeshSharding specifies how a tensor is sharded and distributed across the
process mesh. It is typically used in a `mesh.shard` operation.
The operation has the follwing attributes and operands:

1. `mesh`: this attribute is a FlatSymbolRefAttr that refers to the device
mesh where the distributed tensor is placed. The symbol must resolve to a
`mesh.mesh` operation.

2. `split_axes`: is an array composed of int64_t sub-arrays. The outer array\'s
maximum size is the `rank` of the related tensor. For the i-th sub-array, if
its value is [x, y], it indicates that the tensor\'s i-th dimension is splitted
along the x and y axes of the device mesh.

3. [Optional] `partial_axes`: if not empty, this signifies that the tensor is partial
one along the specified mesh axes. An all-reduce should be applied to obtain
the complete tensor, with reduction type being specified by `partial_type`.

4. [Optional] `partial_type`: indicates the reduction type of the possible all-reduce
op. It has 4 possible values:
`generic`: is not an allowed value inside a shard attribute.

5. [Optional] Sizes of halos to be added for each sharded tensor dimension.
`halo_sizes` is provided as a flattened 1d array of i64s, 2 values for each
sharded dimension. `halo_sizes = [1, 2]` means that the first sharded dimension
gets an additional halo of size 1 at the start of the first dimension and a halo
size is 2 at its end. `halo_sizes = [1, 2, 2, 3]` defines halos for the first 2
sharded dimensions e.g. the first sharded dimension gets `[1,2]` halos and the
seconds gets `[2,3]` halos. `?` indicates dynamic halo sizes.

6. [Optional] Offsets for each shard and sharded tensor dimension.
`sharded_dims_offsets` is provided as a flattened 1d array of i64s. For each
sharded tensor dimension the offsets (starting index) of all shards in that
dimension and an additional value for the end of the last shard are provided.
For a 1d sharding this means that position `i` has the exclusive prefix sum for
shard `i`, and since only contiguous sharding is supported, its inclusive prefix
sum is at position \'i+1\'.

Assuming a 3d-tensor of shape 32x32x32 with the first 2 dimensions being sharded,
`sharded_dims_offsets` = [0, 24, 32, 0, 20, 32] means that the first device of
the device-mesh will get a shard of shape 24x20x32 and the second device will get
a shard of shape 8x12x32. `?` indicates dynamic shard dimensions.

`halo_sizes` and `sharded_dims_offsets` are mutually exclusive.

Examples:

```
mesh.mesh @mesh0(shape = 2x2x4)
mesh.mesh @mesh1d_4(shape = 4)

// The tensor is fully replicated on @mesh0.
// Currently, there must be at least one sub-array present in axes, even
// if it\'s empty. Otherwise, a parsing error will occur.
%sharding0 = mesh.sharding @mesh0 split_axes = [[]]

// The tensor is sharded on the first dimension along axis 0 of @mesh0
%sharding1 = mesh.sharding @mesh0 split_axes = [[0]]

// The tensor is sharded on its first dimension along axis 0 of @mesh0 and
// it is also a partial_sum along mesh axis 1.
%sharding2 = mesh.sharding @mesh0 split_axes = [[0] split_axes = []] partial = sum[1]

// The tensor is sharded on its first dimension along axis 0 of @mesh0 and
// it is also a partial_max along mesh axis 1.
%sharding3 = mesh.sharding @mesh0 split_axes = [[0]] partial = max[1]

// Could be used for a mesh.shard op
%sharded0 = mesh.shard %arg0 to %sharding3 : tensor<4x8xf32>

// The tensor is sharded on its first dimension along axis 0 of @mesh0 and
// and it has halo-sizes of 1 and 2 on the sharded dim.
%halo_sharding = mesh.sharding @mesh0 split_axes = [[0]] halo_sizes = [1, 2]
%sharded1 = mesh.shard %arg0 to %halo_sharding : tensor<4x8xf32>

// The tensor is sharded on its second dimension along axis 0 of @mesh1d_4
// and it has pre-defined shard sizes. The shards of the devices will have
// the following shapes: [4x2, 4x3, 4x4, 4x5]
%sharding4 = mesh.sharding @mesh1d_4 split_axes = [[], [0]] sharded_dims_offsets = [0, 2, 5, 9, 14]
%sharded2 = mesh.shard %arg0 to %sharding4 : tensor<4x14xf32>
```
"""
function sharding(dynamic_sharded_dims_offsets::Vector{Value}, dynamic_halo_sizes::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, mesh, split_axes, partial_axes=nothing, partial_type=nothing, static_sharded_dims_offsets=nothing, static_halo_sizes=nothing, location=Location())
    op_ty_results = IR.Type[]
    operands = Value[dynamic_sharded_dims_offsets..., dynamic_halo_sizes..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), NamedAttribute("split_axes", split_axes), ]
    push!(attributes, operandsegmentsizes([length(dynamic_sharded_dims_offsets), length(dynamic_halo_sizes), ]))
    !isnothing(result) && push!(op_ty_results, result)
    !isnothing(partial_axes) && push!(attributes, NamedAttribute("partial_axes", partial_axes))
    !isnothing(partial_type) && push!(attributes, NamedAttribute("partial_type", partial_type))
    !isnothing(static_sharded_dims_offsets) && push!(attributes, NamedAttribute("static_sharded_dims_offsets", static_sharded_dims_offsets))
    !isnothing(static_halo_sizes) && push!(attributes, NamedAttribute("static_halo_sizes", static_halo_sizes))
    
    IR.create_operation(
        "mesh.sharding", location;
        operands, owned_regions, successors, attributes,
        results=(length(op_ty_results) == 0 ? nothing : op_ty_results),
        result_inference=(length(op_ty_results) == 0 ? true : false)
    )
end

"""
`shift`

Within each device group shift along mesh axis `shift_axis` by an offset
`offset`.
The result on devices that do not have a corresponding source is undefined.
`shift_axis` must be one of `mesh_axes`.
If the `rotate` attribute is present,
instead of a shift a rotation is done.

# Example
```
mesh.mesh @mesh0(shape = 2x4)
%1 = mesh.shift on @mesh0 mesh_axes = [1]
  shift_axis = 1 offset = 2 rotate
  : tensor<2xi8> -> tensor<2xi8>
```

Input:
```
mesh axis 1
----------->

+----+----+----+----+
|  1 |  2 |  3 |  4 |
+----+----+----+----+
|  5 |  6 |  7 |  8 |
+----+----+----+----+
```

Result:
```
+----+----+----+----+
|  3 |  4 |  1 |  2 |
+----+----+----+----+
|  7 |  8 |  5 |  6 |
+----+----+----+----+
```
"""
function shift(input::Value; result::IR.Type, mesh, mesh_axes=nothing, shift_axis, offset, rotate=nothing, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), NamedAttribute("shift_axis", shift_axis), NamedAttribute("offset", offset), ]
    !isnothing(mesh_axes) && push!(attributes, NamedAttribute("mesh_axes", mesh_axes))
    !isnothing(rotate) && push!(attributes, NamedAttribute("rotate", rotate))
    
    IR.create_operation(
        "mesh.shift", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`update_halo`

This operation updates halo regions of shards, e.g. if their sharding
specified halos and the actual tensor/memref data might have changed
on the remote devices. Changes might be caused by mutating operations
and/or if the new halo regions are larger than the existing ones.

Destination is supposed to be initialized with the local data (not halos).

Assumes all devices hold tensors with same-sized halo data as specified
by `source_halo_sizes/static_source_halo_sizes` and
`destination_halo_sizes/static_destination_halo_sizes` in source shard
and destination/result shard.

`split_axes` specifies for each tensor axis along which mesh axes its halo
data is updated.
"""
function update_halo(destination::Value, halo_sizes::Vector{Value}; result::IR.Type, mesh, split_axes, static_halo_sizes=nothing, location=Location())
    op_ty_results = IR.Type[result, ]
    operands = Value[destination, halo_sizes..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("mesh", mesh), NamedAttribute("split_axes", split_axes), ]
    !isnothing(static_halo_sizes) && push!(attributes, NamedAttribute("static_halo_sizes", static_halo_sizes))
    
    IR.create_operation(
        "mesh.update_halo", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end
end # mesh

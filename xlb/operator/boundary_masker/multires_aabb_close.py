import warp as wp
import numpy as np
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker import MeshMaskerAABBClose
from xlb.operator.operator import Operator
import neon


# Create a list of the connected components
def find_connected_components(connectivity):
    visited = set()
    components = []

    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor, is_connected in enumerate(connectivity[node]):
            if is_connected and neighbor not in visited:
                dfs(neighbor, component)

    for i in range(1, len(connectivity)):
        if i not in visited:
            component = []
            dfs(i, component)
            components.append(component)

    return components

# Create a list of combined sizes of the connected components
def calculate_component_sizes(components, comp_sizes):
    sizes = []
    for component in components:
        size = sum(comp_sizes[seed] for seed in component)
        sizes.append(size)
    return sizes

# Create a map from seeds to one for seeds in the largest connected component and zero for all other seeds
def map_seeds_to_zero_one(components, largest_component_index, num_seeds):
    seed_map = np.zeros(num_seeds + 1, dtype=int)
    for seed in components[largest_component_index]:
        seed_map[seed] = 1
    return seed_map


class MultiresMeshMaskerAABBClose(MeshMaskerAABBClose):
    """
    Operator for creating boundary missing_mask from mesh using Axis-Aligned Bounding Box (AABB) voxelization
    in multiresolution simulations (NEON backend). It takes in a number of close_voxels to perform morphological
    operations (dilate followed by erode) to ensure small channels are filled with solid voxels.

    This version provides NEON-specific functionals working on multires partitions (mPartition) and bIndex.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        close_voxels: int = None,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend, close_voxels)
        if self.compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend.")

        # Build and store NEON dicts
        self.neon_functional_dict, self.neon_container_dict = self._construct_neon()

    def _construct_neon(self):
        # Use the warp functionals from the base (for reference), but implement NEON variants here
        functional_dict_warp, _ = self._construct_warp()
        functional_erode_warp = functional_dict_warp.get("functional_erode")
        functional_dilate_warp = functional_dict_warp.get("functional_dilate")
        functional_solid = functional_dict_warp.get("functional_solid")
        # We will not directly reuse functional_solid / functional_aabb from warp; we write NEON-specific ones.

        # We also need lattice info for neighbor iteration
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices

        # Set local constants
        lattice_central_index = self.velocity_set.center_index

        # Main AABB close: sets bc_mask, missing_mask, distances based on solid_mask
        # bc_mask: wp.uint8, missing_mask: wp.uint8, distances: dtype from precision policy (float)
        @wp.func
        def mres_functional_aabb(
            index: Any,
            mesh_id: wp.uint64,
            id_number: wp.int32,
            distances_pn: Any,  # mPartition(dtype=distance type), cardinality=_q
            bc_mask_pn: Any,  # mPartition_uint8, cardinality=1
            missing_mask_pn: Any,  # mPartition_uint8, cardinality=_q
            solid_mask_pn: Any,  # mPartition_uint8, cardinality=1
            needs_mesh_distance: bool,
        ):
            # Cell center from bc_mask partition
            cell_center = self.helper_masker.index_to_position(bc_mask_pn, index)

            # If already solid or bc, mark solid
            solid_val = wp.neon_read(solid_mask_pn, index, 0)
            bc_val = wp.neon_read(bc_mask_pn, index, 0)
            if solid_val == wp.uint8(255) or bc_val == wp.uint8(255):
                wp.neon_write(bc_mask_pn, index, 0, wp.uint8(255))
                return

            # loop lattice directions
            for direction_idx in range(_q):
                # skip central if provided by velocity set
                if direction_idx == lattice_central_index:
                    continue

                # If neighbor index is valid at this resolution level
                ngh = wp.neon_ngh_idx(wp.int8(_c[0, direction_idx]), wp.int8(_c[1, direction_idx]), wp.int8(_c[2, direction_idx]))
                is_valid = wp.bool(False)
                nval = wp.neon_read_ngh(solid_mask_pn, index, ngh, 0, wp.uint8(0), is_valid)
                if is_valid:
                    if nval == wp.uint8(255):
                        # Found solid neighbor -> boundary cell
                        self.write_field(bc_mask_pn, index, 0, wp.uint8(id_number))
                        self.write_field(missing_mask_pn, index, _opp_indices[direction_idx], wp.uint8(True))

                        if not needs_mesh_distance:
                            # No distance needed; continue to next direction
                            continue

                        # Compute mesh distance along lattice direction
                        dir_vec = wp.vec3f(
                            wp.float32(_c[0, direction_idx]),
                            wp.float32(_c[1, direction_idx]),
                            wp.float32(_c[2, direction_idx]),
                        )
                        max_length = wp.length(dir_vec)
                        # Avoid division by zero for any pathological dir (shouldn't happen)
                        norm_dir = dir_vec / (max_length if max_length > 0.0 else 1.0)
                        query = wp.mesh_query_ray(mesh_id, cell_center, norm_dir, 1.5 * max_length)
                        if query.result:
                            pos_mesh = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
                            dist = wp.length(pos_mesh - cell_center) - 0.5 * max_length
                            weight = dist / (max_length if max_length > 0.0 else 1.0)
                            # distances has cardinality _q; store into this channel
                            self.write_field(distances_pn, index, direction_idx, wp.float32(weight))
                        else:
                            self.write_field(distances_pn, index, direction_idx, wp.float32(1.0))

        @wp.func
        def mres_get_seeds(
            index: Any,
            solid_mask_pn: Any,  # mPartition_uint8, cardinality=1
            flood_mask_pn: Any,  # mPartition_uint8, cardinality=1
            num_seeds: wp.int32, 
            seed_count: Any,  # mPartition_int32, cardinality=1
        ):
            solid_val = wp.neon_read(solid_mask_pn, index, 0)
            # If solid and not yet flooded, check if seed
            # if solid_val == wp.uint8(255):
            #     seed_val = wp.atomic_add(seed_count, 0, 1)
            #     if seed_val < num_seeds:
            #         # Mark on flood mask
            #         self.write_field(flood_mask_pn, index, 0, wp.uint8(seed_val+1))
            #         return
            
            # If not solid, check if seed
            if solid_val != wp.uint8(255):
                seed_val = wp.atomic_add(seed_count, 0, 1)
                if seed_val < num_seeds:
                    # Mark on flood mask
                    self.write_field(flood_mask_pn, index, 0, wp.uint8(seed_val+1))
                    return
            
            wp.neon_write(flood_mask_pn, index, 0, wp.uint8(0))
            return
        
        @wp.func
        def mres_flood_fill(
            index: Any,
            solid_mask_pn: Any,  # mPartition_uint8, cardinality=1
            flood_mask_pn: Any,  # mPartition_uint8, cardinality=1
            flood_mask_out_pn: Any,  # mPartition_uint8, cardinality=1
            keep_going: Any,  # mPartition_bool, cardinality=1
        ):
            solid_val = self.read_field(solid_mask_pn, index, 0)
            flood_val = self.read_field(flood_mask_pn, index, 0)

            if flood_val > wp.uint8(0):
                # Already flooded, copy value to output
                self.write_field(flood_mask_out_pn, index, 0, flood_val)
                return
            if solid_val != wp.uint8(255) and flood_val == wp.uint8(0):
                max_val = wp.uint8(0)
                for l in range(_q):
                    if l == lattice_central_index:
                        continue
                    is_valid = wp.bool(False)
                    # direction = wp.vec3i(_c[0, l], _c[1, l], _c[2, l])
                    ngh = wp.neon_ngh_idx(wp.int8(_c[0, l]), wp.int8(_c[1, l]), wp.int8(_c[2, l]))
                    ngh_val = wp.neon_read_ngh(flood_mask_pn, index, ngh, 0, wp.uint8(0), is_valid)
                    if is_valid:
                        max_val = wp.max(max_val, ngh_val)
                    # else:
                    #     # Check to see if there is a finer neighbor that might have a value
                    #     if wp.neon_has_finer_ngh(flood_mask_pn, index, direction):
                    #         ngh_val = wp.neon_getUncle(flood_mask_pn, index, ngh)
                    #         max_val = wp.max(max_val, ngh_val)
                if max_val > wp.uint8(0):
                    self.write_field(flood_mask_out_pn, index, 0, max_val)
                    # self.write_field(keep_going, 0, 0, 1)
                    # wp.atomic_max(keep_going, 0, 1)
                    wp.atomic_add(keep_going, 0, 1)
            
            return
        
        @wp.func
        def mres_count_components(
            index: Any,
            solid_mask_pn: Any,  # mPartition_uint8, cardinality=1
            flood_mask_pn: Any,  # mPartition_uint8, cardinality=1
            connectivity: Any, # num_seeds x num_seeds bool array
            comp_counts: Any, # num_seeds int array
        ):
            solid_val = self.read_field(solid_mask_pn, index, 0)
            flood_val = self.read_field(flood_mask_pn, index, 0)

            if solid_val != wp.uint8(255):
                wp.atomic_add(comp_counts, flood_val, 1)
                for l in range(_q):
                    if l == lattice_central_index:
                        continue
                    is_valid = wp.bool(False)
                    ngh = wp.neon_ngh_idx(wp.int8(_c[0, l]), wp.int8(_c[1, l]), wp.int8(_c[2, l]))
                    ngh_val = wp.neon_read_ngh(flood_mask_pn, index, ngh, 0, wp.uint8(0), is_valid)
                    if is_valid:
                        if ngh_val != flood_val:
                            # We have connectivity
                            # self.write_field(connectivity, flood_val*num_seeds + ngh_val, 0, 1)
                            connectivity[flood_val, ngh_val] = wp.int64(1)
            
            # Should this be atomic?
            # self.write_field(comp_counts, flood_val, 0, self.read_field(comp_counts, flood_val, 0) + 1)
            
            return

        @wp.func
        def mres_update_solid(
            index: Any,
            solid_mask_pn: Any,  # mPartition_uint8, cardinality=1
            flood_mask_pn: Any,  # mPartition_uint8, cardinality=1
            max_comps: Any, # num_seeds int array
        ):
            flood_val = self.read_field(flood_mask_pn, index, 0)
            # max_check = self.read_field(max_comps, flood_val, 0)
            max_check = max_comps[flood_val]

            if max_check == wp.int32(1):
                self.write_field(solid_mask_pn, index, 0, wp.uint8(255))
            
            return

        # Containers
        @neon.Container.factory(name="UpdateSolid")
        def container_update_solid(solid_mask: wp.array3d(dtype=Any), flood_mask: wp.array3d(dtype=Any), max_comps: wp.array(dtype=Any), level: int):
            def update_launcher(loader: neon.Loader):
                loader.set_mres_grid(solid_mask.get_grid(), level)
                solid_mask_pn = loader.get_mres_write_handle(solid_mask)
                flood_mask_pn = loader.get_mres_read_handle(flood_mask)
                # max_comps_pn = loader.get_mres_write_handle(max_comps)

                @wp.func
                def kernel_solid(index: Any):
                    mres_update_solid(index, solid_mask_pn, flood_mask_pn, max_comps)

                loader.declare_kernel(kernel_solid)

            return update_launcher

        @neon.Container.factory(name="GetSeeds")
        def container_get_seeds(solid_mask: wp.array3d(dtype=Any), flood_mask: wp.array3d(dtype=Any), num_seeds: int, seed_count: wp.array3d(dtype=Any), level: int):
            def erode_launcher(loader: neon.Loader):
                loader.set_mres_grid(solid_mask.get_grid(), level)
                solid_mask_pn = loader.get_mres_read_handle(solid_mask)
                flood_mask_pn = loader.get_mres_write_handle(flood_mask)

                @wp.func
                def get_seeds_kernel(index: Any):
                    mres_get_seeds(index, solid_mask_pn, flood_mask_pn, num_seeds, seed_count)

                loader.declare_kernel(get_seeds_kernel)

            return erode_launcher

        @neon.Container.factory(name="FloodFill")
        def container_flood_fill(solid_mask: wp.array3d(dtype=Any), flood_mask: wp.array3d(dtype=Any), flood_mask_out: wp.array3d(dtype=Any), done: wp.array3d(dtype=Any), level: int):
            def erode_launcher(loader: neon.Loader):
                loader.set_mres_grid(solid_mask.get_grid(), level)
                solid_mask_pn = loader.get_mres_read_handle(solid_mask)
                flood_mask_pn = loader.get_mres_read_handle(flood_mask)
                flood_mask_out_pn = loader.get_mres_write_handle(flood_mask_out)
                # done_pn = loader.get_mres_write_handle(done)

                @wp.func
                def flood_fill_kernel(index: Any):
                    mres_flood_fill(index, solid_mask_pn, flood_mask_pn, flood_mask_out_pn, done)

                loader.declare_kernel(flood_fill_kernel)

            return erode_launcher

        @neon.Container.factory(name="CompCounts")
        def container_comp_counts(solid_mask: wp.array3d(dtype=Any), flood_mask: wp.array3d(dtype=Any), connectivity: wp.array3d(dtype=Any), comp_counts: Any, level: int):
            def comp_launcher(loader: neon.Loader):
                loader.set_mres_grid(solid_mask.get_grid(), level)
                solid_mask_pn = loader.get_mres_read_handle(solid_mask)
                flood_mask_pn = loader.get_mres_read_handle(flood_mask)
                # Do I need to load this?
                # connectivity_pn = loader.get_mres_write_handle(connectivity)
                # Do I need to load this?
                # comp_counts_pn = loader.get_mres_write_handle(comp_counts)

                @wp.func
                def comp_kernel(index: Any):
                    mres_count_components(index, solid_mask_pn, flood_mask_pn, connectivity, comp_counts)

                loader.declare_kernel(comp_kernel)

            return comp_launcher

        # Erode: f_field -> f_field_out
        @neon.Container.factory(name="Erode")
        def container_erode(f_field: wp.array3d(dtype=Any), f_field_out: wp.array3d(dtype=Any), level: int):
            def erode_launcher(loader: neon.Loader):
                loader.set_mres_grid(f_field.get_grid(), level)
                f_field_pn = loader.get_mres_read_handle(f_field)
                f_field_out_pn = loader.get_mres_write_handle(f_field_out)

                @wp.func
                def erode_kernel(index: Any):
                    functional_erode_warp(index, f_field_pn, f_field_out_pn)

                loader.declare_kernel(erode_kernel)

            return erode_launcher

        # Dilate: f_field -> f_field_out
        @neon.Container.factory(name="Dilate")
        def container_dilate(f_field: wp.array3d(dtype=Any), f_field_out: wp.array3d(dtype=Any), level: int):
            def dilate_launcher(loader: neon.Loader):
                loader.set_mres_grid(f_field.get_grid(), level)
                f_field_pn = loader.get_mres_read_handle(f_field)
                f_field_out_pn = loader.get_mres_write_handle(f_field_out)

                @wp.func
                def dilate_kernel(index: Any):
                    functional_dilate_warp(index, f_field_pn, f_field_out_pn)

                loader.declare_kernel(dilate_kernel)

            return dilate_launcher

        # Solid mask: voxelize mesh into solid_mask
        @neon.Container.factory(name="Solid")
        def container_solid(mesh_id: wp.uint64, solid_mask: wp.array3d(dtype=wp.uint8), level: int):
            def solid_launcher(loader: neon.Loader):
                loader.set_mres_grid(solid_mask.get_grid(), level)
                solid_mask_pn = loader.get_mres_write_handle(solid_mask)

                @wp.func
                def solid_kernel(index: Any):
                    # apply the functional
                    functional_solid(index, mesh_id, solid_mask_pn, wp.vec3f(0.0, 0.0, 0.0))

                loader.declare_kernel(solid_kernel)

            return solid_launcher

        # Main AABB container
        @neon.Container.factory(name="MeshMaskerAABBClose")
        def container(
            mesh_id: Any,
            id_number: Any,
            distances: Any,
            bc_mask: Any,
            missing_mask: Any,
            solid_mask: Any,
            needs_mesh_distance: Any,
            level: Any,
        ):
            def aabb_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                distances_pn = loader.get_mres_write_handle(distances)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)
                missing_mask_pn = loader.get_mres_write_handle(missing_mask)
                solid_mask_pn = loader.get_mres_write_handle(solid_mask)

                @wp.func
                def aabb_kernel(index: Any):
                    mres_functional_aabb(
                        index,
                        mesh_id,
                        id_number,
                        distances_pn,
                        bc_mask_pn,
                        missing_mask_pn,
                        solid_mask_pn,
                        needs_mesh_distance,
                    )

                loader.declare_kernel(aabb_kernel)

            return aabb_launcher

        container_dict = {
            "container_erode": container_erode,
            "container_dilate": container_dilate,
            "container_solid": container_solid,
            "container_aabb": container,
            "container_get_seeds": container_get_seeds,
            "container_flood_fill": container_flood_fill,
            "container_comp_counts": container_comp_counts,
            "container_update_solid": container_update_solid
        }

        # Expose NEON functionals too (in case callers want to reuse)
        functional_dict = {
            "mres_functional_aabb": mres_functional_aabb,
        }

        return functional_dict, container_dict

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(
        self,
        bc,
        distances,
        bc_mask,
        missing_mask,
        stream=0,
    ):
        # Prepare inputs
        mesh_id, bc_id = self._prepare_kernel_inputs(bc, bc_mask)

        grid = bc_mask.get_grid()
        # Create fields using new_field
        solid_mask = grid.new_field(cardinality=1, dtype=wp.uint8, memory_type=neon.MemoryType.device())
        solid_mask_out = grid.new_field(
            cardinality=1,
            dtype=wp.uint8,
            memory_type=neon.MemoryType.device(),
            # memory_type=neon.MemoryType.host_device()
        )

        flood_mask = grid.new_field(
            cardinality=1,
            dtype=wp.uint8,
            memory_type=neon.MemoryType.device(),
            # memory_type=neon.MemoryType.host_device()
        )
        flood_mask_out = grid.new_field(
            cardinality=1,
            dtype=wp.uint8,
            memory_type=neon.MemoryType.device(),
            # memory_type=neon.MemoryType.host_device()
        )
        
        for level in range(grid.num_levels):
            # Initialize to 0
            solid_mask.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)
            solid_mask_out.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)

            # Launch the neon containers
            container_solid = self.neon_container_dict["container_solid"](mesh_id, solid_mask, level)
            container_solid.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

            for _ in range(self.close_voxels):
                container_dilate = self.neon_container_dict["container_dilate"](solid_mask, solid_mask_out, level)
                container_dilate.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
                solid_mask, solid_mask_out = solid_mask_out, solid_mask

            for _ in range(self.close_voxels):
                container_erode = self.neon_container_dict["container_erode"](solid_mask, solid_mask_out, level)
                container_erode.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
                solid_mask, solid_mask_out = solid_mask_out, solid_mask

            # Now do flood fill, reuse solid_mask_out as flood_mask
            # flood_mask = solid_mask_out
            flood_mask.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)
            flood_mask_out.fill_run(level=level, value=wp.uint8(0), stream_idx=stream)
            num_seeds = 5
            # TODO: use device 0, needs to be changed for multi-gpu
            seed_count_array = wp.zeros(1, dtype=wp.int32)
            if level == 0:
                # Initialize flood seeds
                container_get_seeds = self.neon_container_dict["container_get_seeds"](solid_mask, flood_mask, num_seeds, seed_count_array, level)
                container_get_seeds.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
            
            iters = 0
            growth = 0
            done = False
            while not done:
                # keep_going_array[0] = wp.uint8(0)
                keep_going_array = wp.zeros(1, dtype=wp.int32)
                container_flood = self.neon_container_dict["container_flood_fill"](solid_mask, flood_mask, flood_mask_out, keep_going_array, level)
                container_flood.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
                keep_going = keep_going_array.numpy()
                growth += keep_going[0]
                done = (keep_going[0] == 0)
                flood_mask, flood_mask_out = flood_mask_out, flood_mask
                # done = self.read_field(keep_going_array, 0, 0) == 0
                print(f"Flood fill iter {iters} grew {keep_going[0]} voxels")
                iters += 1
            
            print(f"Flood fill completed in {iters} iterations, filled {growth} voxels")

            connectivity = wp.zeros((num_seeds+1, num_seeds+1), dtype=wp.int64)
            # comp_counts = wp.zeros(num_seeds+1, dtype=wp.int64)
            comp_counts = wp.zeros(num_seeds+1, dtype=wp.int32)
            
            # Could check connectivity and count components here
            container_comp = self.neon_container_dict["container_comp_counts"](solid_mask, flood_mask, connectivity, comp_counts, level)
            container_comp.run(0, container_runtime=neon.Container.ContainerRuntime.neon)
            
            print(connectivity.numpy())
            # Create a list of lists to hold connected components and another list that counts the total number of elements in each component
            components = find_connected_components(connectivity.numpy())
            print("Connected Components:")
            print(components)
            print(comp_counts.numpy())

            component_sizes = calculate_component_sizes(components, comp_counts.numpy())
            print("Component Sizes:")
            print(component_sizes)

            largest_component_index = np.argmax(component_sizes)
            seed_map = map_seeds_to_zero_one(components, largest_component_index, num_seeds)
            print("Seed Map (1 for largest component, 0 for others):")
            print(seed_map)

            # comp_map = wp.array(seed_map, dtype=wp.int32)
            comp_map = wp.from_numpy(seed_map, dtype=wp.int32)
            container_update_solid = self.neon_container_dict["container_update_solid"](solid_mask, flood_mask, comp_map, level)
            container_update_solid.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

            container_aabb = self.neon_container_dict["container_aabb"](
                mesh_id, bc_id, distances, bc_mask, missing_mask, solid_mask, wp.static(bc.needs_mesh_distance), level
            )
            container_aabb.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

        return distances, bc_mask, missing_mask

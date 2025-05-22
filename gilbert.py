#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2024 abetusk

import numpy as np
import sys
import torch


def gilbert_xyz2d(x, y, z, width, height, depth):
    """
    Generalized Hilbert ('Gilbert') space-filling curve for arbitrary-sized
    3D rectangular grids. Generates discrete 3D coordinates to fill a cuboid
    of size (width x height x depth). Even sizes are recommended in 3D.
    """

    if width >= height and width >= depth:
       return gilbert_xyz2d_r(0,x,y,z,
                              0, 0, 0,
                              width, 0, 0,
                              0, height, 0,
                              0, 0, depth)

    elif height >= width and height >= depth:
       return gilbert_xyz2d_r(0,x,y,z,
                              0, 0, 0,
                              0, height, 0,
                              width, 0, 0,
                              0, 0, depth)

    else: # depth >= width and depth >= height
       return gilbert_xyz2d_r(0,x,y,z,
                              0, 0, 0,
                              0, 0, depth,
                              width, 0, 0,
                              0, height, 0)


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)

def in_bounds(x, y, z, x_s, y_s, z_s, ax, ay, az, bx, by, bz, cx, cy, cz):

    dx = ax + bx + cx
    dy = ay + by + cy
    dz = az + bz + cz

    if dx < 0:
        if (x > x_s) or (x <= (x_s + dx)): return False
    else:
        if (x < x_s) or (x >= (x_s + dx)): return False

    if dy < 0:
        if (y > y_s) or (y <= (y_s + dy)): return False
    else:
        if (y < y_s) or (y >= (y_s + dy)): return False

    if dz <0:
        if (z > z_s) or (z <= (z_s + dz)): return False
    else:
        if (z < z_s) or (z >= (z_s + dz)): return False

    return True


def gilbert_xyz2d_r(cur_idx,
                    x_dst,y_dst,z_dst,
                    x, y, z,
                    ax, ay, az,
                    bx, by, bz,
                    cx, cy, cz):

    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)

    (dax, day, daz) = (sgn(ax), sgn(ay), sgn(az)) # unit major direction ("right")
    (dbx, dby, dbz) = (sgn(bx), sgn(by), sgn(bz)) # unit ortho direction ("forward")
    (dcx, dcy, dcz) = (sgn(cx), sgn(cy), sgn(cz)) # unit ortho direction ("up")

    # trivial row/column fills
    if h == 1 and d == 1:
        return cur_idx + (dax*(x_dst - x)) + (day*(y_dst - y)) + (daz*(z_dst - z))

    if w == 1 and d == 1:
        return cur_idx + (dbx*(x_dst - x)) + (dby*(y_dst - y)) + (dbz*(z_dst - z))

    if w == 1 and h == 1:
        return cur_idx + (dcx*(x_dst - x)) + (dcy*(y_dst - y)) + (dcz*(z_dst - z))

    (ax2, ay2, az2) = (ax//2, ay//2, az//2)
    (bx2, by2, bz2) = (bx//2, by//2, bz//2)
    (cx2, cy2, cz2) = (cx//2, cy//2, cz//2)

    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)

    # prefer even steps
    if (w2 % 2) and (w > 2):
       (ax2, ay2, az2) = (ax2 + dax, ay2 + day, az2 + daz)

    if (h2 % 2) and (h > 2):
       (bx2, by2, bz2) = (bx2 + dbx, by2 + dby, bz2 + dbz)

    if (d2 % 2) and (d > 2):
       (cx2, cy2, cz2) = (cx2 + dcx, cy2 + dcy, cz2 + dcz)

    # wide case, split in w only
    if (2*w > 3*h) and (2*w > 3*d):
        if in_bounds(x_dst,y_dst,z_dst,
                     x,y,z,
                     ax2,ay2,az2,
                     bx,by,bz,
                     cx,cy,cz):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x, y, z,
                                   ax2, ay2, az2,
                                   bx, by, bz,
                                   cx, cy, cz)
        cur_idx += abs( (ax2 + ay2 + az2)*(bx + by + bz)*(cx + cy + cz) )

        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+ax2, y+ay2, z+az2,
                               ax-ax2, ay-ay2, az-az2,
                               bx, by, bz,
                               cx, cy, cz)

    # do not split in d
    elif 3*h > 4*d:
        if in_bounds(x_dst,y_dst,z_dst,
                     x,y,z,
                     bx2,by2,bz2,
                     cx,cy,cz,
                     ax2,ay2,az2):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x, y, z,
                                   bx2, by2, bz2,
                                   cx, cy, cz,
                                   ax2, ay2, az2)
        cur_idx += abs( (bx2 + by2 + bz2)*(cx + cy + cz)*(ax2 + ay2 + az2) )

        if in_bounds(x_dst,y_dst,z_dst,
                     x+bx2,y+by2,z+bz2,
                     ax,ay,az,
                     bx-bx2,by-by2,bz-bz2,
                     cx,cy,cz):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x+bx2, y+by2, z+bz2,
                                   ax, ay, az,
                                   bx-bx2, by-by2, bz-bz2,
                                   cx, cy, cz)
        cur_idx += abs( (ax + ay + az)*((bx - bx2) + (by - by2) + (bz - bz2))*(cx + cy + cz) )

        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+(ax-dax)+(bx2-dbx),
                               y+(ay-day)+(by2-dby),
                               z+(az-daz)+(bz2-dbz),
                               -bx2, -by2, -bz2,
                               cx, cy, cz,
                               -(ax-ax2), -(ay-ay2), -(az-az2))

    # do not split in h
    elif 3*d > 4*h:
        if in_bounds(x_dst,y_dst,z_dst,
                     x,y,z,
                     cx2,cy2,cz2,
                     ax2,ay2,az2, bx,by,bz):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x, y, z,
                                   cx2, cy2, cz2,
                                   ax2, ay2, az2,
                                   bx, by, bz)
        cur_idx += abs( (cx2 + cy2 + cz2)*(ax2 + ay2 + az2)*(bx + by + bz) )

        if in_bounds(x_dst,y_dst,z_dst,
                     x+cx2,y+cy2,z+cz2,
                     ax,ay,az, bx,by,bz,
                     cx-cx2,cy-cy2,cz-cz2):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x+cx2, y+cy2, z+cz2,
                                   ax, ay, az,
                                   bx, by, bz,
                                   cx-cx2, cy-cy2, cz-cz2)
        cur_idx += abs( (ax + ay + az)*(bx + by + bz)*((cx - cx2) + (cy - cy2) + (cz - cz2)) )

        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+(ax-dax)+(cx2-dcx),
                               y+(ay-day)+(cy2-dcy),
                               z+(az-daz)+(cz2-dcz),
                               -cx2, -cy2, -cz2,
                               -(ax-ax2), -(ay-ay2), -(az-az2),
                               bx, by, bz)

    # regular case, split in all w/h/d
    if in_bounds(x_dst,y_dst,z_dst,
                 x,y,z,
                 bx2,by2,bz2,
                 cx2,cy2,cz2,
                 ax2,ay2,az2):
        return gilbert_xyz2d_r(cur_idx,x_dst,y_dst,z_dst,
                              x, y, z,
                              bx2, by2, bz2,
                              cx2, cy2, cz2,
                              ax2, ay2, az2)
    cur_idx += abs( (bx2 + by2 + bz2)*(cx2 + cy2 + cz2)*(ax2 + ay2 + az2) )

    if in_bounds(x_dst,y_dst,z_dst,
                 x+bx2, y+by2, z+bz2,
                 cx, cy, cz,
                 ax2, ay2, az2,
                 bx-bx2, by-by2, bz-bz2):
        return gilbert_xyz2d_r(cur_idx,
                              x_dst,y_dst,z_dst,
                              x+bx2, y+by2, z+bz2,
                              cx, cy, cz,
                              ax2, ay2, az2,
                              bx-bx2, by-by2, bz-bz2)
    cur_idx += abs( (cx + cy + cz)*(ax2 + ay2 + az2)*((bx - bx2) + (by - by2) + (bz - bz2)) )

    if in_bounds(x_dst,y_dst,z_dst,
                 x+(bx2-dbx)+(cx-dcx),
                 y+(by2-dby)+(cy-dcy),
                 z+(bz2-dbz)+(cz-dcz),
                 ax, ay, az,
                 -bx2, -by2, -bz2,
                 -(cx-cx2), -(cy-cy2), -(cz-cz2)):
        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+(bx2-dbx)+(cx-dcx),
                               y+(by2-dby)+(cy-dcy),
                               z+(bz2-dbz)+(cz-dcz),
                               ax, ay, az,
                               -bx2, -by2, -bz2,
                               -(cx-cx2), -(cy-cy2), -(cz-cz2))
    cur_idx += abs( (ax + ay + az)*(-bx2 - by2 - bz2)*(-(cx - cx2) - (cy - cy2) - (cz - cz2)) )

    if in_bounds(x_dst,y_dst,z_dst,
                 x+(ax-dax)+bx2+(cx-dcx),
                 y+(ay-day)+by2+(cy-dcy),
                 z+(az-daz)+bz2+(cz-dcz),
                 -cx, -cy, -cz,
                 -(ax-ax2), -(ay-ay2), -(az-az2),
                 bx-bx2, by-by2, bz-bz2):
        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+(ax-dax)+bx2+(cx-dcx),
                               y+(ay-day)+by2+(cy-dcy),
                               z+(az-daz)+bz2+(cz-dcz),
                               -cx, -cy, -cz,
                               -(ax-ax2), -(ay-ay2), -(az-az2),
                               bx-bx2, by-by2, bz-bz2)
    cur_idx += abs( (-cx - cy - cz)*(-(ax - ax2) - (ay - ay2) - (az - az2))*((bx - bx2) + (by - by2) + (bz - bz2)) )

    return gilbert_xyz2d_r(cur_idx,
                           x_dst,y_dst,z_dst,
                           x+(ax-dax)+(bx2-dbx),
                           y+(ay-day)+(by2-dby),
                           z+(az-daz)+(bz2-dbz),
                           -bx2, -by2, -bz2,
                           cx2, cy2, cz2,
                           -(ax-ax2), -(ay-ay2), -(az-az2))

def transpose_gilbert_mapping(dims, order=None):
    """
    Create mapping between linear indices and Gilbert curve indices, supporting different axis orders
    
    Parameters:
        dims: List or tuple of three dimensions, e.g. [t, h, w]
        order: Order of axes, default is [0,1,2], representing [t,h,w]
               Can be specified as [2,1,0] to represent [w,h,t] or other orders
        
    Returns:
        linear_to_hilbert: List of length dims[0]*dims[1]*dims[2], storing Gilbert curve indices corresponding to linear indices
        hilbert_to_linear: List of length dims[0]*dims[1]*dims[2], storing linear indices corresponding to Gilbert curve indices
    """
    if len(dims) != 3:
        raise ValueError("Dimensions must be three-dimensional")
    
    # If no order specified, use default [0,1,2]
    if order is None:
        order = [0, 1, 2]
    
    if len(order) != 3 or set(order) != {0, 1, 2}:
        raise ValueError("order must be a permutation of 0,1,2")
    
    # Extract original dimensions
    dims_array = np.array(dims)
    
    # Rearrange dimensions according to order
    t, h, w = dims_array[order]
    
    # Calculate total number of points
    total_points = np.prod(dims)
    
    # Initialize mapping arrays
    linear_to_hilbert = [0] * total_points
    hilbert_to_linear = [0] * total_points
    
    print(f"Computing transposed Gilbert curve mapping ({dims} axis order:{order})...")
    
    # Calculate Gilbert indices for all points
    # Create iterator for all coordinates
    coords_iter = np.ndindex(*dims)
    
    for linear_idx, coords in enumerate(coords_iter):
        # Rearrange coordinates according to order
        # For example, if order=[2,1,0], then x corresponds to coords[2], y to coords[1], z to coords[0]
        transposed_coords = [coords[order[2]], coords[order[1]], coords[order[0]]]
        
        # Calculate Gilbert curve index
        x, y, z = transposed_coords
        hilbert_idx = gilbert_xyz2d(x, y, z, w, h, t)
        
        # Set mapping
        linear_to_hilbert[linear_idx] = hilbert_idx
        hilbert_to_linear[hilbert_idx] = linear_idx
    
    print(f"Transposed Gilbert curve mapping completed, total {total_points} points")
    return linear_to_hilbert, hilbert_to_linear

def gilbert_mapping(t, h, w, transpose_order=None):
    """
    Create mapping between linear indices and Gilbert curve indices, optionally supporting transposition
    
    Parameters:
        t: Size of the first dimension
        h: Size of the second dimension
        w: Size of the third dimension
        transpose_order: Axis order, default is None (using standard order [0,1,2])
                        Can be specified as [2,1,0] or other orders
        
    Returns:
        linear_to_hilbert: List of length t*h*w, storing Gilbert curve indices corresponding to linear indices
        hilbert_to_linear: List of length t*h*w, storing linear indices corresponding to Gilbert curve indices
    """
    dims = [t, h, w]
    
    if transpose_order is None:
        # Standard Gilbert mapping, no transposition
        total_points = t * h * w
        
        # Initialize mapping arrays
        linear_to_hilbert = [0] * total_points
        hilbert_to_linear = [0] * total_points
        
        print(f"Computing Gilbert curve mapping ({w}×{h}×{t})...")
        
        # Calculate Gilbert indices for all points
        for z in range(t):
            for y in range(h):
                for x in range(w):
                    # Calculate linear index (row-major order: z*h*w + y*w + x)
                    linear_idx = z * h * w + y * w + x
                    
                    # Calculate Gilbert curve index
                    hilbert_idx = gilbert_xyz2d(x, y, z, w, h, t)
                    
                    # Set mapping
                    linear_to_hilbert[linear_idx] = hilbert_idx
                    hilbert_to_linear[hilbert_idx] = linear_idx
        
        print(f"Gilbert curve mapping completed, total {total_points} points")
    else:
        # Use transposed mapping
        linear_to_hilbert, hilbert_to_linear = transpose_gilbert_mapping(dims, transpose_order)
    
    return linear_to_hilbert, hilbert_to_linear

def block_wise_mapping(t, h, w, block_size=[4, 4, 8]):
    """
    Create block-based mapping, dividing 3D space into fixed-size blocks
    
    Parameters:
        t, h, w: The three dimensions of the overall space
        block_size: Size of each block [bt, bh, bw]
        
    Returns:
        linear_to_block_order: List storing the block number corresponding to each linear index
        block_order: List storing the starting linear index of each block
        block_neighbor_mask: List storing the 26-neighborhood (plus itself) mask for each block
    """
    bt, bh, bw = block_size
    
    # Calculate number of blocks in each dimension
    num_blocks_t = (t + bt - 1) // bt
    num_blocks_h = (h + bh - 1) // bh
    num_blocks_w = (w + bw - 1) // bw
    total_blocks = num_blocks_t * num_blocks_h * num_blocks_w
    
    # Initialize mapping arrays
    total_points = t * h * w
    linear_to_block_order = [0] * total_points
    block_order = [0] * total_blocks
    
    print(f"Computing block mapping ({t}×{h}×{w}) -> block size({bt}×{bh}×{bw})")
    
    # Assign block number to each point
    for z in range(t):
        block_z = z // bt
        for y in range(h):
            block_y = y // bh
            for x in range(w):
                block_x = x // bw
                
                # Calculate linear index
                linear_idx = z * h * w + y * w + x
                
                # Calculate block number (using row-major order)
                block_idx = (block_z * num_blocks_h * num_blocks_w + 
                           block_y * num_blocks_w + 
                           block_x)
                
                linear_to_block_order[linear_idx] = block_idx
    
    # Calculate starting linear index for each block
    for block_z in range(num_blocks_t):
        z_start = block_z * bt
        for block_y in range(num_blocks_h):
            y_start = block_y * bh
            for block_x in range(num_blocks_w):
                x_start = block_x * bw
                
                # Calculate block number
                block_idx = (block_z * num_blocks_h * num_blocks_w + 
                           block_y * num_blocks_w + 
                           block_x)
                
                # Calculate starting linear index for this block
                block_order[block_idx] = z_start * h * w + y_start * w + x_start
    
    # Create block_neighbor_mask
    block_neighbor_mask = []
    
    # Calculate neighborhood mask for each block
    for block_z in range(num_blocks_t):
        for block_y in range(num_blocks_h):
            for block_x in range(num_blocks_w):
                current_block_idx = (block_z * num_blocks_h * num_blocks_w + 
                                   block_y * num_blocks_w + 
                                   block_x)
                
                # Store block numbers of current block and its neighbors
                neighbors = []
                
                # Traverse 3x3x3 neighborhood
                for dz in [-1, 0, 1]:
                    nz = block_z + dz
                    if nz < 0 or nz >= num_blocks_t:
                        continue
                        
                    for dy in [-1, 0, 1]:
                        ny = block_y + dy
                        if ny < 0 or ny >= num_blocks_h:
                            continue
                            
                        for dx in [-1, 0, 1]:
                            nx = block_x + dx
                            if nx < 0 or nx >= num_blocks_w:
                                continue
                                
                            # Calculate neighbor block number
                            neighbor_idx = (nz * num_blocks_h * num_blocks_w + 
                                         ny * num_blocks_w + 
                                         nx)
                            
                            # Divide block number by block_size to get reordered block number
                            reordered_idx = block_order[neighbor_idx] // (bt * bh * bw)
                            neighbors.append(reordered_idx)
                
                # Sort neighbors to maintain consistency
                neighbors.sort()
                block_neighbor_mask.append(neighbors)
    
    return linear_to_block_order, block_order, block_neighbor_mask

def gilbert_block_neighbor_mapping(t, h, w, block_size=128, transpose_order=None):
    """
    Based on Gilbert curve mapping, find the neighborhood blocks for each block in 3D space
    
    Parameters:
        t, h, w: Dimensions of 3D space
        block_size: Number of tokens in each block, default 128
        transpose_order: Axis order for Gilbert curve, default None
        
    Returns:
        block_neighbors_list: List of neighborhood blocks for each block
    """
    # 1. Calculate total points and total blocks
    total_points = t * h * w
    total_blocks = (total_points + block_size - 1) // block_size
    
    print(f"Space size: {t}×{h}×{w}, total points: {total_points}, total blocks: {total_blocks}")
    
    # 2. Create block coloring map for 3D space
    block_color_map = np.zeros((w, h, t), dtype=int)
    
    # 3. Color points along the gilbert curve
    for x in range(w):
        for y in range(h):
            for z in range(t):
                # Calculate Gilbert curve index
                hilbert_idx = gilbert_xyz2d(x, y, z, w, h, t)
                
                # Calculate block number
                block_idx = hilbert_idx // block_size
                
                # Coloring: mark this position with its block
                block_color_map[x, y, z] = block_idx
    
    # 4. Initialize neighborhood sets
    block_neighbors = [set() for _ in range(total_blocks)]
    
    # 5. Traverse 3D space, update neighborhood relationships
    for x in range(w):
        for y in range(h):
            for z in range(t):
                current_block = block_color_map[x, y, z]
                
                # Add itself to its own neighborhood
                block_neighbors[current_block].add(current_block)
                
                # Check 26-neighborhood
                for dx in [-1, 0, 1]:
                    nx = x + dx
                    if nx < 0 or nx >= w:
                        continue
                        
                    for dy in [-1, 0, 1]:
                        ny = y + dy
                        if ny < 0 or ny >= h:
                            continue
                            
                        for dz in [-1, 0, 1]:
                            nz = z + dz
                            if nz < 0 or nz >= t:
                                continue
                            
                            # Skip itself (although already added)
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                                
                            # Get neighbor's block
                            neighbor_block = block_color_map[nx, ny, nz]
                            
                            # Add to current block's neighborhood
                            block_neighbors[current_block].add(neighbor_block)
    
    # 6. Convert neighborhood sets to sorted lists
    block_neighbors_list = [sorted(neighbors) for neighbors in block_neighbors]
    # convert to one-hot tensor
    block_neighbor_tensor = torch.zeros((total_blocks, total_blocks), dtype=torch.bool)
    for i, neighbors in enumerate(block_neighbors_list):
        block_neighbor_tensor[i, neighbors] = True
    print(f"Calculated neighborhood relationships for {len(block_neighbors_list)} blocks")
    # print(block_neighbors_list)
    return block_neighbor_tensor

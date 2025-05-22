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
    创建线性索引与Gilbert曲线索引之间的映射，支持不同的轴顺序
    
    参数:
        dims: 三个维度的尺寸列表或元组，例如[t, h, w]
        order: 轴的顺序，默认为[0,1,2]，表示[t,h,w]
               可以指定为[2,1,0]表示[w,h,t]等不同顺序
        
    返回:
        linear_to_hilbert: 长度为dims[0]*dims[1]*dims[2]的列表，存储线性索引对应的Gilbert曲线索引
        hilbert_to_linear: 长度为dims[0]*dims[1]*dims[2]的列表，存储Gilbert曲线索引对应的线性索引
    """
    if len(dims) != 3:
        raise ValueError("维度必须是三维的")
    
    # 如果未指定顺序，默认使用[0,1,2]
    if order is None:
        order = [0, 1, 2]
    
    if len(order) != 3 or set(order) != {0, 1, 2}:
        raise ValueError("order必须是0,1,2的一个排列")
    
    # 提取原始尺寸
    dims_array = np.array(dims)
    
    # 根据顺序重新排列尺寸
    t, h, w = dims_array[order]
    
    # 计算总点数
    total_points = np.prod(dims)
    
    # 初始化映射数组
    linear_to_hilbert = [0] * total_points
    hilbert_to_linear = [0] * total_points
    
    print(f"正在计算转置Gilbert曲线映射 ({dims} 轴顺序:{order})...")
    
    # 计算所有点的Gilbert索引
    # 创建所有坐标的迭代器
    coords_iter = np.ndindex(*dims)
    
    for linear_idx, coords in enumerate(coords_iter):
        # 根据order重新排列坐标
        # 例如，如果order=[2,1,0]，则x对应coords[2]，y对应coords[1]，z对应coords[0]
        transposed_coords = [coords[order[2]], coords[order[1]], coords[order[0]]]
        
        # 计算Gilbert曲线索引
        x, y, z = transposed_coords
        hilbert_idx = gilbert_xyz2d(x, y, z, w, h, t)
        
        # 设置映射
        linear_to_hilbert[linear_idx] = hilbert_idx
        hilbert_to_linear[hilbert_idx] = linear_idx
    
    print(f"转置Gilbert曲线映射计算完成，共 {total_points} 个点")
    return linear_to_hilbert, hilbert_to_linear

def gilbert_mapping(t, h, w, transpose_order=None):
    """
    创建线性索引与Gilbert曲线索引之间的映射，可选地支持转置
    
    参数:
        t: 第一个维度的大小
        h: 第二个维度的大小
        w: 第三个维度的大小
        transpose_order: 轴顺序，默认为None (使用标准顺序[0,1,2])
                        可以指定为[2,1,0]等不同顺序
        
    返回:
        linear_to_hilbert: 长度为t*h*w的列表，存储线性索引对应的Gilbert曲线索引
        hilbert_to_linear: 长度为t*h*w的列表，存储Gilbert曲线索引对应的线性索引
    """
    dims = [t, h, w]
    
    if transpose_order is None:
        # 标准Gilbert映射，不进行转置
        total_points = t * h * w
        
        # 初始化映射数组
        linear_to_hilbert = [0] * total_points
        hilbert_to_linear = [0] * total_points
        
        print(f"正在计算Gilbert曲线映射 ({w}×{h}×{t})...")
        
        # 计算所有点的Gilbert索引
        for z in range(t):
            for y in range(h):
                for x in range(w):
                    # 计算线性索引 (row-major order: z*h*w + y*w + x)
                    linear_idx = z * h * w + y * w + x
                    
                    # 计算Gilbert曲线索引
                    hilbert_idx = gilbert_xyz2d(x, y, z, w, h, t)
                    
                    # 设置映射
                    linear_to_hilbert[linear_idx] = hilbert_idx
                    hilbert_to_linear[hilbert_idx] = linear_idx
        
        print(f"Gilbert曲线映射计算完成，共 {total_points} 个点")
    else:
        # 使用转置映射
        linear_to_hilbert, hilbert_to_linear = transpose_gilbert_mapping(dims, transpose_order)
    
    return linear_to_hilbert, hilbert_to_linear

def block_wise_mapping(t, h, w, block_size=[4, 4, 8]):
    """
    创建基于块的映射，将3D空间划分为固定大小的块
    
    参数:
        t, h, w: 整体空间的三个维度大小
        block_size: 每个块的大小 [bt, bh, bw]
        
    返回:
        linear_to_block_order: 列表，存储每个线性索引对应的块序号
        block_order: 列表，存储每个块的起始线性索引
        block_neighbor_mask: 列表，存储每个块的26邻域(加上自己)的mask
    """
    bt, bh, bw = block_size
    
    # 计算在每个维度上的块数
    num_blocks_t = (t + bt - 1) // bt
    num_blocks_h = (h + bh - 1) // bh
    num_blocks_w = (w + bw - 1) // bw
    total_blocks = num_blocks_t * num_blocks_h * num_blocks_w
    
    # 初始化映射数组
    total_points = t * h * w
    linear_to_block_order = [0] * total_points
    block_order = [0] * total_blocks
    
    print(f"正在计算块映射 ({t}×{h}×{w}) -> 块大小({bt}×{bh}×{bw})")
    
    # 为每个点分配块序号
    for z in range(t):
        block_z = z // bt
        for y in range(h):
            block_y = y // bh
            for x in range(w):
                block_x = x // bw
                
                # 计算线性索引
                linear_idx = z * h * w + y * w + x
                
                # 计算块序号 (使用行优先顺序)
                block_idx = (block_z * num_blocks_h * num_blocks_w + 
                           block_y * num_blocks_w + 
                           block_x)
                
                linear_to_block_order[linear_idx] = block_idx
    
    # 计算每个块的起始线性索引
    for block_z in range(num_blocks_t):
        z_start = block_z * bt
        for block_y in range(num_blocks_h):
            y_start = block_y * bh
            for block_x in range(num_blocks_w):
                x_start = block_x * bw
                
                # 计算块序号
                block_idx = (block_z * num_blocks_h * num_blocks_w + 
                           block_y * num_blocks_w + 
                           block_x)
                
                # 计算该块的起始线性索引
                block_order[block_idx] = z_start * h * w + y_start * w + x_start
    
    # 创建block_neighbor_mask
    block_neighbor_mask = []
    
    # 对每个块计算其邻域mask
    for block_z in range(num_blocks_t):
        for block_y in range(num_blocks_h):
            for block_x in range(num_blocks_w):
                current_block_idx = (block_z * num_blocks_h * num_blocks_w + 
                                   block_y * num_blocks_w + 
                                   block_x)
                
                # 存储当前块及其邻居的块序号
                neighbors = []
                
                # 遍历3x3x3邻域
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
                                
                            # 计算邻居块的序号
                            neighbor_idx = (nz * num_blocks_h * num_blocks_w + 
                                         ny * num_blocks_w + 
                                         nx)
                            
                            # 将块序号除以block_size，得到重排序后的块序号
                            reordered_idx = block_order[neighbor_idx] // (bt * bh * bw)
                            neighbors.append(reordered_idx)
                
                # 对neighbors进行排序以保持一致性
                neighbors.sort()
                block_neighbor_mask.append(neighbors)
    
    return linear_to_block_order, block_order, block_neighbor_mask

def gilbert_block_neighbor_mapping(t, h, w, block_size=128, transpose_order=None):
    """
    基于Gilbert曲线映射找出3D空间中每个block的邻域blocks
    
    参数:
        t, h, w: 3D空间的尺寸
        block_size: 每个block包含的token数量，默认128
        transpose_order: Gilbert曲线的轴顺序，默认为None
        
    返回:
        block_neighbors_list: 每个block的邻域block列表
    """
    # 1. 计算总点数和总block数
    total_points = t * h * w
    total_blocks = (total_points + block_size - 1) // block_size
    
    print(f"空间大小: {t}×{h}×{w}, 总点数: {total_points}, 总block数: {total_blocks}")
    
    # 2. 创建3D空间的block染色图
    block_color_map = np.zeros((w, h, t), dtype=int)
    
    # 3. 将gilbert曲线上的点染色
    for x in range(w):
        for y in range(h):
            for z in range(t):
                # 计算Gilbert曲线索引
                hilbert_idx = gilbert_xyz2d(x, y, z, w, h, t)
                
                # 计算block序号
                block_idx = hilbert_idx // block_size
                
                # 染色：将该位置标记为所属的block
                block_color_map[x, y, z] = block_idx
    
    # 4. 初始化邻域集合
    block_neighbors = [set() for _ in range(total_blocks)]
    
    # 5. 遍历3D空间，更新邻域关系
    for x in range(w):
        for y in range(h):
            for z in range(t):
                current_block = block_color_map[x, y, z]
                
                # 将自己加入到自己的邻域中
                block_neighbors[current_block].add(current_block)
                
                # 检查26邻域
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
                            
                            # 跳过自身(虽然已经加入过了)
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                                
                            # 获取邻居的block
                            neighbor_block = block_color_map[nx, ny, nz]
                            
                            # 添加到当前block的邻域中
                            block_neighbors[current_block].add(neighbor_block)
    
    # 6. 将邻域集合转换为排序列表
    block_neighbors_list = [sorted(neighbors) for neighbors in block_neighbors]
    # convert to one-hot tensor
    block_neighbor_tensor = torch.zeros((total_blocks, total_blocks), dtype=torch.bool)
    for i, neighbors in enumerate(block_neighbors_list):
        block_neighbor_tensor[i, neighbors] = True
    print(f"已计算 {len(block_neighbors_list)} 个block的邻域关系")
    # print(block_neighbors_list)
    return block_neighbor_tensor

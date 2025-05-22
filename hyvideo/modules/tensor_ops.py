
def reshape_to_blocks(x, h, w, t, block_size):
    """Reshape tensor into blocks for hierarchical attention.
    
    Args:
        x: Input tensor of shape [B, (t h w), H, D]
        h, w, t: Original dimensions
        block_size: List of [block_h, block_w, block_t]
    
    Returns:
        Tensor of shape [B, (bt bh bw), (t_local h_local w_local), H, D]
        and a dict containing metadata for reshape back
    """
    B, L, num_heads, D = x.shape
    blocks_h = h // block_size[0]
    blocks_w = w // block_size[1] 
    blocks_t = t // block_size[2]
    
    # Store metadata for reshaping back
    metadata = {
        'original_shape': x.shape,
        'h': h, 'w': w, 't': t,
        'blocks_h': blocks_h, 
        'blocks_w': blocks_w,
        'blocks_t': blocks_t,
        'block_size': block_size
    }
    
    # First reshape to [B, t, h, w, H, D]
    x = x.view(B, t, h, w, num_heads, D)
    
    # Reshape spatial dimensions into blocks
    x = x.view(B, blocks_t, block_size[2],  # time
                  blocks_h, block_size[0],   # height
                  blocks_w, block_size[1],   # width
                  num_heads, D)
    
    # Permute and reshape to get blocks
    x = x.permute(0, 1, 3, 5,                    # B, bt, bh, bw
                     2, 4, 6,                     # t_local, h_local, w_local
                     7, 8)                        # H, D
    
    num_blocks = blocks_t * blocks_h * blocks_w
    tokens_per_block = block_size[0] * block_size[1] * block_size[2]
    
    x_blocks = x.reshape(B, num_blocks, tokens_per_block, num_heads, D)
    
    return x_blocks, metadata

def reshape_from_blocks(x_blocks, metadata):
    """Reshape blocked tensor back to original shape.
    
    Args:
        x_blocks: Tensor of shape [B, (bt bh bw), (t_local h_local w_local), H, D]
        metadata: Dict containing reshape metadata
    
    Returns:
        Tensor of original shape [B, (t h w), H, D]
    """
    B, num_blocks, tokens_per_block, num_heads, D = x_blocks.shape
    block_size = metadata['block_size']
    h, w, t = metadata['h'], metadata['w'], metadata['t']
    blocks_h = metadata['blocks_h']
    blocks_w = metadata['blocks_w']
    blocks_t = metadata['blocks_t']
    
    # Reshape to blocked format
    x = x_blocks.view(B, blocks_t, blocks_h, blocks_w,  # B, bt, bh, bw
                        block_size[2],                   # t_local
                        block_size[0],                   # h_local
                        block_size[1],                   # w_local
                        num_heads, D)
    
    # Permute back to original order
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7, 8)
    
    # Reshape back to original spatial dimensions
    x = x.reshape(B, t, h, w, num_heads, D)
    
    # Final reshape to original shape
    x = x.view(B, t * h * w, num_heads, D)
    
    return x

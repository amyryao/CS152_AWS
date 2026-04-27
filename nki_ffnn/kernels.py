import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

from utils import BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from matmul_kernels import nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_

@nki.jit
def nki_transpose(in_tensor):
    """NKI kernel to transpose a 2D tensor.

    Args:
        in_tensor: an input tensor of shape [#rows, #cols]

    Returns:
        out_tensor: an output (transposed) tensor of shape [#cols, #rows]
    """
    i_rows, i_cols = in_tensor.shape
    o_rows, o_cols = i_cols, i_rows

    out_tensor = nl.ndarray((o_rows, o_cols), dtype=in_tensor.dtype, buffer=nl.hbm)

    # max partition dimension.. left most dimensino = partition dimensino
    TILE = nl.tile_size.pmax # is it just 128?
    # setting the tile
    tile = nl.mgrid[0:TILE, 0:TILE]
    # sequence of nums for parallel loop iterators
    # Use affine_range to loop over tiles
    for m in nl.affine_range(i_rows// TILE):
      for n in nl.affine_range(i_cols// TILE):
        # load and transpose of tile
        transpose = nl.load_transpose2d(in_tensor[m * TILE + tile.p, n*TILE + tile.x])
        # store it back to out_tensor
        nl.store(out_tensor[n * TILE + tile.p, m * TILE + tile.x], value=transpose)

    return out_tensor

@nki.jit
def nki_bias_add_act(A, b, act='relu'):
    """NKI kernel to add a bias vector to each row of a 2D tensor, and apply activation.

    Args:
        A: an input tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
        b: a bias vector of shape [1, HIDDEN_SIZE]
        act: an activation function to apply (e.g., 'relu', 'softmax')
    Returns:
        result: the resulting output tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
    """
    # Gather input shapes
    BATCH_SIZE, HIDDEN_SIZE = A.shape
    _, HIDDEN_SIZE_ = b.shape
    assert HIDDEN_SIZE == HIDDEN_SIZE_, "A and b must have the same HIDDEN_SIZE"

    # Create an output tensor
    result = nl.ndarray((BATCH_SIZE, HIDDEN_SIZE), dtype=A.dtype, buffer=nl.hbm)

    # folloiwng matmul kernels convention
    TILE_M = nl.tile_size.pmax
    idx = nl.mgrid[0:TILE_M, 0:HIDDEN_SIZE] 
    r = idx.p
    c = idx.x

    for i in nl.affine_range(BATCH_SIZE// TILE_M):
      A_tile = nl.load(A[i * TILE_M + r, c])
      b_tile = nl.load(b[0, c])

      z = A_tile + b_tile

      # if activation is relu: 
      # return np.maximum(0, x) 
      if act == 'relu':
          out = nl.maximum(z, 0)

      # if softmax:
      # e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
      # then e_x / np.sum(e_x, axis=1, keepdims=True)
      elif act == 'softmax':
          row_max = nl.max(z, axis=1)
          shift = z - row_max # our actual exp shift
          exp_z = nl.exp(shift)
          row_sum = nl.sum(exp_z, axis=1) 
          out = exp_z / row_sum

      else:
          out = z

      nl.store(result[i * TILE_M + r, c], value=out)

    return result

@nki.jit
def nki_forward(
    X,
    W1,
    b1,
    W2,
    b2,
    matmul_kernel='tiled'
):
  """NKI kernel to compute the forward pass of the feedforward neural network with 1 hidden layer.

  Args:
      X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
      W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
      b1: the bias vector of shape [HIDDEN_SIZE]
      W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
      b2: the bias vector of shape [OUTPUT_SIZE]
  Returns:
      probs: the resulting probability output tensor of shape [BATCH_SIZE, OUTPUT_SIZE]
  
  Option:
      matmul_kernel: the matrix multiplication kernel to use 
        - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'
  """
  if matmul_kernel == 'tiled':
    nki_matmul = nki_matmul_tiled_
  elif matmul_kernel == 'hoist_load':
    nki_matmul = nki_matmul_hoist_load_
  elif matmul_kernel == 'block_free_dimension':
    nki_matmul = nki_matmul_block_free_dimension_
  elif matmul_kernel == 'fully_optimized':
    nki_matmul = nki_matmul_fully_optimized_
  else:
    raise ValueError(f"Unsupported matmul kernel: {matmul_kernel}")

  # Layer 1
  #  self.z1 = X @ self.W1 + self.b1
  #  self.a1 = relu(self.z1)
  XT = nki_transpose(X) # transpose X for matmul (lhsT should alr be tarnsposed)
  z1 = nki_matmul(XT, W1)
  a1 = nki_bias_add_act(z1, b1, act='relu')

  # Layer 2 (output)
  #  self.z2 = self.a1 @ self.W2 + self.b2
  # self.a2 = softmax(self.z2)
  # essentially: take result a1 @ W2 + b2, then call softmax
  a1T = nki_transpose(a1) 
  z2 = nki_matmul(a1T, W2)
  probs = nki_bias_add_act(z2, b2, act='softmax')

  return probs


@nki.jit
def nki_predict(
    X,
    W1,
    b1,
    W2,
    b2,
    matmul_kernel='tiled'
):
  """NKI kernel run forward pass and predict the classes of the input tensor.

  Args:
      X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
      W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
      b1: the bias vector of shape [HIDDEN_SIZE]
      W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
      b2: the bias vector of shape [OUTPUT_SIZE]
  Returns:
      predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
  
  Option:
      matmul_kernel: the matrix multiplication kernel to use 
        - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'

  Returns:
      predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
  """
  # probs = self.forward(X) 
  probs = nki_forward(X, W1, b1, W2, b2, matmul_kernel)
  BATCH_SIZE, OUTPUT_SIZE = probs.shape
  predictions = nl.ndarray((BATCH_SIZE,), dtype=np.int32, buffer=nl.hbm)

  # return np.argmax(probs, axis=1).astype(np.int32)- can't use actual argmax
  # aka: return the max of of all of these rows
  TILE_M = nl.tile_size.pmax
  idx = nl.mgrid[0:TILE_M, 0:OUTPUT_SIZE]
  r = idx.p
  c = idx.x

  idx8 = nl.mgrid[0:TILE_M, 0:8]
  r8 = idx8.p

  for i in nl.affine_range(BATCH_SIZE // TILE_M):
      # loads a block of rows and output columns
      probs_tile = nl.load(probs[i * TILE_M + r, c])
      # finds top 8 values per row: (Find the 8 largest values in each partition of the source tile.)
      vals = nisa.max8(src=probs_tile)
      # gets index of these vals..  (Find indices of the 8 given vals in each partition of the data tensor.)
      inds = nisa.nc_find_index8(data=probs_tile, vals=vals)
     
      # store predictions- first column. of info (largest)
      pred = nl.copy(inds[r8, 0], dtype=np.int32)
      nl.store(predictions[i * TILE_M + r8], value=pred)

  return predictions
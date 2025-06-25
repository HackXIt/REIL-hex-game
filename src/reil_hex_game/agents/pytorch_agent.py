from pathlib import Path
import random
import numpy as np
import torch, torch.nn.functional as F

# ---------- load weights once ----------------------------------------
_w = None
def _load():
    global _w
    if _w is None:
        p = Path(__file__).with_name("weights.npz")
        _w = dict(np.load(p))
    return _w

# ---------- preprocess board  → (3,H,W) float32 -----------------------
def _obs_tensor(board):
    b = np.asarray(board, dtype=np.float32)
    p1  = (b ==  1).astype(np.float32)
    p_1 = (b == -1).astype(np.float32)
    turn = np.full_like(b, 1.0 if np.count_nonzero(b==1)==np.count_nonzero(b==-1) else 0.0)
    return np.stack([p1, p_1, turn], axis=0)               # (3,H,W)

# ---------- minimal forward pass -------------------------------------
def _policy_logits(obs: np.ndarray) -> np.ndarray:
    w = _load()
    # torch.from_file(Path(__file__).with_name("weights.npz"), "r") ???
    # tensors without gradients
    conv_w = torch.from_numpy(w["conv_w"])
    conv_b = torch.from_numpy(w["conv_b"])
    hid_w  = torch.from_numpy(w["hid_w"])
    hid_b  = torch.from_numpy(w["hid_b"])
    fc_w   = torch.from_numpy(w["fc_w"])
    fc_b   = torch.from_numpy(w["fc_b"])

    x = torch.tensor(obs)[None]                        # (1,3,7,7)
    x = F.conv2d(x, conv_w, conv_b, padding=1)
    x = torch.tanh(x)
    x = x.mean(dim=(2, 3))                             # (1,32)

    x = torch.tanh(F.linear(x, hid_w, hid_b))          # (1,64)
    logits = F.linear(x, fc_w, fc_b).squeeze(0)        # (49,)
    return logits.numpy()

# ---------- public API ------------------------------------------------
def pytorch_agent(board, action_set):
    """
    Parameters
    ----------
    board : list[list[int]]
        Current 7x7 board with values in {0, 1, -1}.
    action_set : list[(int,int)]
        All legal moves (row, col) for the current player.

    Returns
    -------
    (row, col)  - a *legal* move.
    """
    # 0. degenerate case: only one legal – no need to run the net
    if len(action_set) == 1:
        return action_set[0]

    size   = len(board)
    idx_of = lambda rc: rc[0] * size + rc[1]

    # 1. forward pass
    logits = _policy_logits(_obs_tensor(board))

    # 2. mask illegal moves  →   set their logits to  -inf
    mask = np.full_like(logits, -np.inf)
    for rc in action_set:
        mask[idx_of(rc)] = logits[idx_of(rc)]

    # 3. numerical safety
    if np.all(~np.isfinite(mask)):
        # should never happen, but fallback to *uniform* choice
        return random.choice(action_set)

    # 4. greedy choice among *only* legal moves
    best = int(np.nanargmax(mask))
    return divmod(best, size)          # (row, col)
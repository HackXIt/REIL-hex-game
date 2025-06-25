"""
Facade for Hex AI submission.
Loads a tiny CNN head from weights.npz (NumPy only) and
implements agent(board, action_set) → best legal move.
"""

from pathlib import Path
import math, random, numpy as np

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
    w   = _load()                                      # dict with 6 arrays
    k,w0 = w["conv_w"], w["conv_b"]
    pad  = np.pad(obs, ((0,0),(1,1),(1,1)), mode="constant")
    Cout, _, _, _ = k.shape
    H, W = obs.shape[1:]
    out  = np.zeros((Cout, H, W), dtype=np.float32)

    # hand-rolled conv
    for i in range(3):
        for j in range(3):
            out += (k[:, :, i, j, None, None] *
                    pad[:, i:i+H, j:j+W]).sum(axis=1)
    out += w0[:, None, None]
    out  = np.tanh(out)

    feat = out.mean(axis=(1, 2))                       # (32,)

    # hidden layer 32 → 64
    feat = np.tanh(w["hid_w"] @ feat + w["hid_b"])     # (64,)

    # head 64 → 49
    logits = w["fc_w"] @ feat + w["fc_b"]              # (49,)
    return logits

# ---------- public API ------------------------------------------------
def numpy_agent(board, action_set):
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

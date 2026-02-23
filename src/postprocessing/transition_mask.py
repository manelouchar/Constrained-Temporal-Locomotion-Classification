import numpy as np    


STATES    = ['LW', 'SA', 'SD', 'RA', 'RD']
STATE2IDX = {s: i for i, s in enumerate(STATES)}

_ALLOWED = np.array([
    [1, 1, 1, 1, 1],   # from LW
    [1, 1, 0, 0, 0],   # from SA
    [1, 0, 1, 0, 0],   # from SD
    [1, 0, 0, 1, 0],   # from RA
    [1, 0, 0, 0, 1],   # from RD
], dtype=bool)

LOG_TRANS = np.where(_ALLOWED, 0.0, -np.inf)

def is_allowed(src, dst):
    return bool(_ALLOWED[src, dst])

def print_mask():
    print("\nTransition Mask (✓ allowed  ✗ forbidden)")
    print("        " + "  ".join(f"{s:>4}" for s in STATES))
    for i, row in enumerate(_ALLOWED):
        cells = "  ".join("  ✓ " if v else "  ✗ " for v in row)
        print(f"  {STATES[i]:>4} | {cells}")

print_mask()
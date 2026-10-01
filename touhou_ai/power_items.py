"""Pinned TH10 item pickup switch (0x41b3c1 / table 0x41b8a0).

Types 1/10 add one raw unit; 4/11 add twenty. Twenty raw units = 1 Power.
The grid encodes nominal collectible amount, independent of current player cap.
"""
POWER_RAW_BY_TYPE = {1: 1, 4: 20, 10: 1, 11: 20}
POWER_GRID_SCALE = 100  # Value 1 represents 5 Power; linear even above 1.
MAX_ITEMS = 2198
POWER_GRID_MAX = MAX_ITEMS * 20 / POWER_GRID_SCALE


def power_item_raw(item):
    kind = item.get('type')
    if type(kind) is not int or kind not in range(1, 12):
        raise ValueError('unknown item type for P separation')
    return POWER_RAW_BY_TYPE.get(kind, 0)

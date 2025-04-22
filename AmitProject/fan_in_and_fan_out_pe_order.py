import numpy as np
import math

def generate_pe_order(n_echo, Ny, order_type='fan-in'):
    """
    Generate phase encoding order matrix of shape (n_echo, n_ex)
    based on either 'fan-in' or 'fan-out' phase encoding order.

    Parameters:
    - n_echo: number of echoes per shot
    - Ny: number of phase encoding lines
    - order_type: 'fan-in' or 'fan-out'

    Ny must be divisible by n_echo (Ny % n_echo = 0)

    Return:
    - pe_order: numpy array of shape (n_echo, n_ex)
    """
    n_ex = math.floor(Ny /n_echo)

    if order_type == 'fan-in':
        # Generate full fan-in order: [-Ny/2, Ny/2-1, -Ny/2+1, Ny/2-2, ...]
        order = []
        left = -Ny // 2
        right = Ny // 2 - 1
        for i in range(Ny // 2):
            order.append(left + i)
            order.append(right - i)
        if Ny % 2 != 0:
            order.append(0)
        order = np.array(order)#[:Ny])
        pe_order = order.reshape((n_ex, n_echo)).T  # shape (n_echo, n_ex)

    elif order_type == 'fan-out':
        # Generate relative fan-out pattern: [0, 1, -1, 2, -2, ...]
        fanout = []
        for i in range(n_echo // 2):
            fanout.append(i)
            fanout.append(-i - 1)
        if n_echo % 2 != 0:
            fanout.append(n_echo // 2)
        fanout = np.array(fanout)

        pe_order = np.zeros((n_echo, n_ex), dtype=int)
        for k_ex in range(n_ex):
            block_start = k_ex * (n_echo // 2)
            for k_echo in range(n_echo):
                if k_echo % 2 == 0:
                    pe_order[k_echo, k_ex] = block_start + fanout[k_echo]
                else:
                    pe_order[k_echo, k_ex] = -block_start + fanout[k_echo]
    else:
        raise ValueError("order_type must be either 'fan-in' or 'fan-out'")

    return pe_order

Ny = 64
n_echo = 8
#     Ny must be divisible by n_echo (Ny % n_echo = 0)
pe_order_fanin = generate_pe_order(n_echo, Ny, order_type='fan-in')
pe_order_fanout = generate_pe_order(n_echo, Ny, order_type='fan-out')
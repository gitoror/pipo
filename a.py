SIZE_X, SIZE_Y = 2, 3
LATTICE_Q = 9


def invIDXY(i):
    return i//(SIZE_Y*LATTICE_Q), (i//LATTICE_Q) % SIZE_Y, i % LATTICE_Q


def IDXY(x, y, q):
    return x*SIZE_Y*LATTICE_Q + y*LATTICE_Q + q


print(invIDXY(4))
x, y, q = invIDXY(4)
print(IDXY(x, y, q))

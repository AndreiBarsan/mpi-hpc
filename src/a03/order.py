
import numpy as np


def build_S(rows, cols):
    S = np.zeros((rows * cols, rows * cols))
    for i in range(S.shape[0]):
        S[i, i] = 5
        if i + 1 < S.shape[1]:
            S[i, i + 1] = 3
        if i - 1 >= 0:
            S[i, i - 1] = 3
        # IMPORTANT: This is not 100% correct!

        if i + 5 < S.shape[1]:
            S[i, i + 5] = 1
        if i - 5 >= 0:
            S[i, i - 5] = 1
    return S

def build_S_4col(rows, cols):
    S = np.zeros((rows * cols, rows * cols))
    for row in range(rows):
        for col in range(cols):
            i = row * cols + col

            S[i, i] = 5
            if i + 1 < S.shape[1] and col < cols - 1:
                S[i, i + 1] = 3
            if i - 1 >= 0 and col > 0:
                S[i, i - 1] = 3

            if i + 4 < S.shape[1] and col > 0:
                S[i, i + 4] = 1
            if i + 5 < S.shape[1]:
                S[i, i + 5] = 1
            if i + 6 < S.shape[1] and col < cols - 1:
                S[i, i + 6] = 1

            if i - 4 >= 0 and col < cols - 1:
                S[i, i - 4] = 1
            if i - 5 >= 0:
                S[i, i - 5] = 1
            if i - 6 >= 0 and col > 0:
                S[i, i - 6] = 1
    return S


def build_perm_2color(rows, cols):
    red_idx = []
    black_idx = []
    ri = 0
    bi = 0
    for row in range(rows):
        for col in range(cols):
            i = row * cols + col
            if row % 2 == col % 2: # RED
                red_idx.append((i, ri))
                ri = ri + 1
            else: # BLACK
                black_idx.append((i, bi))
                bi = bi + 1
    red_idx = np.array(red_idx)
    black_idx = np.array(black_idx)
    black_idx[:, 1] += red_idx.shape[0]

    print("Red count: {}".format(len(red_idx)))
    print("Black count: {}".format(len(black_idx)))
    idx = np.vstack((red_idx, black_idx))
    P = np.zeros((rows * cols, rows * cols))
    for row in idx:
        P[row[1], row[0]] = 1
    return P


def build_perm_4color(rows, cols):
    red_idx = []
    black_idx = []
    green_idx = []
    yellow_idx = []
    ri = 0
    bi = 0
    gi = 0
    yi = 0
    """
    Use the second choice, adapted
    R G R
    Y B Y
    G R G
    """
    for row in range(rows):
        for col in range(cols):
            i = row * cols + col
            if row % 4 == 0:
                if col % 2 == 0:
                    red_idx.append((i, ri))
                    ri = ri + 1
                else:
                    green_idx.append((i, gi))
                    gi += 1
            elif row % 3 == 1:
                if col % 2 == 0:
                    yellow_idx.append((i, yi))
                    yi += 1
                else:
                    black_idx.append((i, bi))
                    bi = bi + 1
            elif row % 3 == 2:
                if col % 2 == 0:
                    green_idx.append((i, gi))
                    gi += 1
                else:
                    red_idx.append((i, ri))
                    ri = ri + 1
            else:
                if col % 2 == 0:
                    black_idx.append((i, bi))
                    bi = bi + 1
                else:
                    yellow_idx.append((i, yi))
                    yi += 1
            # if row % 2 == 0 and col % 2 == 0: # RED
            #     red_idx.append((i, ri))
            #     ri = ri + 1
            # elif row % 2 == 0 and col % 2 == 1:
            #     green_idx.append((i, gi))
            #     gi += 1
            # elif row % 2 == 1 and col % 2 == 0: # GREEN
            #     yellow_idx.append((i, yi))
            #     yi += 1
            # else:   # YELLOW
            #     black_idx.append((i, bi))
            #     bi = bi + 1
    red_idx = np.array(red_idx)
    black_idx = np.array(black_idx)
    green_idx = np.array(green_idx)
    yellow_idx = np.array(yellow_idx)
    # black_idx[:, 1] += red_idx.shape[0]
    # green_idx[:, 1] += red_idx.shape[0] + black_idx.shape[0]
    # yellow_idx[:, 1] += green_idx.shape[0] + red_idx.shape[0] + black_idx.shape[0]

    print("Red count: {}".format(len(red_idx)))
    print("Black count: {}".format(len(black_idx)))
    print("Green count: {}".format(len(green_idx)))
    print("Yellow count: {}".format(len(yellow_idx)))
    idx = np.vstack((red_idx, black_idx, green_idx, yellow_idx))
    # print(idx)
    # P = np.zeros((rows * cols, rows * cols))
    # for row in idx:
    #     P[row[1], row[0]] = 1

    # The indices in the first column describe a permutation; we can pass this to Eigen!
    P_id = np.eye(rows * cols)
    P_id = P_id[idx[:, 0]]
    # assert np.allclose(P_id, P)

    return P_id




def main():
    np.set_printoptions(linewidth=120)
    # n
    rows = 5
    # m
    cols = 5
    S = build_S_4col(rows, cols)
    # P = build_perm_2color(rows, cols)
    P = build_perm_4color(rows, cols)

    np.set_printoptions(formatter={ 'float': lambda x: " " if x == 0 else "x" })
    print("Permutation matrix:")
    print(P)
    print("Original:")
    print(S)
    print()
    print("Permuted:")
    print(np.dot(P, np.dot(S, P.transpose())))
    np.set_printoptions(formatter=None)



if __name__ == '__main__':
    main()

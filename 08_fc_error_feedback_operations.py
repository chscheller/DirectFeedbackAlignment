
c_a = 1
c_m = 2


def op_dfa(l: list):
    n = len(l)
    total = 0
    for i in range(1, n):
        total += l[i-1] * l[i] * c_m
        total += l[n-1] * l[i] * c_m
        total += l[i] * c_m
        total += 2 * l[i] * c_m + l[i] * c_a
    return total


def op_bp(l: list):
    n = len(l)
    total = 0
    for i in range(1, n):
        total += 2 * l[i-1] * l[i] * c_m
        total += l[i] * c_m
        total += 2 * l[i] * c_m + l[i] * c_a
    return total


if __name__ == '__main__':

    l = [784, 500, 500, 500, 500, 500, 10]
    # l = [32*32*3, 1000, 1000, 1000, 10]

    total_dfa = op_dfa(l)
    total_bp = op_bp(l)

    print('l = {}'.format(l))
    print('op_dfa(l) = {} cycles'.format(total_dfa))
    print('op_bp(l) = {} cycles'.format(total_bp))
    print('op_dfa(l) : op_bp(l) = {}'.format(total_dfa / total_bp))


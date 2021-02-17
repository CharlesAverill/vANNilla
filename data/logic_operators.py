def features():
    return [[0, 0],
            [0, 1],
            [1, 0],
            [1, 1]]


def AND():
    labels = [0, 0, 0, 1]
    return features(), labels


def OR():
    labels = [0, 1, 1, 1]
    return features(), labels


def XOR():
    labels = [0, 1, 1, 0]
    return features(), labels

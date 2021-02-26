from vANNilla import Random


def x():
    rng = Random()
    out = []
    for feature in range(100):
        if rng.next(0, 1) < 0.5:
            out.append(feature + rng.next(0, 1))
        else:
            out.append(feature)
    return out


def linear_regression():
    # f(x) = 7x + 9 + random(-num_x / 2, num_x / 2)
    rng = Random()
    features = [[feature] for feature in x()]
    rng_bound = len(features) / 2
    labels = [
        (7 * feature[0]) + 9 + rng.next(-rng_bound, rng_bound)
        for feature in features
    ]

    max_feature = features[-1][0]
    max_label = max(labels)
    features = [[feature[0] / max_feature] for feature in features]
    labels = [label / max_label for label in labels]

    return features, labels


def quadratic_regression():
    # f(x) = 3(x - 50)^2 + 2 + random(-num_x / 3, num_x / 3)
    rng = Random()
    features = [[feature] for feature in x()]
    rng_bound = len(features) * 1.5
    labels = [
        3 * ((feature[0] - 50) ** 2) + 2 + rng.next(-rng_bound, rng_bound)
        for feature in features
    ]

    max_feature = features[-1][0]
    max_label = max(labels)
    features = [[feature[0] / max_feature] for feature in features]
    labels = [label / max_label for label in labels]

    return features, labels

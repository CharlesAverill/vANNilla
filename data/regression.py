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


def linear_regression(apply_scaling=True):
    # f(x) = 7x + 9 + random(-num_x / 3, num_x / 3)
    rng = Random()
    features = [[feature] for feature in x()]
    rng_bound = len(features) / 3
    labels = [(7 * feature[0]) + 9 + rng.next(-rng_bound, rng_bound) for feature in features]

    if apply_scaling:
        max_feature = features[-1][0]
        max_label = max(labels)
        features = [[feature[0] / max_feature] for feature in features]
        labels = [label / max_label for label in labels]

    return features, labels

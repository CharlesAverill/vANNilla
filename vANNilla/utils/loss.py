from vANNilla.utils.tensor import Tensor


def mse(activated, labels, activation_ddx):
    """
    :param activated: Activated predictions from epoch
    :param labels: Labels for input data
    :param activation_ddx: Derivative of activation function
    :return: List of partial slopes and the MSE value
    """
    error = [
        activated_n - labels_n
        for activated_n, labels_n in zip(activated, labels)
    ]
    d_predictions = activation_ddx(activated)
    partial_slope = [error[i] * d_predictions[i] for i in range(len(error))]

    return Tensor(partial_slope), sum(error) / len(error)


LOSS_FUNCTIONS = {"mse": mse}

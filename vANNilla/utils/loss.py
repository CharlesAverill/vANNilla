def mse(predictions, labels, activation, activation_ddx):
    """
    :param predictions: Predictions from epoch
    :param labels: Labels for input data
    :param activation: Activation function
    :param activation_ddx: Derivative of activation function
    :return:
    """
    activated = [activation(xw) for xw in predictions]
    error = [activated_n - labels_n for activated_n, labels_n in zip(activated, labels)]
    d_predictions = [activation_ddx(a) for a in activated]
    partial_slope = [error[i] * d_predictions[i] for i in range(len(error))]

    return partial_slope, sum(error) / len(error)


LOSS_FUNCTIONS = {"mse": mse}

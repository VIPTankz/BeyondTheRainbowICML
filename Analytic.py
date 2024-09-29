import matplotlib.pyplot as plt
import numpy as np


def moving_average(data, window_size):
    """
    Calculate the moving average of the data with the specified window size.

    :param data: List of numerical values.
    :param window_size: Size of the moving window.
    :return: List of smoothed values.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def average_of_last_values(data_list, n):
    """
    Calculate and print the average of the last n values in a list.

    :param data_list: List of numerical values.
    :param n: Number of last values to consider for the average.
    """
    if not data_list:
        print("The list is empty.")
        return

    if n > len(data_list):
        print("The number of values to consider is greater than the length of the list.")
        return

    last_values = data_list[-n:]
    average = sum(last_values) / n
    #print(f"The average of the last {n} values is: {average}")


def plot_smoothed_data(data, window_size):
    """
    Plot the original data and its smoothed version.

    :param data: List of numerical values.
    :param window_size: Size of the moving window for smoothing.
    """
    smoothed_data = moving_average(data, window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Original Data', alpha=0.5)
    plt.plot(range(window_size - 1, len(data)), smoothed_data, label=f'Smoothed Data (window size={window_size})',
             color='red')
    plt.title('Original Data and Smoothed Data')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


class Analytics:
    def __init__(self, agent_name, testing):
        self.agent_name = agent_name
        self.losses = []
        self.churns = []
        self.churn_difs = []
        self.grad_mags = []
        self.churn_actions = []
        self.qvals = []
        self.testing = testing

    def add_qvals(self, qvals):
        self.qvals.append(np.array(qvals))
        if len(self.qvals) % 250 == 0:

            if not self.testing:
                np.save(self.agent_name + "Qvals.npy", np.array(self.qvals))

    def add_loss(self, loss):
        self.losses.append(loss)

        if len(self.losses) % 250 == 0:
            if not self.testing:
                np.save(self.agent_name + "Losses.npy", np.array(self.losses))

    def add_grad_mag(self, mag):
        self.grad_mags.append(mag)

        if len(self.grad_mags) % 250 == 0:
            if not self.testing:
                np.save(self.agent_name + "GradMags.npy", np.array(self.grad_mags))

    def add_churn(self, churn):
        self.churns.append(churn)
        if len(self.churns) % 10 == 0:
            if not self.testing:
                np.save(self.agent_name + "churns.npy", np.array(self.churns))

    def add_churn_dif(self, churn_dif):
        self.churn_difs.append(np.array(churn_dif))
        if len(self.churn_difs) % 10 == 0:
            if not self.testing:
                np.save(self.agent_name + "churns_difs.npy", np.array(self.churn_difs))

    def add_churn_actions(self, actions):
        # these are the batch update actions used for grad step when calculating the policy churn
        self.churn_actions.append(np.array(actions))
        if len(self.churn_actions) % 10 == 0:
            if not self.testing:
                np.save(self.agent_name + "churns_actions.npy", np.array(self.churn_actions))


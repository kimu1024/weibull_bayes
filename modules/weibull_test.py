import pyro
import torch
import pyro.distributions as dist
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sc


class WeibullTest():
    #    torch.manual_seed(5)

    def __init__(self, num=1, weibull_m=0.50, weibull_eta=100.0, censored_prob=0):
        self.weibull_param = [float(weibull_m), float(weibull_eta)]
        self.censored_prob = float(censored_prob)
        self.dist = torch.distributions.Weibull(torch.tensor([self.weibull_param[1]]),
                                                torch.tensor([self.weibull_param[0]]))
        self.sample = []
        self.order = []
        self.num = num
        self.get_sample()

    def get_sample(self):
        time = self.dist.sample([self.num]).squeeze()
        cens = torch.distributions.bernoulli.Bernoulli(0.2).sample([self.num]).squeeze()
        self.sample = torch.stack([time, cens], dim=1)
        self.sample = self.sample[self.sample.argsort(dim=0)[:, 0]]
        self._calc_order()

        self.sample = torch.cat([self.sample, self.order.unsqueeze(1)], dim=1)
        return self.sample

    def _calc_order(self):
        self.order = torch.zeros([self.num])
        if self.sample[0, 1] == 0:
            self.order[0] = 1
        else:
            self.order[0] = 0

        for i in range(1, self.num):
            orderCorrection = (self.num + 1 - self.order[i - 1]) / (1 + (self.num - i))
            if self.sample[i, 1] == 1:
                self.order[i] = self.order[i - 1]
            else:
                self.order[i] = self.order[i - 1] + orderCorrection

    def make_plot(self):
        """
        matplotlib グラフを作成する。 subplots(nrow=1,ncols=2, figsize=(10,5)
        axes[0]がワイブルプロット(メディアンランク法), axes[1]が累積度数分布(cumulative histgram)
        return fig, axes
        """
        data_break = self.sample[self.sample[:, 1] == 0]
        # 不信頼度の計算 (メディアンランク法)
        y = np.log(np.log(1 / (1 - ((data_break[:, 2] - 0.3) / (self.num + 0.4)))))
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        wp_y_ticks_possibility = [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        wp_y_ticks_value = [self._convert_possibility_to_y_value(p) for p in wp_y_ticks_possibility]
        y_range = np.array([0.01,0.999])
        x_range = self._calc_inv_weibull_cumul_freq_dist(y_range)
        x_range[0] *=0.5
        x_range[1] *=2

        axes[0].set_xscale('log')
        axes[0].set_xlim(x_range)
        axes[0].set_ylim(self._convert_possibility_to_y_value(y_range))
        axes[0].set_yticks(wp_y_ticks_value)
        axes[0].set_yticklabels(wp_y_ticks_possibility)
        axes[0].scatter(data_break[:, 0], y, label='simulated value',
                        s=10, marker = 'o', alpha=1, color='g'
                        )

        wp_x_theory = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]))

        axes[0].plot(wp_x_theory, self._convert_possibility_to_y_value(self._calc_weibull_cumul_freq_dist(wp_x_theory)),
                     label='theoretical value')
        axes[0].legend()
        axes[0].set_title('Weibull plot')
        axes[0].set_ylabel('Probability of Failure')
        axes[0].set_xlabel('time')

        axes[1].hist(data_break[:, 0], label='simulated value', histtype='stepfilled',
                     cumulative=True, density=True, alpha=0.5, color='g',
                     bins=np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), 100))

        axes[1].plot(wp_x_theory, self._calc_weibull_cumul_freq_dist(wp_x_theory), label='theoretical value')
        axes[1].set_xscale('log')
        axes[1].set_xlim(x_range)
        axes[1].legend()
        axes[1].set_title('cumulative histgram')
        axes[1].set_ylabel('Probability of Failure')
        axes[1].set_xlabel('time')
        return fig, axes

    def _calc_weibull_cumul_freq_dist(self, x):
        return 1 - np.exp(-(x / self.weibull_param[1]) ** self.weibull_param[0])

    def _calc_inv_weibull_cumul_freq_dist(self, y):
        return (-np.log(1-y))**(1/self.weibull_param[0])*self.weibull_param[1]

    def _convert_possibility_to_y_value(self, p):
        return np.log(np.log(1 / (1 - p)))


if __name__ == '__main__':
    m = WeibullTest(1000, weibull_m=1.3, weibull_eta=100, censored_prob=0.0)
    sns.set()
    # sns.set_style('ticks')
    fig, axes = m.make_plot()
    plt.show()

from modules import weibull_test
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
import pyro.infer.autoguide
import torch
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Weibull_bayes(weibull_test.WeibullTest):
    def __init__(self, filename=None, num=10, weibull_mu=1.3, weibull_eta=100.0, exp_lambda=0, censored_time=None):
        ''' input test data from your file or make test data by the simulation

        Args:
            filename : string or None, csv file about test result. if None, make test data by simulation

            below args is for simulation, if you use an input file, they aren't used.
             num : int, the number of test samples
             weibull_mu: float, the weibull parameter about shape
             weibull_eta: float, the weibull parameter about scale
             exp_lambda=0: float, the factor of the exponential distribution about the probability of accidential test interruption.
             censored_time: float, the limit of the test time. (if a test is not finished by this time, it will be censored)

        '''
        if filename:
            self.test_data = self.read_data(filename)
            self.num = len(self.test_data)
        else:
            self.test = weibull_test.WeibullTest(num=num, weibull_mu=weibull_mu, weibull_eta=weibull_eta,
                                                 exp_lambda=exp_lambda, censored_time=censored_time)
            self.test_data = self.test.get_samples()
        self.weibull_param_samples = None  # MCMC実施後にワイブル分布の形状母数('mu')と尺度母数('eta')のサンプルが入る
        self.failure_prob_samples = None  # MCMC実施後のweibull_param_samplesに対応した、破損確率ごとのサンプル値が入る

    def read_data(self, filename):
        df = pd.read_csv(filename)
        df = df.sort_values('time').reset_index(drop=True)
        df = self._calc_order(df)
        df = self._calc_unreliability(df)
        return torch.tensor(df.values.astype(np.float32))

    def _calc_order(self, df):
        """calculate order static by the johnson's method

        Args:
            df: pandas DataFrame sorted by time.

        Returns:
             df: pandas DataFrame added 'order' columns to input df

        """
        num = len(df)
        order = pd.Series([0 for i in range(num)], dtype='float')
        if df['censored'][0] == 0:
            order[0] = 1
        else:
            order[0] = 0

        for i in range(1, num):
            order_correction = (num + 1 - order[i - 1]) / (1 + (num - i))
            if df['censored'][i] == 1:
                order[i] = order[i - 1]
            else:
                order[i] = order[i - 1] + order_correction
        order.name = 'order'
        return df.join(order)

    def _calc_unreliability(self, df):
        """calculate unreliability by the median rank method

        Args:
            df: pandas DataFrame sorted by time (['time','censored','order'].

        Returns:
             df: pandas DataFrame added 'unreliability' columns to input df
        """

        unreliability = (df['order'] - 0.3) / (len(df) + 0.4)
        unreliability.name = 'unreliability'
        return df.join(unreliability)

    def model(self, data):
        mcmc_m = pyro.sample('mu', self.non_informative_prior_distribution())
        mcmc_eta = pyro.sample('eta', self.non_informative_prior_distribution())
        data_break = data[data[:, 1] == 0]
        data_cens = data[data[:, 1] == 1]
        if data_break.size()[0] > 0:
            with pyro.plate("data_break", data_break.size()[0]):
                obs_break = pyro.sample('sim_break', dist.Weibull(scale=mcmc_eta, concentration=mcmc_m),
                                        obs=data_break[:, 0])
        if data_cens.size()[0] > 0:
            with pyro.plate("data_cens", data_cens.size()[0]):
                cens_flag = torch.ones_like(data_cens[:, 0])
                obs_cdf_prob = dist.Weibull(scale=mcmc_eta, concentration=mcmc_m).cdf(data_cens[:, 0])
                obs_cens = pyro.sample('sim_cens', dist.Bernoulli(1 - obs_cdf_prob), obs=cens_flag)

        return data

    def non_informative_prior_distribution(self, min_range=0.01, max_range=10 ** 5):
        '''Returns the non informative prior distribution of weibull constants (1/x)

        Args:
            min_range : float, the range of weibull constant
            max_range : float, the maximum range of weibull instant

        Returns:
            distribution
        '''
        min_range = torch.Tensor([min_range])
        max_range = torch.Tensor([max_range])
        a = 1 / torch.log(max_range + 1 - min_range)
        b = min_range - 1
        return dist.TransformedDistribution(dist.Uniform(low=torch.Tensor([0.0]), high=torch.Tensor([1.0])),
                                            [torch.distributions.transforms.AffineTransform(loc=0, scale=1 / a),
                                             torch.distributions.transforms.ExpTransform(),
                                             torch.distributions.transforms.AffineTransform(loc=b, scale=1)])

    def mcmc(self, loop_num=1000):
        """MCMCを実施し、事後分布を算出するためのサンプルデータを取得する

        Args:
            loop_num: int, MCMCのループ回数
        Returns:
            samples: dict{パメータ名: torch.FloatTensor(loop_num)}, ワイブルパラメータ名をkeyとした、サンプル事後データ
        """
        nuts_kernel = pyro.infer.NUTS(self.model)
        mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=loop_num, warmup_steps=500)
        mcmc.run(self.test_data[:, [0, 1]])
        self.weibull_param_samples = mcmc.get_samples()
        mcmc.summary()
        self.calculate_credible_interbal_of_failure_probability()
        return self.weibull_param_samples

    def calc_freq_sample(self, freq=0.1):
        """ワイブルパラメータの事後分布サンプルに対する、不信頼度freqにおけるサンプルを計算する
        Args:
            freq: float, 不信頼度
        Returns:
            fp_sample: 不信頼度 freqにおけるサンプル
        """
        ln_fp_sample = (math.log(math.log(1 / (1 - freq))) + self.weibull_param_samples['mu'] * torch.log(
            self.weibull_param_samples['eta'])) / self.weibull_param_samples['mu']
        fp_sample = np.exp(ln_fp_sample.squeeze().to('cpu').detach().numpy())
        return fp_sample

    def calculate_credible_interbal_of_failure_probability(self, fp=[]):
        """破損確率ごとの事後確率分布ベイズ確信区間(credible interbal)を計算する
        Args:
            fp: list, ベイズ確信区間を求めたい破損確率のリスト、空集合の場合はデフォルト値を使用
        Returns:
            credible_interbal_of_failure_probability: np.array([20,loop_num])
        """
        if not fp:
            fp = np.array([0.001, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.995, 0.999, 0.9999])
        loop_num = len(self.weibull_param_samples['mu'])
        t_value = np.zeros([len(fp), loop_num, ])
        for i in range(len(fp)):
            t_value[i, :] = self.calc_freq_sample(fp[i])

        self.failure_prob_samples = [fp, t_value]
        return self.failure_prob_samples

    def make_plot(self, credible_interbal=0.95, show=False, figsize=(10, 5), save=False, save_filename='result.png'):
        """sampleのmatplotlib グラフを作成する。 subplots(nrow=1,ncols=2, figsize=(10,5)
        axes[0]がワイブルプロット(メディアンランク法), axes[1]が累積度数分布(cumulative histgram)

        Args:
            show: bool, plt.show()を関数内で実行するかどうか
            figsize: tuple(int,int), グラフサイズ

        Returns:
             fig: matplotlib.pyplot.figure
             axes: matplotlib.pyplot.axes.Axes
        """
        data_break = self.test_data[self.test_data[:, 1] == 0]
        data_censored = self.test_data[self.test_data[:, 1] == 1]
        sorted_fp_samples = np.sort(self.failure_prob_samples[1], axis=1)
        sample_num = len(self.failure_prob_samples[1][0])

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        wp_y_ticks_possibility = [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        wp_y_ticks_value = [self.convert_unreliability_to_y(p) for p in wp_y_ticks_possibility]
        y_range_p = np.array([0.01, 0.999])
        y_range = self.convert_unreliability_to_y(y_range_p)
        x_range = [sorted_fp_samples[0][int(sample_num * 0.5)],
                   sorted_fp_samples[-1][int(sample_num * 0.5)]]

        wp_x_theory = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]))

        axes[0].set_xscale('log')
        axes[0].set_xlim(x_range)
        axes[0].set_ylim(y_range)
        axes[0].set_yticks(wp_y_ticks_value)
        axes[0].set_yticklabels(wp_y_ticks_possibility)

        if len(data_break[:, 0]) > 0:
            x = data_break[:, 0]
            y = self.convert_unreliability_to_y(data_break[:, 3])

            axes[0].scatter(x, y, label='simulated value',
                            s=10, marker='o', alpha=1, color='g'
                            )
            estimated_line, m_es, eta_es = self.calc_estimated_value(data_break[:, [0, 3]], wp_x_theory)
            axes[0].plot(estimated_line[0], estimated_line[1],
                         label='estimated by LSM:\n  $\\mu$={0:.2f}, $\\eta$={1:.1f}'.format(m_es, eta_es),
                         linestyle='--', color='g', alpha=1)

        num_censored = len(data_censored[:, 0])
        if num_censored > 0:
            axes[0].scatter(data_censored[:, 0],
                            np.array([(y_range[0] + wp_y_ticks_value[0]) / 2 for i in range(num_censored)]),
                            label='censored data\n (num = {})'.format(num_censored), s=8, marker='x', alpha=1,
                            color='r')

        # plot credible interbal
        y_ci = self.convert_unreliability_to_y(self.failure_prob_samples[0])
        x_low = sorted_fp_samples[:, int(sample_num * (1 - credible_interbal) / 2)]
        x_high = sorted_fp_samples[:, int(sample_num * (credible_interbal + (1 - credible_interbal) / 2))]
        axes[0].plot(x_low, y_ci, label='{0:.0f}% credible interbal'.format(credible_interbal * 100), color='g',
                     linestyle="-.",
                     alpha=0.8)
        axes[0].plot(x_high, y_ci, color='g', linestyle="-.", alpha=0.8)

        axes[0].legend()
        axes[0].set_title('Weibull plot (N={0})'.format(self.num))
        axes[0].set_ylabel('Probability of Failure')
        axes[0].set_xlabel('time')
        axes[0].grid(True, which='major', axis='both')
        axes[0].grid(True, which='minor', axis='x')

        if len(data_break[:, 0]) > 0:
            axes[1].plot(wp_x_theory, self._calc_weibull_cumul_freq_dist(wp_x_theory, mu=m_es, eta=eta_es),
                         label='estimated', color='g',
                         linestyle='--', alpha=0.8)
            axes[1].plot(data_break[:, 0], data_break[:, 3], label='simulated value', color='g', alpha=1, linewidth=3)

        axes[1].set_xscale('log')
        axes[1].set_xlim(x_range)
        axes[1].legend()
        axes[1].set_title('cumulative histgram (N={0})'.format(self.num))
        axes[1].set_ylabel('Probability of Failure')
        axes[1].set_xlabel('time')
        axes[1].grid(True, which='major', axis='both')
        axes[1].grid(True, which='minor', axis='x')

        if save:
            fig.savefig(save_filename)

        if show:
            plt.show()

        return fig, axes


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    wb = Weibull_bayes(filename='../test_sample.csv')

    sns.set()
    pyro.set_rng_seed(1)

    #    wb = Weibull_bayes(100, exp_lambda=0.01, censored_time=None)
    print(wb.test_data)
    samples = wb.mcmc(1000)
    wb.calculate_credible_interbal_of_failure_probability()
    wb.make_plot(show=False, save=True, save_filename='test.png')

#    sns.jointplot(samples['mu'], samples['eta'], kind='kde')
#    plt.show()
# fig, axes = wb.test.make_plot(plot_theoritical=True)
# freq = 0.95
# num = len(wb.failure_prob_samples[1][0])
#
# y = wb.test.convert_unreliability_to_y(wb.failure_prob_samples[0])
# sorted_fp_samples = np.sort(wb.failure_prob_samples[1], axis=1)
# x_low = sorted_fp_samples[:, int(num * (1 - freq) / 2)]
# x_high = sorted_fp_samples[:, int(num * (freq + (1 - freq) / 2))]
#
# axes[0].plot(x_low, y, label='{0:.0f}% credible interbal'.format(freq * 100), color='g', linestyle="-.", alpha=0.8)
# axes[0].plot(x_high, y, color='g', linestyle="-.", alpha=0.8)
# axes[0].legend(loc='upper left')
# plt.tight_layout()
# plt.show()


# plt.hist(wb.calc_cumul_freq())
# plt.show()

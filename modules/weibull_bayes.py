from modules import weibull_test
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
import pyro.infer.autoguide
import torch
import math
import numpy as np


class Weibull_bayes():
    def __init__(self, num=10, weibull_mu=1.3, weibull_eta=100.0, exp_lambda=0, censored_time=None):
        self.test = weibull_test.WeibullTest(num=num, weibull_mu=weibull_mu, weibull_eta=weibull_eta,
                                             exp_lambda=exp_lambda, censored_time=censored_time)
        self.weibull_param_samples = None  # MCMC実施後にワイブル分布の形状母数('mu')と尺度母数('eta')のサンプルが入る
        self.failure_prob_samples = None  # MCMC実施後のweibull_param_samplesに対応した、破損確率ごとのサンプル値が入る

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
        mcmc.run(self.test.get_samples()[:, [0, 1]])
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

    def calculate_credible_interbal_of_failure_probability(self):
        """破損確率ごとの事後確率分布ベイズ確信区間(credible interbal)を計算する
        Returns:
            credible_interbal_of_failure_probability: np.array([20,loop_num])
        """
        fp_value = np.array([0.001, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.995, 0.999, 0.9999])
        loop_num = len(self.weibull_param_samples['mu'])
        t_value = np.zeros([len(fp_value), loop_num, ])
        for i in range(len(fp_value)):
            t_value[i, :] = self.calc_freq_sample(fp_value[i])

        self.failure_prob_samples = [fp_value, t_value]
        return self.failure_prob_samples


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()
    pyro.set_rng_seed(1)

    wb = Weibull_bayes(100, exp_lambda=0.01, censored_time=None)
    print(wb.test.get_samples())
    samples = wb.mcmc(1000)

    sns.jointplot(samples['mu'], samples['eta'], kind='kde')
    plt.show()
    fig, axes = wb.test.make_plot(plot_theoritical=True)
    freq = 0.95
    num = len(wb.failure_prob_samples[1][0])

    y = wb.test.convert_unreliability_to_y(wb.failure_prob_samples[0])
    sorted_fp_samples = np.sort(wb.failure_prob_samples[1], axis=1)
    x_low = sorted_fp_samples[:, int(num * (1 - freq) / 2)]
    x_high = sorted_fp_samples[:, int(num * (freq + (1 - freq) / 2))]

    axes[0].plot(x_low, y, label='{0:.0f}% credible interbal'.format(freq * 100), color='g', linestyle="-.", alpha=0.8)
    axes[0].plot(x_high, y, color='g', linestyle="-.", alpha=0.8)
    axes[0].legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    # plt.show()
    # plt.hist(wb.calc_cumul_freq())
    # plt.show()

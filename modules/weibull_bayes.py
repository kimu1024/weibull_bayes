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
    def __init__(self, num=10, weibull_m=1.3, weibull_eta=100.0, censored_prob=0):
        self.test = weibull_test.WeibullTest(num=num, weibull_m=weibull_m, weibull_eta=weibull_eta,
                                             censored_prob=censored_prob)
        self.weibull_param_samples = None  # MCMC実施後にワイブル分布の形状母数('m')と尺度母数('eta')のサンプルが入る
        self.failure_prob_samples = None  # MCMC実施後のweibull_param_samplesに対応した、破損確率ごとのサンプル値が入る

    def model(self, data):
        # simple version, not consider whether the data is censored)
        mcmc_m = pyro.sample('m', self.non_informative_prior_distribution())
        mcmc_eta = pyro.sample('eta', self.non_informative_prior_distribution())
        # mcmc_m = pyro.sample('m', dist.Uniform(low=0.5, high=2.0))
        # mcmc_eta = pyro.sample('eta', dist.Uniform(low=50,high=1000))

        #        mcmc_eta = pyro.sample('eta', dist.Uniform(low=torch.Tensor([1]),high=torch.Tensor([1000])))
        with pyro.plate("data", data.size()[0]):
            obs = pyro.sample('sim', dist.Weibull(scale=mcmc_eta, concentration=mcmc_m), obs=data)
        return obs

    def non_informative_prior_distribution(self, min=0.01, max=10 ** 5):
        '''Returns the non informative prior distribution of weibull constants (1/x)

        Args:
            min : float, the range of weibull constant
            max : float, the maximum range of weibull instant

        Returns:
            distribution
            '''
        min = torch.Tensor([min])
        max = torch.Tensor([max])
        a = 1 / torch.log(max + 1 - min)
        b = min - 1
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
        # initial_params={'m': torch.Tensor([1.3]), 'eta': torch.Tensor([100])})
        mcmc.run(self.test.get_samples()[:, 0])
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
        ln_fp_sample = (math.log(math.log(1 / (1 - freq))) + self.weibull_param_samples['m'] * torch.log(
            self.weibull_param_samples['eta'])) / self.weibull_param_samples['m']
        fp_sample = np.exp(ln_fp_sample.squeeze().to('cpu').detach().numpy())
        return fp_sample

    def calculate_credible_interbal_of_failure_probability(self):
        """破損確率ごとの事後確率分布ベイズ確信区間(credible interbal)を計算する
        Returns:
            credible_interbal_of_failure_probability: np.array([20,loop_num])
        """
        fp_value = np.array([0.001, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.995, 0.999, 0.9999])
        loop_num = len(self.weibull_param_samples['m'])
        t_value = np.zeros([len(fp_value), loop_num, ])
        for i in range(len(fp_value)):
            t_value[i, :] = self.calc_freq_sample(fp_value[i])

        self.failure_prob_samples=[fp_value, t_value]
        return self.failure_prob_samples

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    wb = Weibull_bayes(10)

    samples = wb.mcmc(500)
    print(wb.calculate_credible_interbal_of_failure_probability())
    # plt.plot(samples['m'])
    # plt.show()
    # wb.test.make_plot()
    # plt.show()
    # plt.hist(wb.calc_cumul_freq())
    # plt.show()

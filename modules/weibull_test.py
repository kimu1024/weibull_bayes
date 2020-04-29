import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class WeibullTest():
    """ワイブル分布に従う寿命試験のシミュレーションを行う"""

    def __init__(self, num=10, weibull_m=1.3, weibull_eta=100.0, censored_prob=0):
        """シミュレーション条件の初期化

        Args:
            num: int, 試験点数
            weibull_m: float, ワイブル係数(形状パラメータ)
            weibull_eta: float, 尺度パラメータ
            censored_prob: 打ち切りデータの発生確率 (機能実装中、現状は0以外は動作非保証)

        """
        self.weibull_param = [float(weibull_m), float(weibull_eta)]
        self.censored_prob = float(censored_prob)
        self.dist = torch.distributions.Weibull(torch.tensor([self.weibull_param[1]]),
                                                torch.tensor([self.weibull_param[0]]))
        self.order = []
        self.unreliability = []
        self.num = num
        self.sample = self._make_samples()

    def _make_samples(self):
        """サンプルデータ作成

        Returns:
            self.sample: torch.FloatTensor(num, 4)
                self.sample[:,0] = 各試験の終了時間
                self.sample[:,1] = 打ち切り有無(1: 打ち切り終了、0: 破損終了)
                self.sample[:,2] = 順序統計量 (Johnsonの方法)
                self.sample[:,3] = 不信頼度 (メディアンランク法)
        """
        time = self.dist.sample([self.num]).squeeze()
        cens = torch.distributions.bernoulli.Bernoulli(self.censored_prob).sample([self.num]).squeeze()
        self.sample = torch.stack([time, cens], dim=1)
        self.sample = self.sample[self.sample.argsort(dim=0)[:, 0]]
        self._calc_order()
        self._calc_unreliability()

        self.sample = torch.cat([self.sample, self.order.unsqueeze(1)], dim=1)
        self.sample = torch.cat([self.sample, self.unreliability.unsqueeze(1)], dim=1)
        return self.sample

    def get_samples(self):
        """サンプルデータの取得

        Returns:
            self.sample: torch.FloatTensor(num, 4)
                self.sample[:,0] = 各試験の終了時間
                self.sample[:,1] = 打ち切り有無(1: 打ち切り終了、0: 破損終了)
                self.sample[:,2] = 順序統計量 (Johnsonの方法)
                self.sample[:,3] = 不信頼度 (メディアンランク法)
        """
        return self.sample

    def _calc_order(self):
        """順序統計量をJohnsonの方法により計算する
        """
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
        self.order = self.order

    def _calc_unreliability(self):
        """試験データから不信頼度を計算する(メディアンランク法)"""
        self.unreliability = (self.order - 0.3) / (self.num + 0.4)
        self.unreliability = self.unreliability

    def convert_unreliability_to_y(self, unreliability):
        """不信頼度を、ワイブルプロットの縦軸に変換する
            Args:
                unreliability: torch.FloatTensor, 不信頼度
            Returns:
                y: torch.FloatTensor,ワイブルプロット縦軸
        """
        return np.log(np.log(1 / (1 - unreliability)))

    def make_plot(self, show=False, figsize=(10, 5)):
        """sampleのmatplotlib グラフを作成する。 subplots(nrow=1,ncols=2, figsize=(10,5)
        axes[0]がワイブルプロット(メディアンランク法), axes[1]が累積度数分布(cumulative histgram)

        Args:
            show: bool, plt.show()を関数内で実行するかどうか
            figsize: tuple(int,int), グラフサイズ

        Returns:
             fig: matplotlib.pyplot.figure
             axes: matplotlib.pyplot.axes.Axes
        """
        data_break = self.sample[self.sample[:, 1] == 0]
        x = data_break[:, 0]
        y = self.convert_unreliability_to_y(data_break[:, 3])
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        wp_y_ticks_possibility = [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        wp_y_ticks_value = [self.convert_unreliability_to_y(p) for p in wp_y_ticks_possibility]
        y_range = np.array([0.01, 0.999])
        x_range = self._calc_inv_weibull_cumul_freq_dist(y_range)
        x_range[0] *= 0.5
        x_range[1] *= 2

        axes[0].set_xscale('log')
        axes[0].set_xlim(x_range)
        axes[0].set_ylim(self.convert_unreliability_to_y(y_range))
        axes[0].set_yticks(wp_y_ticks_value)
        axes[0].set_yticklabels(wp_y_ticks_possibility)
        axes[0].scatter(x, y, label='simurated value'.format(self.num),
                        s=10, marker='o', alpha=1, color='g'
                        )

        wp_x_theory = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]))
        estimated_line, m_es, eta_es = self.calc_estimated_value(data_break[:, [0, 3]], wp_x_theory)

        axes[0].plot(wp_x_theory, self.convert_unreliability_to_y(self._calc_weibull_cumul_freq_dist(wp_x_theory)),
                     label='theoretical:\n  $\\mu$={0:.2f}, $\\eta$={1:.1f}'.format(self.weibull_param[0],
                                                                              self.weibull_param[1]), color='black',
                     alpha=0.5, linestyle='--')

        axes[0].plot(estimated_line[0], estimated_line[1],
                     label='estimated:\n  $\\mu$={0:.2f}, $\\eta$={1:.1f}'.format(m_es, eta_es),
                     linestyle='--', color='g', alpha=1)
        axes[0].legend()
        axes[0].set_title('Weibull plot (N={0})'.format(self.num))
        axes[0].set_ylabel('Probability of Failure')
        axes[0].set_xlabel('time')

        axes[1].hist(data_break[:, 0], label='simulated', histtype='stepfilled',
                     cumulative=True, density=True, alpha=0.5, color='g',
                     bins=np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), 100))

        axes[1].plot(wp_x_theory, self._calc_weibull_cumul_freq_dist(wp_x_theory), label='theoretical', color='black',
                     linestyle='--', alpha=0.5)
        axes[1].set_xscale('log')
        axes[1].set_xlim(x_range)
        axes[1].legend()
        axes[1].set_title('cumulative histgram (N={0})'.format(self.num))
        axes[1].set_ylabel('Probability of Failure')
        axes[1].set_xlabel('time')

        if show:
            plt.show()

        return fig, axes

    def calc_estimated_value(self, data, x_range=np.array([])):
        """ワイブルプロット上で、試験データを線形近似し、近似線と推定値を出力する
            Args:
                data: torch.FloatTensor(num, 2), 打ち切りデータを除いたsampleデータ
                    data[:,0] : float, 試験終了時間
                    data[:,1] : float, 不信頼度
                x_range: np.array(n), 最小二乗直線の描画範囲
            Returns:
                estimate_line:np.array(2,n), 近似戦の点列データ
                m_es: float, 形状パラメータの推定値
                eta_es: float, 尺度パラメータの推定値
        """
        y = self.convert_unreliability_to_y(data[:, 1])
        logx = np.log(data[:, 0])
        a_es, b_es = np.polyfit(logx, y, 1)
        estimate_line = np.stack([x_range, a_es * np.log(x_range) + b_es], axis=0)
        m_es = a_es
        eta_es = np.exp(-b_es / m_es)

        return estimate_line, m_es, eta_es

    def _calc_weibull_cumul_freq_dist(self, x):
        return 1 - np.exp(-(x / self.weibull_param[1]) ** self.weibull_param[0])

    def _calc_inv_weibull_cumul_freq_dist(self, y):
        return (-np.log(1 - y)) ** (1 / self.weibull_param[0]) * self.weibull_param[1]


if __name__ == '__main__':
    m = WeibullTest(10, weibull_m=1.3, weibull_eta=100, censored_prob=0.0)
    sns.set()
    # sns.set_style('ticks')
    fig, axes = m.make_plot(show=True)

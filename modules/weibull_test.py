import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class WeibullTest():
    """ワイブル分布に従う寿命試験のシミュレーションを行う"""

    def __init__(self, num=10, weibull_mu=1.3, weibull_eta=100.0, exp_lambda=0, censored_time=None):
        """シミュレーション条件の初期化

        Args:
            num: int, 試験点数
            weibull_mu: float, ワイブル係数(形状パラメータ)
            weibull_eta: float, 尺度パラメータ
            exp_lambda: トラブルなどによる試験の中断が指数分布に従い発生すると仮定 (本値が0の場合中断は発生しない)
            censored_time: float, 試験打ち切り時間(この時間を超える試験データは打ち切りとする)

        """
        self.weibull_param = [float(weibull_mu), float(weibull_eta)]
        self.exp_lambda = float(exp_lambda)
        self.censored_time = float(censored_time) if censored_time else None
        self.dist = torch.distributions.Weibull(torch.tensor([self.weibull_param[1]]),
                                                torch.tensor([self.weibull_param[0]]))
        self.order = []
        self.unreliability = []
        self.num = num
        self.test_data = self._make_samples()

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
        if self.exp_lambda > 0:
            cens_time = torch.distributions.exponential.Exponential(self.exp_lambda).sample([self.num]).squeeze()
            cens = (cens_time < time).float()
            time[cens == 1.0] = cens_time[cens == 1.0]
        else:
            cens = torch.zeros(self.num, dtype=torch.float)
        if self.censored_time:
            cens[time > self.censored_time] = 1.0
            time[time > self.censored_time] = self.censored_time

        self.test_data = torch.stack([time, cens], dim=1)
        self.test_data = self.test_data[self.test_data.argsort(dim=0)[:, 0]]
        self._calc_order()
        self._calc_unreliability()

        self.test_data = torch.cat([self.test_data, self.order.unsqueeze(1)], dim=1)
        self.test_data = torch.cat([self.test_data, self.unreliability.unsqueeze(1)], dim=1)
        return self.test_data

    def get_samples(self):
        """サンプルデータの取得

        Returns:
            self.sample: torch.FloatTensor(num, 4)
                self.sample[:,0] = 各試験の終了時間
                self.sample[:,1] = 打ち切り有無(1: 打ち切り終了、0: 破損終了)
                self.sample[:,2] = 順序統計量 (Johnsonの方法)
                self.sample[:,3] = 不信頼度 (メディアンランク法)
        """
        return self.test_data

    def _calc_order(self):
        """順序統計量をJohnsonの方法により計算する
        """
        self.order = torch.zeros([self.num])
        if self.test_data[0, 1] == 0:
            self.order[0] = 1
        else:
            self.order[0] = 0

        for i in range(1, self.num):
            order_correction = (self.num + 1 - self.order[i - 1]) / (1 + (self.num - i))
            if self.test_data[i, 1] == 1:
                self.order[i] = self.order[i - 1]
            else:
                self.order[i] = self.order[i - 1] + order_correction


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

    def make_plot(self, show=False, plot_theoritical=False, figsize=(10, 5)):
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
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        wp_y_ticks_possibility = [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        wp_y_ticks_value = [self.convert_unreliability_to_y(p) for p in wp_y_ticks_possibility]
        y_range_p = np.array([0.01, 0.999])
        y_range = self.convert_unreliability_to_y(y_range_p)
        x_range = self._calc_inv_weibull_cumul_freq_dist(y_range_p, mu=self.weibull_param[0], eta=self.weibull_param[1])
        x_range[0] *= 0.5
        x_range[1] *= 2
        wp_x_theory = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]))

        axes[0].set_xscale('log')
        axes[0].set_xlim(x_range)
        axes[0].set_ylim(y_range)
        axes[0].set_yticks(wp_y_ticks_value)
        axes[0].set_yticklabels(wp_y_ticks_possibility)

        if plot_theoritical:
            axes[0].plot(wp_x_theory, self.convert_unreliability_to_y(
                self._calc_weibull_cumul_freq_dist(wp_x_theory, mu=self.weibull_param[0], eta=self.weibull_param[1])),
                         label='theoretical:\n  $\\mu$={0:.2f}, $\\eta$={1:.1f}'.format(self.weibull_param[0],
                                                                                        self.weibull_param[1]),
                         color='black',
                         alpha=0.5, linestyle='--')

        if len(data_break[:, 0]) > 0:
            x = data_break[:, 0]
            y = self.convert_unreliability_to_y(data_break[:, 3])

            axes[0].scatter(x, y, label='simurated value',
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


        axes[0].legend()
        axes[0].set_title('Weibull plot (N={0})'.format(self.num))
        axes[0].set_ylabel('Probability of Failure')
        axes[0].set_xlabel('time')
        axes[0].grid(True, which='minor', axis='x')

        if plot_theoritical:
            axes[1].plot(wp_x_theory, self._calc_weibull_cumul_freq_dist(wp_x_theory, mu=self.weibull_param[0],
                                                                         eta=self.weibull_param[1]),
                         label='theoretical', color='black',
                         linestyle='--', alpha=0.5)

        if len(data_break[:, 0]) > 0:
            axes[1].plot(wp_x_theory, self._calc_weibull_cumul_freq_dist(wp_x_theory, mu=m_es, eta = eta_es),
                         label='estimated', color='g',
                         linestyle='--', alpha=0.8)
            axes[1].plot(data_break[:, 0], data_break[:, 3], label='simulated value', color='g', alpha=1, linewidth=3)


        axes[1].set_xscale('log')
        axes[1].set_xlim(x_range)
        axes[1].legend()
        axes[1].set_title('cumulative histgram (N={0})'.format(self.num))
        axes[1].set_ylabel('Probability of Failure')
        axes[1].set_xlabel('time')
        axes[1].grid(True, which='minor', axis='x')

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

    def _calc_weibull_cumul_freq_dist(self, x, mu, eta):
        return 1 - np.exp(-(x / eta) ** mu)

    def _calc_inv_weibull_cumul_freq_dist(self, y, mu, eta):
        return (-np.log(1 - y)) ** (1 / mu) * eta


if __name__ == '__main__':
    m = WeibullTest(10, weibull_mu=1.3, weibull_eta=100, exp_lambda=0.01)
    sns.set(style = "whitegrid")
    fig, axes = m.make_plot(show=True, plot_theoritical=True)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# 设置中文字体（如果需要显示中文）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

logger = logging.getLogger(__name__)


class ChartPlotter:
    """高级图表绘制器，支持技术指标叠加和多股票对比"""

    def __init__(self, figsize: Tuple[int, int] = (16, 10)):
        self.figsize = figsize
        self.output_dir = Path("./output/charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 颜色配置
        self.colors = {
            'up': '#2E7D32',  # 上涨绿色
            'down': '#C62828',  # 下跌红色
            'volume': '#546E7A',  # 成交量灰色
            'ma_short': '#FF6F00',  # 短期均线橙色
            'ma_medium': '#1976D2',  # 中期均线蓝色
            'ma_long': '#7B1FA2',  # 长期均线紫色
            'macd': '#2196F3',  # MACD蓝色
            'signal': '#FF9800',  # 信号线橙色
            'rsi': '#0097A7',  # RSI青色
            'boll_upper': '#8D6E63',  # 布林上轨棕色
            'boll_lower': '#8D6E63',  # 布林下轨棕色
            'custom': '#E91E63'  # 自定义指标粉色
        }

        logger.info("ChartPlotter 初始化完成")

    def plot_stock_analysis(self,
                            df: pd.DataFrame,
                            symbol: str = "",
                            indicators: List[str] = None,
                            title: str = None,
                            output_path: str = None,
                            show: bool = False) -> str:
        """
        绘制完整的股票技术分析图表（K线+指标）

        参数:
            df: 包含OHLCV和指标的DataFrame
            symbol: 股票代码
            indicators: 要显示的指标列表，默认显示常见指标
            title: 图表标题
            output_path: 输出文件路径
            show: 是否显示图表

        返回:
            保存的文件路径
        """
        if df.empty:
            logger.warning(f"数据为空，无法绘制图表")
            return ""

        # 设置默认指标
        if indicators is None:
            indicators = ['candlestick', 'volume', 'ma', 'macd', 'rsi']

        # 计算指标数量
        num_plots = 1  # 主图（K线）
        if 'volume' in indicators:
            num_plots += 1
        if 'macd' in indicators:
            num_plots += 1
        if 'rsi' in indicators:
            num_plots += 1
        if any(x.startswith('custom_') for x in df.columns):
            num_plots += 1

        # 创建图形
        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(num_plots, 1, height_ratios=[3] + [1] * (num_plots - 1))

        # 获取时间序列
        dates = df.index
        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values if 'volume' in df.columns else None

        current_axis = 0

        # 1. 主图（K线图）
        ax1 = plt.subplot(gs[current_axis])
        if current_axis == 0:
            ax1.set_title(title or f"{symbol} 技术分析", fontsize=16, fontweight='bold')

        # 绘制K线
        self._plot_candlestick(ax1, dates, opens, highs, lows, closes)

        # 添加均线
        if 'ma' in indicators:
            self._add_moving_averages(ax1, dates, df)

        # 添加布林带（如果有）
        if 'boll_upper' in df.columns and 'boll_lower' in df.columns:
            self._add_bollinger_bands(ax1, dates, df)

        ax1.set_ylabel('价格', fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left')

        current_axis += 1

        # 2. 成交量图
        if 'volume' in indicators and volumes is not None:
            ax2 = plt.subplot(gs[current_axis], sharex=ax1)
            self._plot_volume(ax2, dates, volumes, closes)
            ax2.set_ylabel('成交量', fontsize=12)
            ax2.grid(True, alpha=0.3, linestyle='--')
            current_axis += 1

        # 3. MACD图
        if 'macd' in indicators and all(col in df.columns for col in ['macd', 'macd_signal']):
            ax3 = plt.subplot(gs[current_axis], sharex=ax1)
            self._plot_macd(ax3, dates, df)
            ax3.set_ylabel('MACD', fontsize=12)
            ax3.grid(True, alpha=0.3, linestyle='--')
            current_axis += 1

        # 4. RSI图
        if 'rsi' in indicators and 'rsi' in df.columns:
            ax4 = plt.subplot(gs[current_axis], sharex=ax1)
            self._plot_rsi(ax4, dates, df)
            ax4.set_ylabel('RSI', fontsize=12)
            ax4.grid(True, alpha=0.3, linestyle='--')
            current_axis += 1

        # 5. 自定义指标图
        custom_indicators = [col for col in df.columns if col.startswith('custom_')]
        for custom_indicator in custom_indicators[:1]:  # 先绘制第一个自定义指标
            ax5 = plt.subplot(gs[current_axis], sharex=ax1)
            ax5.plot(dates, df[custom_indicator],
                     color=self.colors['custom'],
                     linewidth=1.5,
                     label=custom_indicator)

            # 添加零线
            ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.5)

            ax5.set_ylabel(custom_indicator, fontsize=12)
            ax5.grid(True, alpha=0.3, linestyle='--')
            ax5.legend(loc='upper left')
            current_axis += 1

        # 设置x轴格式
        plt.xlabel('日期', fontsize=12)
        plt.xticks(rotation=45)

        # 自动调整布局
        plt.tight_layout()

        # 保存图表
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"{symbol}_{timestamp}_analysis.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"图表已保存: {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return str(output_path)

    def plot_multiple_stocks(self,
                             data_dict: Dict[str, pd.DataFrame],
                             indicators: List[str] = None,
                             title: str = "多股票对比",
                             output_path: str = None,
                             show: bool = False) -> str:
        """
        将多只股票的指定指标绘制在同一张图上进行对比（核心功能）

        参数:
            data_dict: 字典，键为股票代码，值为DataFrame
            indicators: 要对比的指标列表，默认['close']
            title: 图表标题
            output_path: 输出路径
            show: 是否显示

        返回:
            保存的文件路径
        """
        if not data_dict:
            logger.warning("没有数据可供对比")
            return ""

        if indicators is None:
            indicators = ['close']

        # 创建图形
        fig, axes = plt.subplots(len(indicators), 1, figsize=self.figsize)
        if len(indicators) == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 为每只股票分配颜色
        colors = plt.cm.tab20(np.linspace(0, 1, len(data_dict)))

        for idx, indicator in enumerate(indicators):
            ax = axes[idx]

            for (symbol, df), color in zip(data_dict.items(), colors):
                if indicator in df.columns:
                    # 只绘制最近N个交易日的数据，避免过于拥挤
                    plot_data = df[indicator].iloc[-100:] if len(df) > 100 else df[indicator]

                    ax.plot(plot_data.index, plot_data.values,
                            color=color, linewidth=2, label=symbol)

                    # 在线的末尾添加标签
                    if not plot_data.empty:
                        last_value = plot_data.iloc[-1]
                        ax.text(plot_data.index[-1], last_value,
                                f' {symbol}:{last_value:.2f}',
                                fontsize=9, color=color,
                                verticalalignment='center')

            ax.set_ylabel(indicator.capitalize(), fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', fontsize=9, ncol=2)

            # 设置x轴格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.xlabel('日期', fontsize=12)
        plt.tight_layout()

        # 保存图表
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            indicator_str = '_'.join(indicators[:2])
            output_path = self.output_dir / f"multi_stock_{indicator_str}_{timestamp}.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"多股对比图已保存: {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return str(output_path)

    def plot_screening_summary(self,
                               screening_results: Dict[str, Any],
                               output_path: str = None,
                               show: bool = False) -> str:
        """
        绘制筛选结果的汇总图表

        参数:
            screening_results: 筛选结果字典
            output_path: 输出路径
            show: 是否显示
        """
        if not screening_results:
            logger.warning("没有筛选结果可供绘制")
            return ""

        # 提取选中的股票
        selected_stocks = {k: v for k, v in screening_results.items() if v.get('selected', False)}

        if not selected_stocks:
            logger.info("没有股票被选中")
            return ""

        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. 饼图：选中比例
        total_analyzed = len(screening_results)
        total_selected = len(selected_stocks)
        total_rejected = total_analyzed - total_selected

        ax1 = axes[0]
        sizes = [total_selected, total_rejected]
        labels = [f'选中\n{total_selected}只', f'未选中\n{total_rejected}只']
        colors = ['#4CAF50', '#F44336']

        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'股票筛选结果 (共分析{total_analyzed}只)', fontsize=14)

        # 2. 柱状图：选中的股票
        ax2 = axes[1]

        if selected_stocks:
            # 提取选中的股票和得分（如果有）
            symbols = list(selected_stocks.keys())

            # 尝试获取分数或排序依据
            scores = []
            for symbol in symbols:
                score = selected_stocks[symbol].get('score', 0)
                scores.append(score)

            # 如果没有分数，使用顺序
            if all(s == 0 for s in scores):
                scores = range(len(symbols))

            # 排序
            sorted_indices = np.argsort(scores)[::-1]  # 降序
            symbols = [symbols[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]

            # 绘制柱状图
            bars = ax2.bar(range(len(symbols)), scores, color='#2196F3')
            ax2.set_xlabel('股票代码', fontsize=12)
            ax2.set_ylabel('得分/排序', fontsize=12)
            ax2.set_title('选中股票排序', fontsize=14)
            ax2.set_xticks(range(len(symbols)))
            ax2.set_xticklabels(symbols, rotation=45)

            # 在柱子上添加数值
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{score:.1f}' if isinstance(score, float) else f'{score}',
                         ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # 保存图表
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"screening_summary_{timestamp}.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"筛选汇总图已保存: {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return str(output_path)

    # ========== 私有辅助方法 ==========

    def _plot_candlestick(self, ax, dates, opens, highs, lows, closes):
        """绘制K线图"""
        # 确定上涨和下跌
        up = closes >= opens
        down = closes < opens

        # 设置宽度
        width = 0.6
        width2 = 0.1

        # 绘制上涨K线
        if up.any():
            ax.bar(dates[up], closes[up] - opens[up], width,
                   bottom=opens[up], color=self.colors['up'], edgecolor=self.colors['up'])
            ax.bar(dates[up], highs[up] - closes[up], width2,
                   bottom=closes[up], color=self.colors['up'], edgecolor=self.colors['up'])
            ax.bar(dates[up], opens[up] - lows[up], width2,
                   bottom=lows[up], color=self.colors['up'], edgecolor=self.colors['up'])

        # 绘制下跌K线
        if down.any():
            ax.bar(dates[down], closes[down] - opens[down], width,
                   bottom=opens[down], color=self.colors['down'], edgecolor=self.colors['down'])
            ax.bar(dates[down], highs[down] - opens[down], width2,
                   bottom=opens[down], color=self.colors['down'], edgecolor=self.colors['down'])
            ax.bar(dates[down], closes[down] - lows[down], width2,
                   bottom=lows[down], color=self.colors['down'], edgecolor=self.colors['down'])

    def _plot_volume(self, ax, dates, volumes, closes):
        """绘制成交量图"""
        # 根据价格涨跌确定颜色
        colors = []
        for i in range(1, len(closes)):
            if i < len(closes) and i < len(volumes):
                if closes[i] >= closes[i - 1]:
                    colors.append(self.colors['up'])
                else:
                    colors.append(self.colors['down'])

        # 确保长度一致
        min_len = min(len(dates), len(volumes), len(colors) + 1)

        if min_len > 1:
            ax.bar(dates[:min_len - 1], volumes[:min_len - 1],
                   color=colors[:min_len - 1], width=0.6, alpha=0.7)

    def _add_moving_averages(self, ax, dates, df):
        """添加移动平均线"""
        ma_cols = [col for col in df.columns if col.startswith('ma_')]

        for ma_col in ma_cols:
            if ma_col in df.columns:
                # 根据周期分配颜色
                period = ma_col.split('_')[1] if '_' in ma_col else ''
                if period.isdigit():
                    period_num = int(period)
                    if period_num <= 10:
                        color = self.colors['ma_short']
                        label = f'MA{period_num}'
                    elif period_num <= 30:
                        color = self.colors['ma_medium']
                        label = f'MA{period_num}'
                    else:
                        color = self.colors['ma_long']
                        label = f'MA{period_num}'
                else:
                    color = self.colors['ma_medium']
                    label = ma_col

                ax.plot(dates, df[ma_col], color=color, linewidth=1.5, alpha=0.8, label=label)

    def _add_bollinger_bands(self, ax, dates, df):
        """添加布林带"""
        if 'boll_upper' in df.columns and 'boll_lower' in df.columns:
            ax.fill_between(dates, df['boll_upper'], df['boll_lower'],
                            color=self.colors['boll_upper'], alpha=0.2, label='布林带')

    def _plot_macd(self, ax, dates, df):
        """绘制MACD指标"""
        ax.plot(dates, df['macd'], color=self.colors['macd'], linewidth=1.5, label='MACD')
        ax.plot(dates, df['macd_signal'], color=self.colors['signal'], linewidth=1.5, label='信号线')

        # 绘制MACD柱状图
        macd_hist = df['macd'] - df['macd_signal']
        colors_macd_hist = np.where(macd_hist >= 0, self.colors['up'], self.colors['down'])
        ax.bar(dates, macd_hist, color=colors_macd_hist, alpha=0.5, width=0.6)

        # 添加零线
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.5)
        ax.legend(loc='upper left')

    def _plot_rsi(self, ax, dates, df):
        """绘制RSI指标"""
        ax.plot(dates, df['rsi'], color=self.colors['rsi'], linewidth=1.5, label='RSI')

        # 添加超买超卖线
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, linewidth=1, label='超买线 (70)')
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, linewidth=1, label='超卖线 (30)')
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5, linewidth=0.5)

        # 填充RSI区域
        ax.fill_between(dates, df['rsi'], 70, where=(df['rsi'] >= 70),
                        color='red', alpha=0.2)
        ax.fill_between(dates, df['rsi'], 30, where=(df['rsi'] <= 30),
                        color='green', alpha=0.2)

        ax.set_ylim(0, 100)
        ax.legend(loc='upper left')
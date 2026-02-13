import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ScreeningRule:
    """筛选规则定义"""
    name: str  # 规则名称
    condition_func: Callable  # 条件函数
    weight: float = 1.0  # 规则权重
    description: str = ""  # 规则描述

    def __post_init__(self):
        if not self.description:
            self.description = self.name


class StockScreener:
    """股票筛选器，支持多规则组合筛选"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化筛选器

        参数:
            config: 配置字典，包含筛选参数
        """
        self.config = config or self._get_default_config()
        self.rules = self._initialize_rules()
        self.results_cache = {}

        logger.info(f"StockScreener 初始化完成，共加载 {len(self.rules)} 条规则")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'min_price': 1.0,  # 最低价格
            'max_price': 1000.0,  # 最高价格
            'min_volume': 1000000,  # 最低成交量
            'min_market_cap': 1e9,  # 最低市值（如有）
            'rsi_overbought': 70,  # RSI超买线
            'rsi_oversold': 30,  # RSI超卖线
            'ma_periods': [5, 10, 20, 30, 60],  # 均线周期
            'volatility_period': 20,  # 波动率计算周期
            'min_days': 60,  # 最少数据天数
            'custom_thresholds': {  # 自定义指标阈值
                'custom_indicator_1': {'min': -2, 'max': 2},
                'momentum_score': {'min': 0.5}
            }
        }

    def _initialize_rules(self) -> List[ScreeningRule]:
        """初始化筛选规则"""
        rules = []

        # 1. 基本规则
        rules.append(ScreeningRule(
            name="价格合理性",
            condition_func=self._check_price_range,
            weight=1.0,
            description="价格在合理范围内（避免仙股和异常高价）"
        ))

        rules.append(ScreeningRule(
            name="流动性充足",
            condition_func=self._check_liquidity,
            weight=1.0,
            description="成交量达到最低要求"
        ))

        # 2. 技术指标规则
        rules.append(ScreeningRule(
            name="趋势向上",
            condition_func=self._check_uptrend,
            weight=1.5,
            description="短期均线上穿长期均线，呈上升趋势"
        ))

        rules.append(ScreeningRule(
            name="RSI合理",
            condition_func=self._check_rsi_range,
            weight=1.2,
            description="RSI不在极端超买或超卖区域"
        ))

        rules.append(ScreeningRule(
            name="突破均线",
            condition_func=self._check_price_above_ma,
            weight=1.3,
            description="价格突破关键均线（如20日均线）"
        ))

        # 3. 波动率规则
        rules.append(ScreeningRule(
            name="波动适中",
            condition_func=self._check_volatility,
            weight=0.8,
            description="波动率在合理范围内，避免暴涨暴跌"
        ))

        # 4. 自定义指标规则（可根据需要添加）
        rules.append(ScreeningRule(
            name="自定义指标1",
            condition_func=self._check_custom_indicator_1,
            weight=1.0,
            description="自定义指标1在设定范围内"
        ))

        return rules

    def apply_screening(self,
                        stock_data: Dict[str, pd.DataFrame],
                        rule_names: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        应用筛选规则到股票数据

        参数:
            stock_data: 股票数据字典，键为代码，值为DataFrame
            rule_names: 指定使用的规则名称列表，None表示使用所有规则

        返回:
            筛选结果字典
        """
        results = {}

        # 过滤要使用的规则
        if rule_names is not None:
            active_rules = [r for r in self.rules if r.name in rule_names]
        else:
            active_rules = self.rules

        logger.info(f"开始筛选 {len(stock_data)} 只股票，使用 {len(active_rules)} 条规则")

        for symbol, df in stock_data.items():
            try:
                # 检查数据是否足够
                if len(df) < self.config['min_days']:
                    results[symbol] = {
                        'selected': False,
                        'score': 0,
                        'reasons': ['数据不足'],
                        'details': {}
                    }
                    continue

                # 计算技术指标（如果不存在）
                df_with_indicators = self._ensure_indicators(df.copy())

                # 应用每条规则
                rule_results = {}
                total_score = 0
                max_possible_score = 0
                passed_rules = []
                failed_rules = []

                for rule in active_rules:
                    try:
                        rule_passed, rule_score, rule_details = rule.condition_func(
                            df_with_indicators, symbol
                        )

                        rule_results[rule.name] = {
                            'passed': rule_passed,
                            'score': rule_score * rule.weight,
                            'details': rule_details
                        }

                        if rule_passed:
                            total_score += rule_score * rule.weight
                            passed_rules.append(rule.name)
                        else:
                            failed_rules.append(rule.name)

                        max_possible_score += rule.weight

                    except Exception as e:
                        logger.warning(f"规则 {rule.name} 应用失败 ({symbol}): {e}")
                        rule_results[rule.name] = {
                            'passed': False,
                            'score': 0,
                            'details': {'error': str(e)}
                        }

                # 计算最终得分（归一化到0-100）
                normalized_score = 0
                if max_possible_score > 0:
                    normalized_score = (total_score / max_possible_score) * 100

                # 判断是否选中（可根据需要调整阈值）
                selected = normalized_score >= 60  # 得分60以上为选中

                # 记录结果
                results[symbol] = {
                    'selected': selected,
                    'score': normalized_score,
                    'passed_rules': passed_rules,
                    'failed_rules': failed_rules,
                    'total_rules': len(active_rules),
                    'passed_count': len(passed_rules),
                    'details': rule_results,
                    'reasons': self._generate_reasons(passed_rules, failed_rules, normalized_score)
                }

                logger.debug(f"{symbol}: 得分 {normalized_score:.1f}, 选中: {selected}")

            except Exception as e:
                logger.error(f"筛选股票 {symbol} 时出错: {e}")
                results[symbol] = {
                    'selected': False,
                    'score': 0,
                    'reasons': [f'处理错误: {str(e)}'],
                    'details': {}
                }

        # 按得分排序
        sorted_results = dict(sorted(
            results.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        ))

        self.results_cache = sorted_results.copy()

        # 记录筛选统计
        selected_count = sum(1 for r in sorted_results.values() if r['selected'])
        logger.info(f"筛选完成: 共分析 {len(sorted_results)} 只股票，选中 {selected_count} 只")

        return sorted_results

    def filter_by_custom_indicators(self,
                                    stock_data: Dict[str, pd.DataFrame],
                                    indicator_filters: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        根据自定义指标进行筛选（核心功能：灵活组合自定义指标）

        参数:
            stock_data: 股票数据
            indicator_filters: 指标过滤条件，格式为:
                {
                    'indicator_name': {
                        'min': 最小值,
                        'max': 最大值,
                        'condition': '>'  # 或 '<', '==', 'between' 等
                    }
                }

        返回:
            符合条件的数据字典
        """
        filtered_data = {}

        for symbol, df in stock_data.items():
            try:
                passed_all = True

                for indicator_name, filter_cond in indicator_filters.items():
                    if indicator_name not in df.columns:
                        logger.warning(f"指标 {indicator_name} 不存在于 {symbol} 数据中")
                        passed_all = False
                        break

                    # 获取最新指标值
                    latest_value = df[indicator_name].iloc[-1]

                    # 应用过滤条件
                    condition = filter_cond.get('condition', 'between')

                    if condition == '>':
                        if not (latest_value > filter_cond.get('min', -np.inf)):
                            passed_all = False
                            break
                    elif condition == '<':
                        if not (latest_value < filter_cond.get('max', np.inf)):
                            passed_all = False
                            break
                    elif condition == 'between':
                        min_val = filter_cond.get('min', -np.inf)
                        max_val = filter_cond.get('max', np.inf)
                        if not (min_val <= latest_value <= max_val):
                            passed_all = False
                            break
                    elif condition == 'outside':
                        min_val = filter_cond.get('min', -np.inf)
                        max_val = filter_cond.get('max', np.inf)
                        if min_val <= latest_value <= max_val:
                            passed_all = False
                            break

                if passed_all:
                    filtered_data[symbol] = df

            except Exception as e:
                logger.error(f"自定义指标筛选 {symbol} 时出错: {e}")

        logger.info(f"自定义指标筛选: 从 {len(stock_data)} 只中筛选出 {len(filtered_data)} 只")
        return filtered_data

    def get_top_stocks(self,
                       results: Dict[str, Dict[str, Any]],
                       top_n: int = 10,
                       min_score: float = 0) -> List[str]:
        """
        获取排名靠前的股票

        参数:
            results: 筛选结果
            top_n: 返回前N名
            min_score: 最低得分要求

        返回:
            股票代码列表
        """
        qualified = [
            (symbol, info['score'])
            for symbol, info in results.items()
            if info['selected'] and info['score'] >= min_score
        ]

        # 按得分排序
        qualified.sort(key=lambda x: x[1], reverse=True)

        top_stocks = [symbol for symbol, _ in qualified[:top_n]]

        logger.info(f"获取前{top_n}名股票 (最低得分{min_score}): {len(top_stocks)}只")
        return top_stocks

    def generate_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """生成筛选报告"""
        if not results:
            return "没有筛选结果"

        selected = [s for s, info in results.items() if info['selected']]

        report_lines = [
            "=" * 60,
            "股票筛选报告",
            "=" * 60,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"分析股票数量: {len(results)}",
            f"选中股票数量: {len(selected)}",
            f"选中比例: {len(selected) / len(results) * 100:.1f}%",
            "\n选中股票详情:"
        ]

        for i, symbol in enumerate(selected[:20], 1):  # 最多显示20只
            info = results[symbol]
            report_lines.append(
                f"{i:2d}. {symbol}: 得分 {info['score']:.1f}, "
                f"通过规则 {info['passed_count']}/{info['total_rules']}"
            )

        # 添加规则通过率统计
        if selected:
            report_lines.append("\n规则通过率统计:")
            rule_stats = {}
            for symbol, info in results.items():
                for rule_name, rule_info in info.get('details', {}).items():
                    if rule_name not in rule_stats:
                        rule_stats[rule_name] = {'passed': 0, 'total': 0}
                    rule_stats[rule_name]['total'] += 1
                    if rule_info.get('passed', False):
                        rule_stats[rule_name]['passed'] += 1

            for rule_name, stats in rule_stats.items():
                rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
                report_lines.append(f"  {rule_name}: {rate:.1f}% ({stats['passed']}/{stats['total']})")

        report_lines.append("\n" + "=" * 60)

        return "\n".join(report_lines)

    # ========== 规则条件函数 ==========

    def _check_price_range(self, df: pd.DataFrame, symbol: str) -> tuple:
        """检查价格范围"""
        latest_close = df['close'].iloc[-1]
        min_price = self.config['min_price']
        max_price = self.config['max_price']

        passed = min_price <= latest_close <= max_price
        score = 1.0 if passed else 0.0
        details = {
            'price': latest_close,
            'min_allowed': min_price,
            'max_allowed': max_price
        }

        return passed, score, details

    def _check_liquidity(self, df: pd.DataFrame, symbol: str) -> tuple:
        """检查流动性"""
        avg_volume = df['volume'].tail(20).mean()
        min_volume = self.config['min_volume']

        passed = avg_volume >= min_volume
        score = min(1.0, avg_volume / min_volume)  # 成交量越大得分越高
        details = {
            'avg_volume': avg_volume,
            'min_required': min_volume
        }

        return passed, score, details

    def _check_uptrend(self, df: pd.DataFrame, symbol: str) -> tuple:
        """检查上升趋势"""
        # 确保均线存在
        df = self._ensure_indicators(df)

        # 检查短期均线是否在长期均线之上
        if 'ma_5' in df.columns and 'ma_20' in df.columns:
            latest_ma5 = df['ma_5'].iloc[-1]
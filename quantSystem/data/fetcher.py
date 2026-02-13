## 数据引入fetcher
import pandas as pd
import akshare as ak
from pathlib import Path
import logging
from typing import Optional, Union

class StockDataFetcher:
    """股票数据获取器 (使用AkShare)"""
    def __init__(self, cache_dir: str = "./data/cache"):
        """
        初始化获取器
        :param cache_dir: 本地缓存目录，用于存储已下载的数据
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)  # 创建缓存目录
        self.logger = logging.getLogger(__name__)

    def _get_cache_path(self, symbol: str, period: str) -> Path:
        """生成缓存文件路径"""
        return self.cache_dir / f"{symbol}_{period}.parquet"

    def get_daily_data(self,
                       symbol: str,
                       period: str = "daily",
                       adjust: str = "hfq",
                       use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        获取股票日线数据 (核心方法)
        :param symbol: 股票代码，如 "000001" (深交所) 或 "600000" (上交所)
        :param period: 周期，默认日线 "daily"，可选 "weekly"， "monthly"
        :param adjust: 复权类型，"hfq" (后复权)， "qfq" (前复权)， "" (不复权)
        :param use_cache: 是否使用本地缓存
        :return: 包含OHLCV等数据的DataFrame，索引为datetime
        """
        # 1. 检查缓存
        cache_file = self._get_cache_path(symbol, period)
        if use_cache and cache_file.exists():
            self.logger.info(f"读取缓存数据: {symbol}")
            try:
                df = pd.read_parquet(cache_file)
                # 确保索引是时间类型
                df.index = pd.to_datetime(df.index)
                return df
            except Exception as e:
                self.logger.warning(f"读取缓存失败，重新下载: {e}")

        # 2. 调用AkShare API获取数据
        self.logger.info(f"下载数据: {symbol}, 周期: {period}, 复权: {adjust}")
        try:
            # 这里以A股历史行情接口为例，请根据你的需求查阅AkShare文档选择最合适的接口
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                adjust=adjust
            )
        except Exception as e:
            self.logger.error(f"从AkShare获取数据失败: {e}")
            return None

        # 3. 数据清洗与格式化 (关键！)
        # 重命名列，统一为英文小写，方便后续处理
        column_map = {
            '日期': 'date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'turnover',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover_rate'
        }
        df.rename(columns=column_map, inplace=True)

        # 设置日期索引并排序
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)  # 确保时间升序

        # 4. 缓存到本地 (使用parquet格式，高效且节省空间)
        if use_cache:
            try:
                df.to_parquet(cache_file)
                self.logger.info(f"数据已缓存至: {cache_file}")
            except Exception as e:
                self.logger.warning(f"数据缓存失败: {e}")

        return df

    def get_batch_daily_data(self,
                             symbols: list,
                             **kwargs) -> dict:
        """
        批量获取多只股票数据
        :param symbols: 股票代码列表
        :param kwargs: 传递给get_daily_data的其他参数
        :return: 字典，键为股票代码，值为对应的DataFrame
        """
        results = {}
        for sym in symbols:
            df = self.get_daily_data(sym, **kwargs)
            if df is not None and not df.empty:
                results[sym] = df
            else:
                self.logger.warning(f"股票 {sym} 数据获取失败，已跳过")
        return results


# 提供一个全局实例，方便直接导入使用 (单例模式思想)
default_fetcher = StockDataFetcher()

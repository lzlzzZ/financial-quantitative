from quantSystem.data.fetcher import default_fetcher,StockDataFetcher

class TestEnv:


    df_000001 = default_fetcher.get_daily_data("000001", adjust="hfq")
    if df_000001 is not None:
        print(f"获取到 {len(df_000001)} 行数据")
        print(df_000001[['open', 'close', 'volume']].head())
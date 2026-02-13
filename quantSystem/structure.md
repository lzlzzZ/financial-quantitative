quant_system/
├── data/           # 数据层
│   ├── fetcher.py    # 封装 AkShare 数据获取
│   └── database.py   # 数据缓存与管理（使用mongoDB）
├── analysis/       # 计算层
│   ├── calculator.py  # 指标计算（内置、自定义）
│   └── screener.py    # 选股逻辑封装
├── visualization/  # 可视化层
│   └── plotter.py     # 基于 mplchart 的绘图函数
├── strategy/       # 策略层（预留）
│   ├── models.py      # 预测模型
│   └── backtest.py    # 回测引擎
├── config.py       # 全局配置（如API密钥、股票池）
└── main.py         # 主程序，串联工作流
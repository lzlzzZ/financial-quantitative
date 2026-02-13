#!/usr/bin/env python3
"""
交互式量化系统主程序
运行后输入命令进行操作，例如：plot 000001, fetch 000001, screen, exit
"""
import cmd  # Python内置的命令行模块
import sys
import logging
from pathlib import Path

# 导入你的自定义模块
sys.path.append(str(Path(__file__).parent))
from data.fetcher import StockDataFetcher
from visualization.plotter import ChartPlotter
from analysis.screener import StockScreener


class QuantCommandInterpreter(cmd.Cmd):
    """量化系统命令解释器"""

    prompt = '(QuantSys) > '  # 命令行提示符

    def __init__(self):
        super().__init__()
        self.fetcher = StockDataFetcher()
        self.plotter = ChartPlotter()
        self.screener = StockScreener()
        self.current_data = {}  # 缓存当前数据
        print("个人量化交易系统 (交互模式) 已启动")
        print("输入 'help' 查看可用命令，'exit' 退出")

    def do_fetch(self, arg):
        """获取单只或多只股票数据: fetch 000001 或 fetch 000001,000858"""
        if not arg:
            print("错误: 请指定股票代码，例如: fetch 000001")
            return

        symbols = [s.strip() for s in arg.split(',')]
        for symbol in symbols:
            try:
                print(f"正在获取 {symbol} 的数据...")
                df = self.fetcher.get_daily_data(symbol=symbol, adjust="qfq")
                if df is not None:
                    self.current_data[symbol] = df
                    print(f"✅ {symbol}: 获取成功，共 {len(df)} 条数据")
                    print(f"   最新日期: {df.index[-1].strftime('%Y-%m-%d')}, 收盘价: {df['close'].iloc[-1]:.2f}")
                else:
                    print(f"❌ {symbol}: 获取失败")
            except Exception as e:
                print(f"❌ {symbol}: 获取时出错 - {e}")

    def do_plot(self, arg):
        """绘制股票图表: plot 000001 (可加指标参数: plot 000001 -i ma,rsi)"""
        if not arg:
            print("错误: 请指定股票代码，例如: plot 000001")
            return

        # 简单解析参数
        args = arg.split()
        symbol = args[0]
        indicators = ['ma', 'volume']  # 默认指标

        if len(args) > 2 and args[1] == '-i':
            indicators = args[2].split(',')

        if symbol not in self.current_data:
            print(f"警告: {symbol} 数据未加载，正在尝试获取...")
            self.do_fetch(symbol)
            if symbol not in self.current_data:
                return

        try:
            print(f"正在为 {symbol} 生成图表，指标: {indicators}")
            # 这里调用你的绘图模块
            self.plotter.plot_stock_with_indicators(
                self.current_data[symbol],
                symbol=symbol,
                indicators=indicators,
                save_path=f"./output/{symbol}_chart.png"
            )
            print(f"✅ 图表已生成: ./output/{symbol}_chart.png")
            # 可以添加自动打开图片的代码
            # import os; os.startfile(f"./output/{symbol}_chart.png")  # Windows
        except Exception as e:
            print(f"❌ 绘图失败: {e}")

    def do_screen(self, arg):
        """执行股票筛选: screen (可使用参数: screen -p 000001,000002)"""
        # 如果指定了股票池，先获取数据
        if arg and arg.startswith('-p '):
            symbols = arg[3:].split(',')
            for symbol in symbols:
                if symbol not in self.current_data:
                    self.do_fetch(symbol)

        try:
            print("正在执行筛选...")
            if not self.current_data:
                print("没有可用数据，请先使用 fetch 命令获取数据")
                return

            results = self.screener.apply_screening(self.current_data)
            selected = [k for k, v in results.items() if v['selected']]

            print(f"筛选完成: 共分析 {len(results)} 只股票")
            if selected:
                print("✅ 选中股票:")
                for symbol in selected:
                    print(f"  • {symbol}: {results[symbol].get('reason', '符合条件')}")
            else:
                print("⚠️  没有股票符合筛选条件")
        except Exception as e:
            print(f"❌ 筛选失败: {e}")

    def do_show(self, arg):
        """显示当前缓存中的数据: show 或 show 000001"""
        if arg:
            if arg in self.current_data:
                df = self.current_data[arg]
                print(f"\n{arg} 数据概要:")
                print(f"时间范围: {df.index[0].date()} 至 {df.index[-1].date()}")
                print(f"数据条数: {len(df)}")
                print("\n最新5条数据:")
                print(df[['open', 'high', 'low', 'close', 'volume']].tail())
            else:
                print(f"没有 {arg} 的数据，请先使用 fetch {arg} 获取")
        else:
            print(f"\n当前缓存中股票: {list(self.current_data.keys())}")
            print(f"总计: {len(self.current_data)} 只股票")

    def do_exit(self, arg):
        """退出系统"""
        print("正在退出量化系统...")
        return True  # 返回True表示退出循环

    def do_help(self, arg):
        """显示帮助信息"""
        commands = {
            'fetch': '获取股票数据: fetch <代码> (多个代码用逗号分隔)',
            'plot': '绘制股票图表: plot <代码> [-i 指标列表]',
            'screen': '执行股票筛选: screen [-p 股票代码列表]',
            'show': '显示缓存数据: show [股票代码]',
            'exit': '退出系统'
        }
        print("\n可用命令:")
        for cmd, desc in commands.items():
            print(f"  {cmd:10} {desc}")
        print()


# 简化的直接交互版本（不使用cmd模块）
def simple_interactive_loop():
    """一个更简单的交互循环，适合快速实现"""
    fetcher = StockDataFetcher()

    print("量化系统交互模式启动 (简单版)")
    print("命令: fetch <代码>, plot <代码>, screen, exit")

    data_cache = {}

    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input:
                continue

            parts = user_input.split()
            command = parts[0].lower()

            if command == "exit":
                print("退出系统")
                break

            elif command == "fetch" and len(parts) > 1:
                symbol = parts[1]
                print(f"获取 {symbol} 数据...")
                df = fetcher.get_daily_data(symbol=symbol)
                if df is not None:
                    data_cache[symbol] = df
                    print(f"✅ 成功获取 {len(df)} 条数据")
                else:
                    print("❌ 获取失败")

            elif command == "plot" and len(parts) > 1:
                symbol = parts[1]
                if symbol in data_cache:
                    print(f"绘制 {symbol} 图表...")
                    # 调用绘图函数
                    print("✅ 图表已生成")
                else:
                    print(f"请先使用 'fetch {symbol}' 获取数据")

            elif command == "screen":
                print("执行筛选...")
                if data_cache:
                    # 调用筛选函数
                    print("✅ 筛选完成")
                else:
                    print("请先使用 fetch 命令获取数据")

            elif command == "help":
                print("可用命令: fetch <股票代码>, plot <股票代码>, screen, exit")

            else:
                print(f"未知命令: {command}，输入 help 查看帮助")

        except KeyboardInterrupt:
            print("\n用户中断，退出系统")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    # 两种启动方式任选其一：

    # 方式1: 使用完整的cmd框架（更强大，支持自动补全、历史记录等）
    interpreter = QuantCommandInterpreter()
    interpreter.cmdloop()

    # 方式2: 使用简单交互循环（更直观，易于扩展）
    # simple_interactive_loop()

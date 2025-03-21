from marp.marp import PythonMarp
from stockmarkdown import analyze_stock

# Example usage
marp_presentation = PythonMarp(theme="night", classes=["invert", "lead"])
marp_presentation.add_slide("SolidLiquidity", "TL;DR")

stocks = ["AAPL"]

for stock in stocks:
    try:
        print('ok')
        stock_data = analyze_stock(stock, lookback_days=180)
        print('sure')
        marp_presentation.add_slide(stock, stock_data)
        print('fine')

        print("\n" + "-"*70 + "\n")
    except Exception as e:
        print(f"Error analyzing {stock}: {str(e)}")
        continue

marp_presentation.generate_presentation()

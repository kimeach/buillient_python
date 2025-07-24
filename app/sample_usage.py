import pandas as pd
from app.quant_analysis import moving_average, rsi, bollinger_bands


def main(path: str):
    df = pd.read_csv(path)
    df['ma_20'] = moving_average(df, 'close', 20)
    df['rsi_14'] = rsi(df, 'close', 14)
    bb = bollinger_bands(df, 'close', 20, 2)
    df = pd.concat([df, bb.add_prefix('bb_')], axis=1)
    print(df.tail())


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python sample_usage.py <csv_path>")
        raise SystemExit(1)
    main(sys.argv[1])

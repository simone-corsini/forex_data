import argparse
import os
import pandas as pd
import numpy as np

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, Task
from rich.console import Console

class TotalTimeColumn(TextColumn):
    def render(self, task: Task) -> str:
        # Calcola il tempo totale trascorso
        total_time = task.finished_time if task.finished_time else task.elapsed
        # Mostra il tempo totale in formato mm:ss
        return f"[green]{total_time:.2f}s[/green]"

def calculate_shadows(row):
    lower_shadow = min(row['open'], row['close']) - row['low']
    upper_shadow = row['high'] - max(row['open'], row['close'])
    return lower_shadow, upper_shadow

def calculate_body(row):
    return abs(row['open'] - row['close'])

def gap_1(prev_candle, current_candle):
    prev_lower_shadow, prev_upper_shadow = calculate_shadows(prev_candle)
    current_lower_shadow, current_upper_shadow = calculate_shadows(current_candle)
    lower_shadow = round((prev_lower_shadow + current_lower_shadow) / 2, 5)
    upper_shadow = round((prev_upper_shadow + current_upper_shadow) / 2, 5)
    return {
        'timestamp': prev_candle['timestamp'] + pd.Timedelta(minutes=1), 
        'open': prev_candle['close'], 
        'high': max(prev_candle['close'], current_candle['open']) + upper_shadow,
        'low': min(prev_candle['close'], current_candle['open']) - lower_shadow,
        'close': current_candle['open'], 
        'volume': min(prev_candle['volume'],current_candle['volume']),
        'spread': max(prev_candle['spread'], current_candle['spread'])
    }

def gap_2(prev_candle, current_candle):
    prev_lower_shadow, prev_upper_shadow = calculate_shadows(prev_candle)
    current_lower_shadow, current_upper_shadow = calculate_shadows(current_candle)
    lower_shadow = round((prev_lower_shadow + current_lower_shadow) / 2, 5)
    upper_shadow = round((prev_upper_shadow + current_upper_shadow) / 2, 5)
    prev_body = calculate_body(prev_candle)
    current_body = calculate_body(current_candle)
    body = round((prev_body + current_body) / 2, 5)
    gap_1_open = prev_candle['close']
    gap_2_close = current_candle['open']
    gap_1_close = gap_2_open = gap_1_open + body if gap_1_open < gap_2_close else gap_1_open - body
    volume = min(prev_candle['volume'],current_candle['volume'])
    spread = max(prev_candle['spread'], current_candle['spread'])
    return {
        'timestamp': prev_candle['timestamp'] + pd.Timedelta(minutes=1), 
        'open': gap_1_open, 
        'high': max(gap_1_open, gap_1_close) + upper_shadow,
        'low': min(gap_1_open, gap_1_close) - lower_shadow,
        'close': gap_1_close, 
        'volume': volume,
        'spread': spread
    }, {
        'timestamp': prev_candle['timestamp'] + pd.Timedelta(minutes=2), 
        'open': gap_2_open, 
        'high': max(gap_2_open, gap_2_close) + upper_shadow,
        'low': min(gap_2_open, gap_2_close) - lower_shadow,
        'close': gap_2_close, 
        'volume': volume,
        'spread': spread
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare train data', add_help=True)
    parser.add_argument("-i", "--input", type=str, help="Input forex file", default='./data/EURUSD_M1_ohcl.csv')
    parser.add_argument("-o", "--output", type=str, help="Output for forex files", default='./data/corrected')

    args = parser.parse_args()

    base_path = args.output
    os.makedirs(base_path, exist_ok=True)

    console = Console()
    progress = Progress(
            TextColumn("[progress.description]"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            TotalTimeColumn("[green][progress.elapsed]{task.elapsed:>5.2f}s[/green]"),
            TextColumn("{task.description}"),
            console=console
        )

    args = parser.parse_args()
    df = pd.read_csv(args.input, nrows=100000)

    df.drop(columns=['tick_volume'], inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    prev_candle = None
    file_task = progress.add_task(f"Process main file", total=df.shape[0])
    reconstructed = []

    with progress:
        for i, row in df.iterrows():
            if prev_candle is not None:
                gap = (row['timestamp'] - prev_candle['timestamp']).total_seconds() / 60

                if gap == 1:
                    reconstructed.append(row.to_dict())
                    prev_candle = row
                elif gap == 2:
                    reconstructed.append(gap_1(prev_candle, row))
                    reconstructed.append(row.to_dict())
                    prev_candle = row
                elif gap == 3:
                    gap_1_candle, gap_2_candle = gap_2(prev_candle, row)
                    reconstructed.append(gap_1_candle)
                    reconstructed.append(gap_2_candle)
                    reconstructed.append(row.to_dict())
                    prev_candle = row
                else:
                    if len(reconstructed) > 0:
                        new_df = pd.DataFrame(reconstructed)
                        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])

                        new_df = pd.DataFrame(reconstructed)
                        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
                        new_df = new_df.sort_values(by='timestamp').reset_index(drop=True)
                        if new_df.isna().sum().sum() > 0 or np.isinf(new_df).values.sum() > 0:
                            print('Valori non validi nei dati => NaN o Inf')
                        else:
                            first_date = new_df['timestamp'].iloc[0].strftime('%Y%m%d%H%M')
                            last_date = new_df['timestamp'].iloc[-1].strftime('%Y%m%d%H%M')
                            file_name = os.path.join(base_path, f'data_{first_date}_{last_date}.csv')
                            new_df.to_csv(file_name, index=False)

                    reconstructed = []
                    prev_candle = None
                    continue
            else:
                prev_candle = row

            progress.update(file_task, advance=1)

        if len(reconstructed) > 0:
            new_df = pd.DataFrame(reconstructed)
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
            
            if new_df.isna().sum().sum() > 0 or np.isinf(new_df).values.sum() > 0:
                progress.console.print('Valori non validi nei dati => NaN o Inf')
            else:
                first_date = new_df['timestamp'].iloc[0].strftime('%Y%m%d%H%M')
                last_date = new_df['timestamp'].iloc[-1].strftime('%Y%m%d%H%M')
                file_name = os.path.join(base_path, f'data_{first_date}_{last_date}.csv')
                new_df.to_csv(file_name, index=False)

        progress.update(file_task, completed=True)
        progress.stop()
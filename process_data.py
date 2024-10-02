import argparse
import pandas as pd
import numpy as np
import os
import shutil
import glob
import zipfile
import h5py
from collections import defaultdict
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, Task
from rich.console import Console

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

def calculate_category(row, price_col, spread_col, targets, base_commission=0.00007):
    if pd.isna(row[price_col]) or pd.isna(row[spread_col]):
        return np.nan

    price_change = row[price_col]
    total_commission = base_commission + row[spread_col]

    adjusted_targets = [(target * 0.0001) + total_commission for target in targets]
    
    if -adjusted_targets[0] <= price_change <= adjusted_targets[0]:
        return 0

    for i in range(1, len(adjusted_targets)):
        if adjusted_targets[i - 1] < price_change <= adjusted_targets[i]:
            return i

    if price_change > adjusted_targets[-1]:
        return len(adjusted_targets)

    for i in range(1, len(adjusted_targets)):
        if -adjusted_targets[i] <= price_change < -adjusted_targets[i - 1]:
            return len(adjusted_targets) + i

    if price_change < -adjusted_targets[-1]:
        return len(adjusted_targets) * 2

    return np.nan

def max_spread(series, n=10):
    spreads = []
    for i in range(len(series)):
        if i + n >= len(series):
            spreads.append(np.nan)
        else:
            diffs = series[i] - series[i+1:i+n+1]
            diffs = diffs.to_numpy()
            max_abs_diff = diffs[np.argmax(np.abs(diffs))]
            spreads.append(max_abs_diff)
    return spreads

def mean_spread(series, n=10):
    spreads = []
    for i in range(len(series)):
        if i + n >= len(series):
            spreads.append(np.nan)
        else:
            diffs = series[i] - series[i+1:i+n+1]
            spreads.append(diffs.mean())
    return spreads

def prepare_data(df_original, 
                 slow_sma, fast_sma, slow_ema, middle_ema, fast_ema, 
                 bollinger_sma, bollinger_deviation, 
                 spread_sma, volume_sma, 
                 future_lenght, targets, target_type, base_commission=0.00007):
    max_window = max(slow_sma, fast_sma, slow_ema, middle_ema, fast_ema, bollinger_sma, spread_sma, volume_sma)

    df = df_original.copy()

    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df['close_diff'] = df['close'].diff()
    df['body'] = df['close'] - df['open']
    df['lower_shadow'] = df.apply(lambda row: min(row['close'], row['open']) - row['low'], axis=1) 
    df['upper_shadow'] = df.apply(lambda row: row['high'] - max(row['close'], row['open']), axis=1) 
    df['slow_sma'] = df['close_diff'].rolling(window=slow_sma).mean()
    df['fast_sma'] = df['close_diff'].rolling(window=fast_sma).mean()
    df['sma_spread'] = df['fast_sma'] - df['slow_sma']
    df['slow_ema'] = df['close_diff'].ewm(span=slow_ema, adjust=False).mean()
    df['middle_ema'] = df['close_diff'].ewm(span=middle_ema, adjust=False).mean()
    df['fast_ema'] = df['close_diff'].ewm(span=fast_ema, adjust=False).mean()
    df['ema_slow_fast_spread'] = df['fast_ema'] - df['slow_ema']
    df['ema_slow_middle_spread'] = df['middle_ema'] - df['slow_ema']
    df['ema_middle_fast_spread'] = df['fast_ema'] - df['middle_ema']
    df['bb_sma'] = df['close_diff'].rolling(window=bollinger_sma).mean()
    df['bb_dev'] = df['close_diff'].rolling(window=bollinger_sma).std()
    df['bb_upper'] = df['bb_sma'] + (bollinger_deviation * df['bb_dev'])
    df['bb_lower'] = df['bb_sma'] - (bollinger_deviation * df['bb_dev'])
    df['bb_upper_spread'] = df['bb_upper'] - df['close_diff']
    df['bb_lower_spread'] = df['close_diff'] - df['bb_lower']
    df['spread'] = df['spread'] * 0.00001
    df['spread_sma'] = df['spread'].rolling(window=spread_sma).mean()
    df['spread_spread'] = df['spread'] - df['spread_sma']
    df['volume'] = df['volume'] * 0.00001
    df['volume_sma'] = df['volume'].rolling(window=volume_sma).mean()
    df['volume_spread'] = df['volume'] - df['volume_sma']
    df['next_price_range_value_mean'] = mean_spread(df['close'], future_lenght)
    #df['next_price_range_value_mean'] = df['close_diff'].shift(-future_lenght).rolling(window=future_lenght).mean() 
    df['next_price_range_value_max'] = max_spread(df['close'], future_lenght)
    df['label'] = df.apply(lambda row: calculate_category(row, f'next_price_range_value_{target_type}', 'spread', targets, base_commission), axis=1)
    #df['label'] = df['label'].astype('Int64')

    df = df[max_window:-future_lenght]

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread',
            'close_diff', 'body', 'lower_shadow', 'upper_shadow',
            'sma_spread', 'ema_slow_fast_spread', 'ema_slow_middle_spread', 'ema_middle_fast_spread',
            'bb_upper_spread', 'bb_lower_spread',
            'spread_spread', 'volume_spread', 
            'next_price_range_value_mean', 'next_price_range_value_max',
            'label']]

    return df

def zip_files(folder_path, zip_name, extension):
    # Crea un file ZIP
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith(extension):
                    file_path = os.path.join(foldername, filename)
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)


def view_file_sturcture(datafile):
    with h5py.File(datafile, 'r') as file:
        for key in file.keys():
            print(f'{key}: {file[key].shape}')

def create_dataset_file(datafile, observation_length, features, labels):
    with h5py.File(datafile, 'w') as file:
        for label in labels:
            file.create_dataset(label, shape=(0, observation_length, features), maxshape=(None, observation_length, features), dtype='float32')

        file.create_dataset('X_train', shape=(0, observation_length, features), maxshape=(None, observation_length, features), dtype='float32')
        file.create_dataset('X_val', shape=(0, observation_length, features), maxshape=(None, observation_length, features), dtype='float32')
        file.create_dataset('X_test', shape=(0, observation_length, features), maxshape=(None, observation_length, features), dtype='float32')

        file.create_dataset('y_train', shape=(0, 1), maxshape=(None, 1), dtype='int')
        file.create_dataset('y_val', shape=(0, 1), maxshape=(None, 1), dtype='int')
        file.create_dataset('y_test', shape=(0, 1), maxshape=(None, 1), dtype='int')

def add_sample_to_dataset(datafile, labels, X_sets):
    with h5py.File(datafile, 'a') as file:
        for label in labels:
            if len(X_sets[label]) > 0:
                X_dset = file[label]
                X = np.array(X_sets[label], dtype='float32')
                X_dset.resize(X_dset.shape[0] + X.shape[0], axis=0)
                X_dset[-X.shape[0]:] = X   

def prepare_dataset(datafile, progress, base_path, phase, observation_length, features, labels):
    X_sets = defaultdict(list)

    files = glob.glob(f'{base_path}/{phase}/*.csv')
    file_task = progress.add_task(f"Process files phase {phase}", total=len(files))
    for file in files:
        df = pd.read_csv(file, parse_dates=['timestamp'], index_col='timestamp')
        label_values = df['label'].values
        df = df[features]

        for i in range(len(df) - (observation_length - 1)):
            label = str(int(label_values[i + observation_length - 1]))
            X_sets[label].append(df.iloc[i:i+observation_length].values)

        add_sample_to_dataset(datafile, labels, X_sets)

        for label in labels:
            X_sets[label].clear()

        progress.update(file_task, advance=1)

    #progress.update(file_task, completed=True)

def clear_label_samples(datafile, labels):
    with h5py.File(datafile, 'a') as file:
        for label in labels:
            dset = file[label]
            dset.resize((0,) + dset.shape[1:])

def drop_label_samples(datafile, labels):
    with h5py.File(datafile, 'a') as file:
        for label in labels:
            del file[label]     

def prepare_balanced_dataset(datafile, progress, phase, labels, batch_size=10000):
    with h5py.File(datafile, 'a') as file:
        progress.console.print(f'[green]Balancing dataset for {phase}[/green]')
        for label in labels:
            progress.console.print(f'[green]\t{label}=>{file[label].shape[0]}[/green]')

        min_samples = min([file[label].shape[0] for label in labels])

        progress.console.print(f'[green]\tMinimum samples: {min_samples}[/green]')
        progress.console.print(f'[green]\tTotal samples: {min_samples * len(labels)}[/green]')

        indexes = {str(label): list(np.random.choice(file[label].shape[0], size=min_samples, replace=False)) for label in labels}
        file_task = progress.add_task(f"Process balancing phase {phase}", total=min_samples * len(labels))

        # Batch accumulators
        X_batch = []
        y_batch = []

        while not all(len(lst) == 0 for lst in indexes.values()):
            label = str(np.random.choice(labels))
            if len(indexes[label]) > 0:
                index = indexes[label].pop()

                X = file[label][index]
                y = np.array([label], dtype='int')

                X_batch.append(X)
                y_batch.append(y)

                # Scrivi i dati quando raggiungi il batch size
                if len(X_batch) >= batch_size:
                    # Resize una sola volta e aggiungi tutto il batch
                    file[f'X_{phase}'].resize(file[f'X_{phase}'].shape[0] + len(X_batch), axis=0)
                    file[f'X_{phase}'][-len(X_batch):] = X_batch

                    file[f'y_{phase}'].resize(file[f'y_{phase}'].shape[0] + len(y_batch), axis=0)
                    file[f'y_{phase}'][-len(y_batch):] = y_batch

                    # Svuota i batch
                    X_batch.clear()
                    y_batch.clear()

                progress.update(file_task, advance=1, description=f'Balanced dataset for {phase} => {file[f'X_{phase}'].shape} {file[f'y_{phase}'].shape}')

        # Scrivi i dati rimanenti nel batch (se ne sono rimasti)
        if X_batch:
            file[f'X_{phase}'].resize(file[f'X_{phase}'].shape[0] + len(X_batch), axis=0)
            file[f'X_{phase}'][-len(X_batch):] = X_batch

            file[f'y_{phase}'].resize(file[f'y_{phase}'].shape[0] + len(y_batch), axis=0)
            file[f'y_{phase}'][-len(y_batch):] = y_batch

        progress.update(file_task, description=f'Balanced dataset for {phase} => {file[f'X_{phase}'].shape} {file[f'y_{phase}'].shape}')
        

class TotalTimeColumn(TextColumn):
    def render(self, task: Task) -> str:
        # Calcola il tempo totale trascorso
        total_time = task.finished_time if task.finished_time else task.elapsed
        # Mostra il tempo totale in formato mm:ss
        return f"[green]{total_time:.2f}s[/green]"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare train data', add_help=True)
    parser.add_argument("-i", "--input", type=str, help="Input forex file", default='./data/EURUSD_M1_ohcl.csv')
    parser.add_argument("-o", "--output", type=str, help="Output folder", default='./data/train_sets')
    parser.add_argument("-fl", "--future_length", type=int, help="Number of files to process", default=10)
    parser.add_argument("-ol", "--observation_length", type=int, help="Number of files to process", default=180)
    parser.add_argument("-mfl", "--min_file_length", type=int, help="Minimum sample per file", default=1440)
    parser.add_argument("--val_percent", type=float, help="Validation percent", default=0.2)
    parser.add_argument("--test_percent", type=float, help="Test percent", default=0)
    parser.add_argument("-t", "--targets", nargs="+", type=int, default=[1, 3, 6], help="List of targets (--targets 1 2 3), default=[1, 3, 6]")
    parser.add_argument("-tt", "--target_type", choices=['mean', 'max'], default='mean', help="Type of target to calculate (mean, max), default=mean")
    parser.add_argument('--test', action='store_true', help="Lancia in test mode")
    args = parser.parse_args()

    base_path = args.output
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

    progress.console.print('[green]Caricamento dati[/green]')

    if args.test:
        df = pd.read_csv(args.input, nrows=10000)
    else:
        df = pd.read_csv(args.input)
    os.makedirs(f'{base_path}/processed', exist_ok=True)
    os.makedirs(f'{base_path}/test', exist_ok=True)
    os.makedirs(f'{base_path}/val', exist_ok=True)
    os.makedirs(f'{base_path}/train', exist_ok=True)

    df.drop(columns=['tick_volume'], inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    data_files = []
    min_file_lenght = args.min_file_length

    test_percentage = args.test_percent
    validation_percentage = args.val_percent

    slow_sma = 100
    fast_sma = 50
    slow_ema = 9
    middle_ema = 21 
    fast_ema = 55 
    bollinger_sma = 10
    bollinger_deviation = 2.5 
    spread_sma = 50
    volume_sma = 50
    future_lenght = 10
    targets = args.targets
    targets_string = '_'.join(map(str, targets))

    future_length = args.future_length
    observation_lenght = args.observation_length

    base_file_output_name = f'o{observation_lenght}_f{future_length}_t{targets_string}'
    datafile = f'{base_path}/set_{base_file_output_name}.h5'
    phases = ['train', 'val', 'test']
    labels = [str(i) for i in range(len(targets) * 2 + 1)]

    max_window = max(slow_sma, fast_sma, slow_ema, middle_ema, fast_ema, bollinger_sma, spread_sma, volume_sma)

    features = ['close_diff', 'body', 'lower_shadow', 'upper_shadow',
                'sma_spread', 'ema_slow_fast_spread', 'ema_slow_middle_spread', 'ema_middle_fast_spread',
                'bb_upper_spread', 'bb_lower_spread',
                'spread_spread', 'volume_spread']
    min_length = 1 + max_window + observation_lenght + future_length
    reconstructed = []
    prev_candle = None

    file_task = progress.add_task(f"Process main file", total=df.shape[0])
    global_class_distribution = defaultdict(int)

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
                    if len(reconstructed) >= min_length:
                        new_df = pd.DataFrame(reconstructed)
                        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
                        new_df = prepare_data(new_df,
                                                slow_sma, fast_sma, slow_ema, middle_ema, fast_ema, 
                                                bollinger_sma, bollinger_deviation, 
                                                spread_sma, volume_sma, 
                                                future_lenght, targets, args.target_type)
                        if new_df.isna().sum().sum() > 0 or np.isinf(new_df).values.sum() > 0:
                            print('Valori non validi nei dati => NaN o Inf')
                        elif new_df[features].apply(lambda x: (x < -1) | (x > 1)).any().any():
                            progress.console.print('Valori non validi nei dati => out of range')
                            out_values = new_df[features].apply(lambda x: (x < -1) | (x > 1))
                            out_of_range_rows = new_df[out_values.any(axis=1)][features]
                            progress.console.print(out_of_range_rows.values)
                            exit()
                        else:
                            if new_df.shape[0] >= observation_lenght:
                                first_date = new_df['timestamp'].iloc[0].strftime('%Y%m%d%H%M')
                                last_date = new_df['timestamp'].iloc[-1].strftime('%Y%m%d%H%M')
                                file_name = f'{base_path}/processed/data_{first_date}_{last_date}.csv'
                                class_distribution = new_df['label'].value_counts().to_dict()
                                data_files.append({
                                    'file_name': file_name,
                                    'length': new_df.shape[0],
                                    'class_distribution': class_distribution
                                })

                                for class_label, count in class_distribution.items():
                                    global_class_distribution[class_label] += count

                                new_df.to_csv(file_name, index=False)

                    reconstructed = []
                    prev_candle = None
                    continue
            else:
                prev_candle = row

            progress.update(file_task, advance=1, description=f'Process main file => {dict(sorted(global_class_distribution.items()))}')

        if len(reconstructed) >= min_length:
            new_df = pd.DataFrame(reconstructed)
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
            new_df = prepare_data(new_df,
                                    slow_sma, fast_sma, slow_ema, middle_ema, fast_ema, 
                                    bollinger_sma, bollinger_deviation, 
                                    spread_sma, volume_sma, 
                                    future_lenght, targets, args.target_type)
            
            if new_df.isna().sum().sum() > 0 or np.isinf(new_df).values.sum() > 0:
                progress.console.print('Valori non validi nei dati => NaN o Inf')
            elif new_df[features].apply(lambda x: (x < -1) | (x > 1)).any().any():
                progress.console.print('Valori non validi nei dati => out of range')
                out_values = new_df[features].apply(lambda x: (x < -1) | (x > 1))
                out_of_range_rows = new_df[out_values.any(axis=1)][features]
                progress.console.print(out_of_range_rows.values)
                exit()
            else:
                if new_df.shape[0] >= observation_lenght:
                    first_date = new_df['timestamp'].iloc[0].strftime('%Y%m%d%H%M')
                    last_date = new_df['timestamp'].iloc[-1].strftime('%Y%m%d%H%M')
                    file_name = f'{base_path}/processed/data_{first_date}_{last_date}.csv'
                    class_distribution = new_df['label'].value_counts().to_dict()
                    data_files.append({
                        'file_name': file_name,
                        'length': new_df.shape[0],
                        'class_distribution': class_distribution
                    })

                    for class_label, count in class_distribution.items():
                        global_class_distribution[class_label] += count

                    new_df.to_csv(file_name, index=False)
                    progress.update(file_task, advance=1, description=f'Process main file => {dict(sorted(global_class_distribution.items()))}')

        #progress.update(file_task, completed=True)

        data_files_df = pd.DataFrame(data_files)
        num_record = data_files_df['length'].sum()

        min_key = min(global_class_distribution, key=global_class_distribution.get)
        min_value = global_class_distribution[min_key]

        test_target = int(min_value * test_percentage)
        validation_target = int(min_value * validation_percentage)

        # test_target = int(num_record * test_percentage)
        # validation_target = int(num_record * validation_percentage)
        df_test_validation = data_files_df[data_files_df['length'] >= min_file_lenght]
        df_test_validation = df_test_validation.sample(frac=1, random_state=42).reset_index(drop=True)

        current_sum = 0
        selected_rows = []

        file_task = progress.add_task(f"Select train/val/test", total=df_test_validation.shape[0])
        val_sum = 0
        test_sum = 0
        # Itera attraverso il DataFrame casualmente ordinato
        for i, row in df_test_validation.iterrows():
            if current_sum + row['class_distribution'][min_key] <= validation_target:
                base_file_name = os.path.basename(row['file_name'])
                shutil.move(row['file_name'], f'{base_path}/val/{base_file_name}')
                current_sum += row['class_distribution'][min_key]
                val_sum += row['class_distribution'][min_key]
            elif current_sum + row['class_distribution'][min_key] <= validation_target + test_target:
                base_file_name = os.path.basename(row['file_name'])
                shutil.move(row['file_name'], f'{base_path}/test/{base_file_name}')
                current_sum += row['class_distribution'][min_key]
                test_sum += row['class_distribution'][min_key]
            else:
                break

            progress.update(file_task, advance=1, description=f'Select train/val/test')

        for file in glob.glob(f'{base_path}/processed/*.csv'):
            base_file_name = os.path.basename(file)
            shutil.move(file, f'{base_path}/train/{base_file_name}')

        progress.console.print('[green]Create dataset file[/green]')
        create_dataset_file(datafile, observation_lenght, len(features), labels)

        for phase in phases:
            progress.console.print(f'[green]Process phase: {phase}[/green]')
            prepare_dataset(datafile, progress, base_path, phase, observation_lenght, features, labels)
            prepare_balanced_dataset(datafile, progress, phase, labels)
            clear_label_samples(datafile, labels)

        drop_label_samples(datafile, labels)

        for phase in phases:
            progress.console.print(f'[green]Zip csv phase: {phase}[/green]')
            zip_files(f'{base_path}/{phase}', f'{base_path}/{phase}_{base_file_output_name}_csv.zip', '.csv')
            for file in glob.glob(f'{base_path}/{phase}/*.csv'):
                os.remove(file)

            os.removedirs(f'{base_path}/{phase}')

        os.removedirs(f'{base_path}/processed')
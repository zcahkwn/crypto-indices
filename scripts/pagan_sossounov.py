import pandas as pd
import numpy as np

def _calculate_market_phase(price_series, window=30, min_phase_days=112, min_cycle_days=490, threshold=0.20):
    """
    Identifies bull/bear markets using modified Pagan-Sossounov algorithm
    
    Parameters:
    price_series (pd.Series): Bitcoin closing prices
    window (int): Lookback/lookahead window for peak/trough detection (days)
    min_phase_days (int): Minimum phase duration (16 weeks ≈ 112 days)
    min_cycle_days (int): Minimum cycle duration (70 weeks ≈ 490 days)
    threshold (float): 20% price change threshold for short phases
    
    Returns:
    pd.Series: 1 for bull markets, 0 for bear markets
    """
    df = pd.DataFrame({'price': price_series})
    df = df.dropna()
    
    # Step 1: Find initial peaks and troughs
    df['peak'] = _find_peaks(df['price'], window)
    df['trough'] = _find_troughs(df['price'], window)
    
    # Step 2: Phase validation
    phases = []
    current_state = None
    start_idx = 0
    
    for i in range(len(df)):
        if df['peak'].iloc[i]:
            if current_state != 'peak':
                phases.append(('peak', start_idx, i))
                current_state = 'peak'
                start_idx = i
        elif df['trough'].iloc[i]:
            if current_state != 'trough':
                phases.append(('trough', start_idx, i))
                current_state = 'trough'
                start_idx = i
    
    # Step 3: Apply duration and magnitude filters
    valid_phases = []
    for j in range(1, len(phases)):
        prev_type, prev_start, prev_end = phases[j-1]
        curr_type, curr_start, curr_end = phases[j]
        
        duration = curr_start - prev_start
        price_change = (df['price'].iloc[curr_start] / df['price'].iloc[prev_start]) - 1
        
        # Check cycle duration and phase constraints
        if duration >= min_cycle_days:
            valid_phases.append((prev_type, prev_start, prev_end))
        else:
            if abs(price_change) >= threshold:
                valid_phases.append((prev_type, prev_start, prev_end))
    
    # Step 4: Create market phase labels
    df['market_phase'] = np.nan
    for phase in valid_phases:
        phase_type, start, end = phase
        df.iloc[start:end+1, df.columns.get_loc('market_phase')] = (
            1 if phase_type == 'trough' else 0  # Bull markets start at troughs
        )
    
    # Forward fill missing values
    df['market_phase'] = df['market_phase'].ffill().fillna(0)
    return df['market_phase']

def _find_peaks(series, window):
    peaks = np.zeros(len(series), dtype=bool)
    for i in range(window, len(series)-window):
        if (series.iloc[i] == series.iloc[i-window:i+window].max() 
            and series.iloc[i] > series.iloc[i-1] 
            and series.iloc[i] > series.iloc[i+1]):
            peaks[i] = True
    return peaks

def _find_troughs(series, window):
    troughs = np.zeros(len(series), dtype=bool)
    for i in range(window, len(series)-window):
        if (series.iloc[i] == series.iloc[i-window:i+window].min() 
            and series.iloc[i] < series.iloc[i-1] 
            and series.iloc[i] < series.iloc[i+1]):
            troughs[i] = True
    return troughs

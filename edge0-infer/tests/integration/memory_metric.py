"""Select an available native process peak without changing its meaning."""
def process_peak(snapshot):
    for metric in ('physical', 'rss'):
        item = snapshot[metric]
        peak = (item.get('values') or {}).get('peak_bytes')
        if peak is not None:
            return metric, item.get('source', metric), peak
    raise ValueError('Neither physical nor RSS peak is available')

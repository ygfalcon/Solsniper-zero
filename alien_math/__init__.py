def pyramid_frequency(series):
    """Return the estimated frequency of a cyclical wave pattern."""
    series = list(map(float, series))
    if len(series) < 3:
        return 0.0
    peaks = []
    direction = 0
    for i in range(1, len(series)):
        diff = series[i] - series[i-1]
        if diff == 0:
            continue
        new_dir = 1 if diff > 0 else -1
        if direction and new_dir != direction:
            peaks.append(i-1)
        direction = new_dir
    if len(peaks) < 2:
        return 0.0
    intervals = [peaks[i] - peaks[i-1] for i in range(1, len(peaks))]
    if not intervals:
        return 0.0
    avg = sum(intervals) / len(intervals)
    if avg == 0:
        return 0.0
    return 1.0 / avg

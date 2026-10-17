"""Online-adaptation metrics."""

import numpy as np


def update_frequency(history):
    total = len(history)
    updated = sum(1 for h in history if h.get('updated', False))

    by_type = {}
    for h in history:
        utype = h.get('update_type', 'none')
        by_type[utype] = by_type.get(utype, 0) + 1

    return {
        'total_steps': total,
        'update_count': updated,
        'update_ratio': updated / max(1, total),
        'by_type': by_type
    }


def recovery_time(step_errors, drift_indices=None, threshold_percentile=90):
    errors = np.array(step_errors)
    if len(errors) < 20:
        return {'drift_events': [], 'avg_recovery_time': 0, 'num_drifts': 0}

    if drift_indices is None:
        drift_indices = []
        window = 10
        for i in range(window, len(errors) - window):
            before = np.mean(errors[max(0, i - window):i])
            after = np.mean(errors[i:i + window])
            if after > before * 2:
                drift_indices.append(i)

    baseline = np.percentile(errors[:max(1, len(errors) // 4)], threshold_percentile)
    drift_events = []

    for idx in drift_indices:
        recovered = False
        for j in range(idx, min(idx + len(errors) // 2, len(errors))):
            if errors[j] <= baseline:
                drift_events.append({
                    'drift_at': idx,
                    'recovered_at': j,
                    'recovery_steps': j - idx
                })
                recovered = True
                break

        if not recovered:
            drift_events.append({
                'drift_at': idx,
                'recovered_at': -1,
                'recovery_steps': -1  # -1 means it never recovered
            })

    avg_recovery = np.mean([e['recovery_steps'] for e in drift_events if e['recovery_steps'] > 0]) \
        if any(e['recovery_steps'] > 0 for e in drift_events) else -1

    return {
        'drift_events': drift_events,
        'avg_recovery_time': float(avg_recovery),
        'num_drifts': len(drift_events)
    }


def performance_decay(step_errors, window_size=50):
    errors = np.array(step_errors)
    if len(errors) < window_size * 2:
        return {
            'decay_rate': 0.0,
            'first_window_error': float(np.mean(errors)),
            'last_window_error': float(np.mean(errors))
        }

    first_error = np.mean(errors[:window_size])
    last_error = np.mean(errors[-window_size:])
    decay_rate = (last_error - first_error) / max(first_error, 1e-8)

    return {
        'decay_rate': float(decay_rate),
        'first_window_error': float(first_error),
        'last_window_error': float(last_error)
    }


def evaluate_online_metrics(history, step_errors):
    # Recovery is anchored on the drift points the framework actually detected; the 'error doubled'
    # heuristic below false-alarms on stationary segments and is only a fallback.
    if len(history) == len(step_errors):
        drift_indices = [
            i for i, h in enumerate(history)
            if isinstance(h, dict) and h.get('drift_detected')
        ]
    else:
        drift_indices = None

    results = {
        'update_frequency': update_frequency(history),
        'recovery_time': recovery_time(step_errors, drift_indices=drift_indices),
        'performance_decay': performance_decay(step_errors)
    }

    return results

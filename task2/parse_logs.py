from pathlib import Path
from collections import defaultdict

import numpy as np


def parse_bluegene_logs(logs_dir: Path):
    log_paths = logs_dir.glob('main.*.out')
    result = defaultdict(list)
    for log_path in log_paths:
        with open(log_path, 'rt') as in_f:
            lines = in_f.readlines()
            line = lines[0]

            n_proc, eps = line.split()
            n_proc, eps = int(n_proc), float(eps)

            exec_times = [float(line.split()[-1]) for line in lines[1:]]
            result[(n_proc, eps)] = np.mean(exec_times)
    return result


def parse_polus_logs(logs_dir: Path):
    log_paths = logs_dir.glob('main.*.out*')
    result = defaultdict(list)
    for log_path in log_paths:
        with open(log_path, 'rt') as in_f:
            lines = in_f.readlines()
            idx = lines.index('The output (if any) follows:\n')
            lines = lines[idx + 2:]
            idx = lines.index('PS:\n')
            lines = lines[:idx - 2]

            line = lines[0]

            n_proc, eps = line.split()
            n_proc, eps = int(n_proc), float(eps)

            exec_times = [float(line.split()[-1]) for line in lines[1:]]
            result[(n_proc, eps)] = np.mean(exec_times)
    return result


def main():
    print('Bluegene:')
    result = parse_bluegene_logs(Path('./logs_bluegene'))
    for key in sorted(result.keys()):
        print(key, result[key])
    print('Polus:')
    result = parse_polus_logs(Path('./logs_polus'))
    for key in sorted(result.keys()):
        print(key, result[key])


if __name__ == '__main__':
    main()

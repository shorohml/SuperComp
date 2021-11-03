from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


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
            errs = [float(line.split()[1]) for line in lines[1:]]
            result[(n_proc, eps)] = np.mean(exec_times), np.mean(errs)
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
            errs = [float(line.split()[1]) for line in lines[1:]]
            result[(n_proc, eps)] = np.mean(exec_times), np.mean(errs)
    return result


def get_table(result):
    table = ''
    keys = result.keys()

    n_proc_vals = eps = sorted(set([key[0] for key in keys]))
    eps_vals = list(sorted(set([key[1] for key in keys])))[::-1]

    for eps in eps_vals:
        table += '\hline\n'
        base = result[(1, eps)][0]
        for j, n_proc in enumerate(n_proc_vals):
            if j != 0:
                table += '\cline{2-5}\n'
            eps_s = f'{eps:.0e}' if j == 0 else ''
            exec_time = result[(n_proc, eps)][0]
            mean_err = result[(n_proc, eps)][1]
            table += f'{eps_s} & {n_proc} & {exec_time:0.3f}  & {base / exec_time:0.3f} & {mean_err:.3e} \\\\\n'
    table += '\hline\n'
    return table


def save_plots(result, imgs_dir: Path, prefix: str):
    keys = result.keys()

    n_proc_vals = eps = sorted(set([key[0] for key in keys]))
    eps_vals = list(sorted(set([key[1] for key in keys])))[::-1]

    font = {'family' : 'DejaVu Sans',
            'size'   : 20}
    matplotlib.rc('font', **font)

    _, ax = plt.subplots(figsize=(10, 10))

    for eps in eps_vals: 
        base = result[(1, eps)][0]
        speedups = [base / result[(n_proc, eps)][0] for n_proc in n_proc_vals]
        ax.set_xlim(0, 68)
        ax.set_ylim(0, 68)
        ax.set_xticks(np.arange(0, 70, 4))
        ax.set_yticks(np.arange(0, 70, 4))
        ax.plot(n_proc_vals, speedups, label=f'eps: {eps:.0e}')
    ax.set_xlabel('Число MPI-процессов')
    ax.set_ylabel('Ускорение')
    ax.grid()
    ax.legend()
    plt.savefig(imgs_dir / f"{prefix.replace(' ', '').replace('/', '')}.pdf")


def main():
    imgs_dir = Path('./imgs')
    imgs_dir.mkdir(exist_ok=True)

    print('Bluegene:')
    result = parse_bluegene_logs(Path('./logs_bluegene'))
    for key in sorted(result.keys()):
        print(key, result[key])
    print(get_table(result))
    save_plots(result, imgs_dir, 'Blue Gene/P')

    print('Polus:')
    result = parse_polus_logs(Path('./logs_polus'))
    for key in sorted(result.keys()):
        print(key, result[key])
    print(get_table(result))
    save_plots(result, imgs_dir, 'Polus')


if __name__ == '__main__':
    main()

from pathlib import Path


LOG_DIR = Path('../logs')


def parse_bg_logs(logs_dir: Path):
    logs = {}
    for file_path in logs_dir.glob('*'):
        data = []
        with open(file_path, 'rt') as in_f:
            data = in_f.readlines()        
        l_x = float(data[1].split()[-1])
        res = int(data[5].split()[-1])
        n_proc = int(data[16].split()[-1])
        err = float(data[18].split()[-1])
        elapsed = float(data[19].split()[-1])
        logs[(l_x, res, n_proc)] = err, elapsed
    return logs


def get_bg_tables(mpi_logs, openmp_logs):
    keys = mpi_logs.keys()

    l_x_vals = sorted(set([key[0] for key in keys]))
    res_vals = sorted(set([key[1] for key in keys]))
    n_proc_vals = sorted(set([key[2] for key in keys]))

    tables = {}
    for l_x in l_x_vals:
        table = ''
        for res in res_vals:
            table += '\hline\n'
            base = mpi_logs[(l_x, res, n_proc_vals[0])][1]
            for j, n_proc in enumerate(n_proc_vals):
                if j != 0:
                    table += '\cline{2-7}\n'

                mpi_err, mpi_time = mpi_logs[(l_x, res, n_proc)]
                _, openmp_time = openmp_logs[(l_x, res, n_proc)]

                res_s = f'{res:d}' if j == 0 else ''
                err_s = f'{mpi_err}' if j == 0 else ''

                table += '{} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {}\\\\\n'.format(
                    res_s,
                    n_proc,
                    mpi_time,
                    base / mpi_time,
                    openmp_time,
                    base / openmp_time,
                    mpi_time / openmp_time,
                    err_s
                )
        table += '\hline\n'
        tables[l_x] = table
    return tables


def main():
    mpi_logs =  parse_bg_logs(LOG_DIR / 'logs_bg_mpi')
    openmp_logs = parse_bg_logs(LOG_DIR / 'logs_bg_openmp')
    tables = get_bg_tables(mpi_logs, openmp_logs)
    for l_x, table in tables.items():
        with open(f'{l_x}.txt', 'wt') as out_f:
            out_f.write(table)


if __name__ == '__main__':
    main()

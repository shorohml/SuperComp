MPIEXEC="${1:-mpiexec.mpich}"
NPROC="${2:-12}"
EXEC="${2:-./main}"

${MPIEXEC} -np ${NPROC} ${EXEC} config.ini

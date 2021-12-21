for proc in 64 128
do
    for N in 128 256 512
    do
        echo configs/config_${N}.ini 
        mpisubmit.bg -n $proc -w 00:10:00 -m smp main -- configs/config_${N}.ini
        sleep 60
        echo configs/config_${N}_pi.ini
        mpisubmit.bg -n $proc -w 00:10:00 -m smp main -- configs/config_${N}_pi.ini
        sleep 60
    done
done


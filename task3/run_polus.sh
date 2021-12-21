for proc in 10 20 40
do
    for N in 128 256 512
    do
        echo configs/config_${N}.ini 
        mpisubmit.pl -p $proc -w 00:15 main -- configs/config_${N}.ini
        sleep 60
        echo configs/config_${N}_pi.ini
        mpisubmit.pl -p $proc -w 00:15 main -- configs/config_${N}_pi.ini
        sleep 60
    done
done


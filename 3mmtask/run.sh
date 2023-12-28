ARGC=$#
EXP_ARGC=1

rm *.err

if [ $ARGC -eq $EXP_ARGC ]; then 
	xlc 3mmtask.c -o 3mm -qsmp=omp -D$1
	echo "[+] Execute 3mm.c with $1"
else 
	xlc 3mmtask.c -o 3mm -qsmp=omp
	echo "[+] Execute 3mm.c by default"
fi;

bsub < tasks

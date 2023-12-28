ARGC=$#
EXP_ARGC=1

rm *.err
module load SpectrumMPI

if [ $ARGC -eq $EXP_ARGC ]; then 
	mpixlc 3mm_mpi.c -o 3mm -D$1
	echo "[+] Execute 3mm.c with $1"
else 
	mpixlc 3mm_mpi.c -o 3mm
	echo "[+] Execute 3mm.c by default"
fi;

bsub < tasks

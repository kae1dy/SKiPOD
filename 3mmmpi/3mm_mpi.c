/* Include benchmark-specific header. */
#include "3mm.h"
#include <mpi.h>

enum
{
    CODE_SUCCESS = 0,
    CODE_FAILURE = -1
};

double bench_t_start, bench_t_end;

static double 
rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, NULL);
    if (stat != 0)
        printf("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void
bench_timer_start()
{
    bench_t_start = rtclock();
}

void
bench_timer_stop()
{
    bench_t_end = rtclock();
}

void
bench_timer_print()
{
    printf("%0.6lf\n", bench_t_end - bench_t_start);
}

static void
init_array_BCD(
        int nj, int nk, int nl, int nm,
        double B[nk][nj],
        double C[nj][nm],
        double D[nm][nl])
{
    int i, j;

    for (i = 0; i < nk; i++)
        for (j = 0; j < nj; j++)
            B[i][j] = (double) ((i * (j + 1) + 2) % nj) / (5 * nj);

    for (i = 0; i < nj; i++)
        for (j = 0; j < nm; j++)
            C[i][j] = (double) (i * (j + 3) % nl) / (5 * nl);

    for (i = 0; i < nm; i++)
        for (j = 0; j < nl; j++)
            D[i][j] = (double) ((i * (j + 2) + 2) % nk) / (5 * nk);
}

static void
init_rows_A(
    int NUM_ROWS,
    int ni, int nk, 
    double A[NUM_ROWS][nk],
    int rank)
{
    int i, j;

    for (i = 0; i < NUM_ROWS; i++)
        for (j = 0; j < nk; j++)
            A[i][j] = (double) (((i + rank * NUM_ROWS) * j + 1) % ni) / (5 * ni);
}

static void
print_array(
        int ni, int nl,
        double G[ni][nl])
{
    int i, j;

    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s", "G");
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++) {
            if ((i * ni + j) % 20 == 0) fprintf(stderr, "\n");
            fprintf(stderr, "%0.2lf ", G[i][j]);
        }
    fprintf(stderr, "\nend   dump: %s\n", "G");
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void
kernel_1mm(
    int ni, int nj, int nk,
    double A[ni][nk],
    double B[nk][nj],
    double out[ni][nj])
{
    int i, j, k;

    /* ============ Perform Rows Multiplication by all processes ============ */
    for (i = 0; i < ni; ++i) {
        for (j = 0; j < nj; ++j) {
            for (k = 0; k < nk; ++k) {
                out[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

/* Unused, matrix partition into blocks */
static void
block_partition(int NP_COLS, int NP_ROWS, int rank, int ni, int nk, double (*A)[ni][nk]) {

    int i, j;
    int BLOCK_ROWS = ni / NP_ROWS;  /* Number of rows in _block_ */
    int BLOCK_COLS = nk / NP_COLS;  /* Number of cols in _block_ */
    const int BLOCK_SIZE = BLOCK_COLS;

    for (i = 0; i < NP_ROWS; ++i) {
        for (j = 0; j < NP_COLS; ++j) { 
            if (rank == i * NP_COLS + j) {
                if (j == NP_COLS - 1) {
                    BLOCK_COLS += nk % NP_COLS;
                }
                if (i == NP_ROWS - 1) {
                    BLOCK_ROWS += ni % NP_ROWS;
                }
            }
        }
    }

    double (*Block)[BLOCK_ROWS][BLOCK_COLS] = (double (*)[BLOCK_ROWS][BLOCK_COLS]) calloc(BLOCK_ROWS * BLOCK_COLS, sizeof(double)); 

    int disps[NP_ROWS * NP_COLS];
    int counts[NP_ROWS * NP_COLS];

    if (rank == 0) {

        for (i = 0; i < NP_ROWS; ++i) {
            for (j = 0; j < NP_COLS; ++j) {

                disps[i * NP_COLS + j] = i * nk * BLOCK_ROWS + j * BLOCK_COLS;
                counts[i * NP_COLS + j] = BLOCK_COLS;
                if (j == NP_COLS - 1) { 
                    counts[i * NP_COLS + j] += nk % NP_COLS;
                }
            }
        }
    }

    MPI_Datatype vec, localvec;
    MPI_Type_vector(BLOCK_SIZE, 1, BLOCK_COLS, MPI_DOUBLE, &localvec);
    MPI_Type_create_resized(localvec, 0, sizeof(double), &localvec);
    MPI_Type_commit(&localvec);

    MPI_Type_vector(BLOCK_SIZE, 1, nk, MPI_DOUBLE, &vec);
    MPI_Type_create_resized(vec, 0, sizeof(double), &vec);
    MPI_Type_commit(&vec);


    MPI_Scatterv(&((*A)[0][0]), counts, disps, vec,
                  &((*Block)[0][0]), BLOCK_COLS, localvec, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Do one more scatter, scattering just the last row of data */
    if (ni % NP_ROWS != 0) {
        if (rank == 0) {
            // MPI_Type_free(&vec);
            for (i = 0; i < NP_ROWS; ++i) {
                for (j = 0; j < NP_COLS; ++j) {

                    disps[i * NP_COLS + j] = 0;
                    counts[i * NP_COLS + j] = 0;

                    if (i == NP_ROWS - 1) {

                        disps[i * NP_COLS + j] = nk * (ni - ni % NP_ROWS) + j * BLOCK_COLS;

                        counts[i * NP_COLS + j] = BLOCK_COLS;
                        if (j == NP_COLS - 1) { 
                            counts[i * NP_COLS + j] += nk % NP_COLS;
                        }

                    }
                }
            }
        }

        MPI_Scatterv(&((*A)[0][0]), counts, disps, vec,
                      &((*Block)[BLOCK_ROWS - ni % NP_ROWS][0]), BLOCK_COLS, localvec, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Type_free(&localvec);
    MPI_Type_free(&vec);

    free(Block);
}

static int
verify(
        int ni, int nj, int nk,
        double A[ni][nk],
        double B[nk][nj],
        double res[ni][nj],
        double DBL_EPS)
{
    int i, j, k;

    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            double tmp = 0;
            for (k = 0; k < nk; ++k) {
                tmp += A[i][k] * B[k][j];
            }
            if (fabs(tmp - res[i][j]) >= DBL_EPS) {
                fprintf(stderr, "[-] Matrix Multiply incorrect.\n");
                return CODE_FAILURE;
            }
        }
    }
    printf("[+] Matrix Multiply correct.\n");
    return CODE_SUCCESS;
}


int
main(int argc, char **argv)
{   
    int p, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int NUM_ROWS = NI / p;                 /* Number of rows in Local Array */
    if (rank == p - 1) NUM_ROWS += NI % p;

    int i, j;
    double (*rowsA)[NUM_ROWS][NK];
    double (*rowsE)[NUM_ROWS][NJ];
    double (*rowsF)[NUM_ROWS][NM];
    double (*rowsG)[NUM_ROWS][NL];

    double (*B)[NK][NJ];
    double (*C)[NJ][NM];
    double (*D)[NM][NL];

    /* ===================== Allocation for matrices ===================== */

    rowsA = (double (*)[NUM_ROWS][NK]) calloc(NUM_ROWS * NK, sizeof(double));
    rowsE = (double (*)[NUM_ROWS][NJ]) calloc(NUM_ROWS * NJ, sizeof(double));
    rowsF = (double (*)[NUM_ROWS][NM]) calloc(NUM_ROWS * NM, sizeof(double));
    rowsG = (double (*)[NUM_ROWS][NL]) calloc(NUM_ROWS * NL, sizeof(double));

    B = (double (*)[NK][NJ]) calloc(NK * NJ, sizeof(double));
    C = (double (*)[NJ][NM]) calloc(NJ * NM, sizeof(double));
    D = (double (*)[NM][NL]) calloc(NM * NL, sizeof(double));

    init_rows_A(NUM_ROWS, NI, NK, *rowsA, rank);
    init_array_BCD(NJ, NK, NL, NM, *B, *C, *D);
    MPI_Barrier(MPI_COMM_WORLD);

    /* ========================= Matrix multiply ========================= */
    if (rank == 0) bench_timer_start();

    kernel_1mm(NUM_ROWS, NJ, NK, *rowsA, *B, *rowsE);
    kernel_1mm(NUM_ROWS, NM, NJ, *rowsE, *C, *rowsF);
    kernel_1mm(NUM_ROWS, NL, NJ, *rowsF, *D, *rowsG);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { 
        bench_timer_stop();
        bench_timer_print();
    }
    /* =================================================================== */

    free((void *) rowsA);
    free((void *) rowsE);
    free((void *) rowsF);
    free((void *) rowsG);

    free((void *) B);
    free((void *) C);
    free((void *) D);    

    MPI_Finalize();

    return CODE_SUCCESS;
}
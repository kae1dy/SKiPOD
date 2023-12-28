/* Include benchmark-specific header. */
#include "3mm.h"
#include <omp.h>


enum
{
    /* > lscpu
         L1d cache: 64K
         L1i cache: 32K
    => MAX_BLOCK_SIZE = sqrt(64 * (1 << 10) / 8 / 3) ~= 50 */
    
    MAX_BLOCK_SIZE = 50,
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
init_array(
        int ni, int nj, int nk, int nl, int nm,
        double A[ni][nk],
        double B[nk][nj],
        double C[nj][nm],
        double D[nm][nl])
{
    int i, j;


    for (i = 0; i < ni; i++)
        for (j = 0; j < nk; j++)
            A[i][j] = (double) ((i * j + 1) % ni) / (5 * ni);

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


/* Unused, replaced by function "kernel1mm" */
static void
kernel_3mm(
        int ni, int nj, int nk, int nl, int nm,
        double E[ni][nj],
        double A[ni][nk],
        double B[nk][nj],
        double F[nj][nl],
        double C[nj][nm],
        double D[nm][nl],
        double G[ni][nl])
{
    int i, j, k;

    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            E[i][j] = 0.0;
            for (k = 0; k < nk; ++k) {
                E[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    for (i = 0; i < nj; i++) {
        for (j = 0; j < nl; j++) {
            F[i][j] = 0.0;
            for (k = 0; k < nm; ++k) {
                F[i][j] += C[i][k] * D[k][j];
            }
        }
    }

    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            G[i][j] = 0.0;
            for (k = 0; k < nj; ++k) {
                G[i][j] += E[i][k] * F[k][j];
            }
        }
    }

}


static void
kernel_1mm(
        int ni, int nj, int nk,
        double A[ni][nk],
        double B[nk][nj],
        double out[ni][nj])
{
    double res = 0.;
    /* Local variable for every thread */
    #pragma omp parallel firstprivate(res)
    /* Thread that divides into tasks */
    #pragma omp single // 
    {
        int i, j, k;
        for (i = 0; i < ni; i += MAX_BLOCK_SIZE) {
            for (j = 0; j < nj; j += MAX_BLOCK_SIZE) {
                for (k = 0; k < nk; k += MAX_BLOCK_SIZE) {

                    /* =============== Blocked Matrix Multiplication =============== */
                    #pragma omp task
                    {
                        int ii, jj, kk;
                        for (ii = i; ii < MIN(ni, i + MAX_BLOCK_SIZE); ++ii) {
                            for (jj = j; jj < MIN(nj, j + MAX_BLOCK_SIZE); ++jj) {
                                res = 0;
                                for (kk = k; kk < MIN(nk, k + MAX_BLOCK_SIZE); ++kk) {
                                    res += A[ii][kk] * B[kk][jj];
                                }
                                #pragma omp atomic
                                out[ii][jj] += res;
                            }
                        } 
                    }
                }
            }
        }
    }
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

    /* ====================== Malloc for A, B, C, D matrices =================== */
    double (*A)[NI][NK];
    A = (double (*)[NI][NK]) malloc((NI) * (NK) * sizeof(double));
    double (*B)[NK][NJ];
    B = (double (*)[NK][NJ]) malloc((NK) * (NJ) * sizeof(double));
    double (*C)[NJ][NM];
    C = (double (*)[NJ][NM]) malloc((NJ) * (NM) * sizeof(double));
    double (*D)[NM][NL];
    D = (double (*)[NM][NL]) malloc((NM) * (NL) * sizeof(double));

    /* ======================== Calloc for E, F, G matrices ==================== */
    double (*E)[NI][NJ];
    E = (double (*)[NI][NJ]) calloc(NI * NJ, sizeof(double));
    double (*F)[NJ][NL];
    F = (double (*)[NJ][NL]) calloc(NJ * NL, sizeof(double));
    double (*G)[NI][NL];
    G = (double (*)[NI][NL]) calloc(NI * NL, sizeof(double));

    init_array(NI, NJ, NK, NL, NM, *A, *B, *C, *D);

    bench_timer_start();

    /* ==================== Matrix multiply with valid check =================== */
    kernel_1mm(NI, NJ, NK, *A, *B, *E);
    // if (verify(NI, NJ, NK, *A, *B, *E, 1e-6) < 0) {
    //     return CODE_FAILURE;
    // }

    kernel_1mm(NJ, NL, NM, *C, *D, *F);
    // if (verify(NJ, NL, NM, *C, *D, *F, 1e-6) < 0) {
    //     return CODE_FAILURE;
    // }

    kernel_1mm(NI, NL, NJ, *E, *F, *G);
    // if (verify(NI, NL, NJ, *E, *F, *G, 1e-6) < 0) {
    //     return CODE_FAILURE;
    // }

    bench_timer_stop();
    bench_timer_print();

    if (argc > 42 && !strcmp(argv[0], "")) print_array(NI, NL, *G);

    free((void *) E);
    free((void *) A);
    free((void *) B);
    free((void *) F);
    free((void *) C);
    free((void *) D);
    free((void *) G);

    return CODE_SUCCESS;
}

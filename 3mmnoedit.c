/* Include benchmark-specific header. */
#include "3mm.h"

double bench_t_start, bench_t_end;

static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
      printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
  bench_t_start = rtclock ();
}

void bench_timer_stop()
{
  bench_t_end = rtclock ();
}

void bench_timer_print()
{
  printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}

static
void init_array(int ni, int nj, int nk, int nl, int nm,
  double A[ ni][nk],
  double B[ nk][nj],
  double C[ nj][nm],
  double D[ nm][nl])
{
  int i, j;


  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (double) ((i*j+1) % ni) / (5*ni);

  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (double) ((i*(j+1)+2) % nj) / (5*nj);

  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = (double) (i*(j+3) % nl) / (5*nl);

  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (double) ((i*(j+2)+2) % nk) / (5*nk);
}

static
void print_array(int ni, int nl,
   double G[ ni][nl])
{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "G");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
 if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
 fprintf (stderr, "%0.2lf ", G[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "G");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static
void kernel_3mm(
  int ni, int nj, int nk, int nl, int nm,
  double E[ ni][nj],
  double A[ ni][nk],
  double B[ nk][nj],
  double F[ nj][nl],
  double C[ nj][nm],
  double D[ nm][nl],
  double G[ ni][nl])
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

int main(int argc, char** argv)
{
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;

  double (*E)[ni][nj]; E = (double(*)[ni][nj])malloc ((ni) * (nj) * sizeof(double));
  double (*A)[ni][nk]; A = (double(*)[ni][nk])malloc ((ni) * (nk) * sizeof(double));
  double (*B)[nk][nj]; B = (double(*)[nk][nj])malloc ((nk) * (nj) * sizeof(double));
  double (*F)[nj][nl]; F = (double(*)[nj][nl])malloc ((nj) * (nl) * sizeof(double));
  double (*C)[nj][nm]; C = (double(*)[nj][nm])malloc ((nj) * (nm) * sizeof(double));
  double (*D)[nm][nl]; D = (double(*)[nm][nl])malloc ((nm) * (nl) * sizeof(double));
  double (*G)[ni][nl]; G = (double(*)[ni][nl])malloc ((ni) * (nl) * sizeof(double));

  init_array (ni, nj, nk, nl, nm,
       *A,
       *B,
       *C,
       *D);

  bench_timer_start();

  kernel_3mm (ni, nj, nk, nl, nm,
       *E,
       *A,
       *B,
       *F,
       *C,
       *D,
       *G);

  bench_timer_stop();
  bench_timer_print();

  if (argc > 42 && !strcmp(argv[0], "")) print_array(ni, nl, *G);

  free((void*)E);
  free((void*)A);
  free((void*)B);
  free((void*)F);
  free((void*)C);
  free((void*)D);
  free((void*)G);

  return 0;
}

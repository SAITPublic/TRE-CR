#ifndef CUBLAS_V1_WRAPPERS_H
#define CUBLAS_V1_WRAPPERS_H


#include <cuda_runtime_api.h>
#include <cublas.h>

extern "C" cublasStatus cublasInit () __attribute__((weak));
#define cublasInit() (cublasInit ? cublasInit() : 0)

extern "C" cublasStatus cublasShutdown () __attribute__((weak));
#define cublasShutdown() (cublasShutdown ? cublasShutdown() : 0)

extern "C" cublasStatus cublasGetError () __attribute__((weak));
#define cublasGetError() (cublasGetError ? cublasGetError() : 0)

extern "C" cublasStatus cublasGetVersion(int *version) __attribute__((weak));
#define cublasGetVersion(version) (cublasGetVersion ? cublasGetVersion(version) : 0)

extern "C" cublasStatus cublasAlloc (int n, int elemSize, void **devicePtr) __attribute__((weak));
#define cublasAlloc(n, elemSize, devicePtr) (cublasAlloc ? cublasAlloc(n, elemSize, devicePtr) : 0)

extern "C" cublasStatus cublasFree (void *devicePtr) __attribute__((weak));
#define cublasFree(devicePtr) (cublasFree ? cublasFree(devicePtr) : 0)

extern "C" cublasStatus cublasSetKernelStream (cudaStream_t stream) __attribute__((weak));
#define cublasSetKernelStream(stream) (cublasSetKernelStream ? cublasSetKernelStream(stream) : 0)

extern "C" float cublasSnrm2 (int n, const float *x, int incx) __attribute__((weak));
#define cublasSnrm2(n, x, incx) (cublasSnrm2 ? cublasSnrm2(n, x, incx) : 0)

extern "C" double cublasDnrm2 (int n, const double *x, int incx) __attribute__((weak));
#define cublasDnrm2(n, x, incx) (cublasDnrm2 ? cublasDnrm2(n, x, incx) : 0)

extern "C" float cublasScnrm2 (int n, const cuComplex *x, int incx) __attribute__((weak));
#define cublasScnrm2(n, x, incx) (cublasScnrm2 ? cublasScnrm2(n, x, incx) : 0)

extern "C" double cublasDznrm2 (int n, const cuDoubleComplex *x, int incx) __attribute__((weak));
#define cublasDznrm2(n, x, incx) (cublasDznrm2 ? cublasDznrm2(n, x, incx) : 0)

extern "C" float cublasSdot (int n, const float *x, int incx, const float *y,  int incy) __attribute__((weak));
#define cublasSdot(n, x, incx, y, incy) (cublasSdot ? cublasSdot(n, x, incx, y, incy) : 0)

extern "C" double cublasDdot (int n, const double *x, int incx, const double *y,  int incy) __attribute__((weak));
#define cublasDdot(n, x, incx, y, incy) (cublasDdot ? cublasDdot(n, x, incx, y, incy) : 0)

extern "C" cuComplex cublasCdotu (int n, const cuComplex *x, int incx, const cuComplex *y,  int incy) __attribute__((weak));
#define cublasCdotu(n, x, incx, y, incy) (cublasCdotu ? cublasCdotu(n, x, incx, y, incy) : 0)

extern "C" cuComplex cublasCdotc (int n, const cuComplex *x, int incx, const cuComplex *y,  int incy) __attribute__((weak));
#define cublasCdotc(n, x, incx, y, incy) (cublasCdotc ? cublasCdotc(n, x, incx, y, incy) : 0)

extern "C" cuDoubleComplex cublasZdotu (int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,  int incy) __attribute__((weak));
#define cublasZdotu(n, x, incx, y, incy) (cublasZdotu ? cublasZdotu(n, x, incx, y, incy) : 0)

extern "C" cuDoubleComplex cublasZdotc (int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y,  int incy) __attribute__((weak));
#define cublasZdotc(n, x, incx, y, incy) (cublasZdotc ? cublasZdotc(n, x, incx, y, incy) : 0)

extern "C" void cublasSscal (int n, float alpha, float *x, int incx) __attribute__((weak));
#define cublasSscal(n, alpha, x, incx) (cublasSscal ? cublasSscal(n, alpha, x, incx) : 0)

extern "C" void cublasDscal (int n, double alpha, double *x, int incx) __attribute__((weak));
#define cublasDscal(n, alpha, x, incx) (cublasDscal ? cublasDscal(n, alpha, x, incx) : 0)

extern "C" void cublasCscal (int n, cuComplex alpha, cuComplex *x, int incx) __attribute__((weak));
#define cublasCscal(n, alpha, x, incx) (cublasCscal ? cublasCscal(n, alpha, x, incx) : 0)

extern "C" void cublasZscal (int n, cuDoubleComplex alpha, cuDoubleComplex *x, int incx) __attribute__((weak));
#define cublasZscal(n, alpha, x, incx) (cublasZscal ? cublasZscal(n, alpha, x, incx) : 0)

extern "C" void cublasCsscal (int n, float alpha, cuComplex *x, int incx) __attribute__((weak));
#define cublasCsscal(n, alpha, x, incx) (cublasCsscal ? cublasCsscal(n, alpha, x, incx) : 0)

extern "C" void cublasZdscal (int n, double alpha, cuDoubleComplex *x, int incx) __attribute__((weak));
#define cublasZdscal(n, alpha, x, incx) (cublasZdscal ? cublasZdscal(n, alpha, x, incx) : 0)

extern "C" void cublasSaxpy (int n, float alpha, const float *x, int incx,  float *y, int incy) __attribute__((weak));
#define cublasSaxpy(n, alpha, x, incx, y, incy) (cublasSaxpy ? cublasSaxpy(n, alpha, x, incx, y, incy) : 0)

extern "C" void cublasDaxpy (int n, double alpha, const double *x,  int incx, double *y, int incy) __attribute__((weak));
#define cublasDaxpy(n, alpha, x, incx, y, incy) (cublasDaxpy ? cublasDaxpy(n, alpha, x, incx, y, incy) : 0)

extern "C" void cublasCaxpy (int n, cuComplex alpha, const cuComplex *x,  int incx, cuComplex *y, int incy) __attribute__((weak));
#define cublasCaxpy(n, alpha, x, incx, y, incy) (cublasCaxpy ? cublasCaxpy(n, alpha, x, incx, y, incy) : 0)

extern "C" void cublasZaxpy (int n, cuDoubleComplex alpha, const cuDoubleComplex *x,  int incx, cuDoubleComplex *y, int incy) __attribute__((weak));
#define cublasZaxpy(n, alpha, x, incx, y, incy) (cublasZaxpy ? cublasZaxpy(n, alpha, x, incx, y, incy) : 0)

extern "C" void cublasScopy (int n, const float *x, int incx, float *y,  int incy) __attribute__((weak));
#define cublasScopy(n, x, incx, y, incy) (cublasScopy ? cublasScopy(n, x, incx, y, incy) : 0)

extern "C" void cublasDcopy (int n, const double *x, int incx, double *y,  int incy) __attribute__((weak));
#define cublasDcopy(n, x, incx, y, incy) (cublasDcopy ? cublasDcopy(n, x, incx, y, incy) : 0)

extern "C" void cublasCcopy (int n, const cuComplex *x, int incx, cuComplex *y, int incy) __attribute__((weak));
#define cublasCcopy(n, x, incx, y, incy) (cublasCcopy ? cublasCcopy(n, x, incx, y, incy) : 0)

extern "C" void cublasZcopy (int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) __attribute__((weak));
#define cublasZcopy(n, x, incx, y, incy) (cublasZcopy ? cublasZcopy(n, x, incx, y, incy) : 0)

extern "C" void cublasSswap (int n, float *x, int incx, float *y, int incy) __attribute__((weak));
#define cublasSswap(n, x, incx, y, incy) (cublasSswap ? cublasSswap(n, x, incx, y, incy) : 0)

extern "C" void cublasDswap (int n, double *x, int incx, double *y, int incy) __attribute__((weak));
#define cublasDswap(n, x, incx, y, incy) (cublasDswap ? cublasDswap(n, x, incx, y, incy) : 0)

extern "C" void cublasCswap (int n, cuComplex *x, int incx, cuComplex *y, int incy) __attribute__((weak));
#define cublasCswap(n, x, incx, y, incy) (cublasCswap ? cublasCswap(n, x, incx, y, incy) : 0)

extern "C" void cublasZswap (int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) __attribute__((weak));
#define cublasZswap(n, x, incx, y, incy) (cublasZswap ? cublasZswap(n, x, incx, y, incy) : 0)

extern "C" int cublasIsamax (int n, const float *x, int incx) __attribute__((weak));
#define cublasIsamax(n, x, incx) (cublasIsamax ? cublasIsamax(n, x, incx) : 0)

extern "C" int cublasIdamax (int n, const double *x, int incx) __attribute__((weak));
#define cublasIdamax(n, x, incx) (cublasIdamax ? cublasIdamax(n, x, incx) : 0)

extern "C" int cublasIcamax (int n, const cuComplex *x, int incx) __attribute__((weak));
#define cublasIcamax(n, x, incx) (cublasIcamax ? cublasIcamax(n, x, incx) : 0)

extern "C" int cublasIzamax (int n, const cuDoubleComplex *x, int incx) __attribute__((weak));
#define cublasIzamax(n, x, incx) (cublasIzamax ? cublasIzamax(n, x, incx) : 0)

extern "C" int cublasIsamin (int n, const float *x, int incx) __attribute__((weak));
#define cublasIsamin(n, x, incx) (cublasIsamin ? cublasIsamin(n, x, incx) : 0)

extern "C" int cublasIdamin (int n, const double *x, int incx) __attribute__((weak));
#define cublasIdamin(n, x, incx) (cublasIdamin ? cublasIdamin(n, x, incx) : 0)

extern "C" int cublasIcamin (int n, const cuComplex *x, int incx) __attribute__((weak));
#define cublasIcamin(n, x, incx) (cublasIcamin ? cublasIcamin(n, x, incx) : 0)

extern "C" int cublasIzamin (int n, const cuDoubleComplex *x, int incx) __attribute__((weak));
#define cublasIzamin(n, x, incx) (cublasIzamin ? cublasIzamin(n, x, incx) : 0)

extern "C" float cublasSasum (int n, const float *x, int incx) __attribute__((weak));
#define cublasSasum(n, x, incx) (cublasSasum ? cublasSasum(n, x, incx) : 0)

extern "C" double cublasDasum (int n, const double *x, int incx) __attribute__((weak));
#define cublasDasum(n, x, incx) (cublasDasum ? cublasDasum(n, x, incx) : 0)

extern "C" float cublasScasum (int n, const cuComplex *x, int incx) __attribute__((weak));
#define cublasScasum(n, x, incx) (cublasScasum ? cublasScasum(n, x, incx) : 0)

extern "C" double cublasDzasum (int n, const cuDoubleComplex *x, int incx) __attribute__((weak));
#define cublasDzasum(n, x, incx) (cublasDzasum ? cublasDzasum(n, x, incx) : 0)

extern "C" void cublasSrot (int n, float *x, int incx, float *y, int incy,  float sc, float ss) __attribute__((weak));
#define cublasSrot(n, x, incx, y, incy, sc, ss) (cublasSrot ? cublasSrot(n, x, incx, y, incy, sc, ss) : 0)

extern "C" void cublasDrot (int n, double *x, int incx, double *y, int incy,  double sc, double ss) __attribute__((weak));
#define cublasDrot(n, x, incx, y, incy, sc, ss) (cublasDrot ? cublasDrot(n, x, incx, y, incy, sc, ss) : 0)

extern "C" void cublasCrot (int n, cuComplex *x, int incx, cuComplex *y,  int incy, float c, cuComplex s) __attribute__((weak));
#define cublasCrot(n, x, incx, y, incy, c, s) (cublasCrot ? cublasCrot(n, x, incx, y, incy, c, s) : 0)

extern "C" void cublasZrot (int n, cuDoubleComplex *x, int incx,  cuDoubleComplex *y, int incy, double sc,  cuDoubleComplex cs) __attribute__((weak));
#define cublasZrot(n, x, incx, y, incy, sc, cs) (cublasZrot ? cublasZrot(n, x, incx, y, incy, sc, cs) : 0)

extern "C" void cublasCsrot (int n, cuComplex *x, int incx, cuComplex *y, int incy, float c, float s) __attribute__((weak));
#define cublasCsrot(n, x, incx, y, incy, c, s) (cublasCsrot ? cublasCsrot(n, x, incx, y, incy, c, s) : 0)

extern "C" void cublasZdrot (int n, cuDoubleComplex *x, int incx,  cuDoubleComplex *y, int incy, double c, double s) __attribute__((weak));
#define cublasZdrot(n, x, incx, y, incy, c, s) (cublasZdrot ? cublasZdrot(n, x, incx, y, incy, c, s) : 0)

extern "C" void cublasSrotg (float *sa, float *sb, float *sc, float *ss) __attribute__((weak));
#define cublasSrotg(sa, sb, sc, ss) (cublasSrotg ? cublasSrotg(sa, sb, sc, ss) : 0)

extern "C" void cublasDrotg (double *sa, double *sb, double *sc, double *ss) __attribute__((weak));
#define cublasDrotg(sa, sb, sc, ss) (cublasDrotg ? cublasDrotg(sa, sb, sc, ss) : 0)

extern "C" void cublasCrotg (cuComplex *ca, cuComplex cb, float *sc, cuComplex *cs) __attribute__((weak));
#define cublasCrotg(ca, cb, sc, cs) (cublasCrotg ? cublasCrotg(ca, cb, sc, cs) : 0)

extern "C" void cublasZrotg (cuDoubleComplex *ca, cuDoubleComplex cb, double *sc, cuDoubleComplex *cs) __attribute__((weak));
#define cublasZrotg(ca, cb, sc, cs) (cublasZrotg ? cublasZrotg(ca, cb, sc, cs) : 0)

extern "C" void cublasSrotm(int n, float *x, int incx, float *y, int incy,  const float* sparam) __attribute__((weak));
#define cublasSrotm(n, x, incx, y, incy, sparam) (cublasSrotm ? cublasSrotm(n, x, incx, y, incy, sparam) : 0)

extern "C" void cublasDrotm(int n, double *x, int incx, double *y, int incy,  const double* sparam) __attribute__((weak));
#define cublasDrotm(n, x, incx, y, incy, sparam) (cublasDrotm ? cublasDrotm(n, x, incx, y, incy, sparam) : 0)

extern "C" void cublasSrotmg (float *sd1, float *sd2, float *sx1,  const float *sy1, float* sparam) __attribute__((weak));
#define cublasSrotmg(sd1, sd2, sx1, sy1, sparam) (cublasSrotmg ? cublasSrotmg(sd1, sd2, sx1, sy1, sparam) : 0)

extern "C" void cublasDrotmg (double *sd1, double *sd2, double *sx1,  const double *sy1, double* sparam) __attribute__((weak));
#define cublasDrotmg(sd1, sd2, sx1, sy1, sparam) (cublasDrotmg ? cublasDrotmg(sd1, sd2, sx1, sy1, sparam) : 0)

extern "C" void cublasSgemv (char trans, int m, int n, float alpha, const float *A, int lda, const float *x, int incx, float beta, float *y, int incy) __attribute__((weak));
#define cublasSgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy) (cublasSgemv ? cublasSgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasDgemv (char trans, int m, int n, double alpha, const double *A, int lda, const double *x, int incx, double beta, double *y, int incy) __attribute__((weak));
#define cublasDgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy) (cublasDgemv ? cublasDgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasCgemv (char trans, int m, int n, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex beta, cuComplex *y, int incy) __attribute__((weak));
#define cublasCgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy) (cublasCgemv ? cublasCgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasZgemv (char trans, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy) __attribute__((weak));
#define cublasZgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy) (cublasZgemv ? cublasZgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasSgbmv (char trans, int m, int n, int kl, int ku,  float alpha, const float *A, int lda,  const float *x, int incx, float beta, float *y,  int incy) __attribute__((weak));
#define cublasSgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) (cublasSgbmv ? cublasSgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasDgbmv (char trans, int m, int n, int kl, int ku,  double alpha, const double *A, int lda,  const double *x, int incx, double beta, double *y,  int incy) __attribute__((weak));
#define cublasDgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) (cublasDgbmv ? cublasDgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasCgbmv (char trans, int m, int n, int kl, int ku,  cuComplex alpha, const cuComplex *A, int lda,  const cuComplex *x, int incx, cuComplex beta, cuComplex *y,  int incy) __attribute__((weak));
#define cublasCgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) (cublasCgbmv ? cublasCgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasZgbmv (char trans, int m, int n, int kl, int ku,  cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,  const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y,  int incy) __attribute__((weak));
#define cublasZgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) (cublasZgbmv ? cublasZgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasStrmv (char uplo, char trans, char diag, int n,  const float *A, int lda, float *x, int incx) __attribute__((weak));
#define cublasStrmv(uplo, trans, diag, n, A, lda, x, incx) (cublasStrmv ? cublasStrmv(uplo, trans, diag, n, A, lda, x, incx) : 0)

extern "C" void cublasDtrmv (char uplo, char trans, char diag, int n,  const double *A, int lda, double *x, int incx) __attribute__((weak));
#define cublasDtrmv(uplo, trans, diag, n, A, lda, x, incx) (cublasDtrmv ? cublasDtrmv(uplo, trans, diag, n, A, lda, x, incx) : 0)

extern "C" void cublasCtrmv (char uplo, char trans, char diag, int n,  const cuComplex *A, int lda, cuComplex *x, int incx) __attribute__((weak));
#define cublasCtrmv(uplo, trans, diag, n, A, lda, x, incx) (cublasCtrmv ? cublasCtrmv(uplo, trans, diag, n, A, lda, x, incx) : 0)

extern "C" void cublasZtrmv (char uplo, char trans, char diag, int n,  const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) __attribute__((weak));
#define cublasZtrmv(uplo, trans, diag, n, A, lda, x, incx) (cublasZtrmv ? cublasZtrmv(uplo, trans, diag, n, A, lda, x, incx) : 0)

extern "C" void cublasStbmv (char uplo, char trans, char diag, int n, int k,  const float *A, int lda, float *x, int incx) __attribute__((weak));
#define cublasStbmv(uplo, trans, diag, n, k, A, lda, x, incx) (cublasStbmv ? cublasStbmv(uplo, trans, diag, n, k, A, lda, x, incx) : 0)

extern "C" void cublasDtbmv (char uplo, char trans, char diag, int n, int k,  const double *A, int lda, double *x, int incx) __attribute__((weak));
#define cublasDtbmv(uplo, trans, diag, n, k, A, lda, x, incx) (cublasDtbmv ? cublasDtbmv(uplo, trans, diag, n, k, A, lda, x, incx) : 0)

extern "C" void cublasCtbmv (char uplo, char trans, char diag, int n, int k,  const cuComplex *A, int lda, cuComplex *x, int incx) __attribute__((weak));
#define cublasCtbmv(uplo, trans, diag, n, k, A, lda, x, incx) (cublasCtbmv ? cublasCtbmv(uplo, trans, diag, n, k, A, lda, x, incx) : 0)

extern "C" void cublasZtbmv (char uplo, char trans, char diag, int n, int k,  const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) __attribute__((weak));
#define cublasZtbmv(uplo, trans, diag, n, k, A, lda, x, incx) (cublasZtbmv ? cublasZtbmv(uplo, trans, diag, n, k, A, lda, x, incx) : 0)

extern "C" void cublasStpmv (char uplo, char trans, char diag, int n, const float *AP, float *x, int incx) __attribute__((weak));
#define cublasStpmv(uplo, trans, diag, n, AP, x, incx) (cublasStpmv ? cublasStpmv(uplo, trans, diag, n, AP, x, incx) : 0)

extern "C" void cublasDtpmv (char uplo, char trans, char diag, int n, const double *AP, double *x, int incx) __attribute__((weak));
#define cublasDtpmv(uplo, trans, diag, n, AP, x, incx) (cublasDtpmv ? cublasDtpmv(uplo, trans, diag, n, AP, x, incx) : 0)

extern "C" void cublasCtpmv (char uplo, char trans, char diag, int n, const cuComplex *AP, cuComplex *x, int incx) __attribute__((weak));
#define cublasCtpmv(uplo, trans, diag, n, AP, x, incx) (cublasCtpmv ? cublasCtpmv(uplo, trans, diag, n, AP, x, incx) : 0)

extern "C" void cublasZtpmv (char uplo, char trans, char diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx) __attribute__((weak));
#define cublasZtpmv(uplo, trans, diag, n, AP, x, incx) (cublasZtpmv ? cublasZtpmv(uplo, trans, diag, n, AP, x, incx) : 0)

extern "C" void cublasStrsv (char uplo, char trans, char diag, int n, const float *A, int lda, float *x, int incx) __attribute__((weak));
#define cublasStrsv(uplo, trans, diag, n, A, lda, x, incx) (cublasStrsv ? cublasStrsv(uplo, trans, diag, n, A, lda, x, incx) : 0)

extern "C" void cublasDtrsv (char uplo, char trans, char diag, int n, const double *A, int lda, double *x, int incx) __attribute__((weak));
#define cublasDtrsv(uplo, trans, diag, n, A, lda, x, incx) (cublasDtrsv ? cublasDtrsv(uplo, trans, diag, n, A, lda, x, incx) : 0)

extern "C" void cublasCtrsv (char uplo, char trans, char diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx) __attribute__((weak));
#define cublasCtrsv(uplo, trans, diag, n, A, lda, x, incx) (cublasCtrsv ? cublasCtrsv(uplo, trans, diag, n, A, lda, x, incx) : 0)

extern "C" void cublasZtrsv (char uplo, char trans, char diag, int n, const cuDoubleComplex *A, int lda,  cuDoubleComplex *x, int incx) __attribute__((weak));
#define cublasZtrsv(uplo, trans, diag, n, A, lda, x, incx) (cublasZtrsv ? cublasZtrsv(uplo, trans, diag, n, A, lda, x, incx) : 0)

extern "C" void cublasStpsv (char uplo, char trans, char diag, int n, const float *AP,  float *x, int incx) __attribute__((weak));
#define cublasStpsv(uplo, trans, diag, n, AP, x, incx) (cublasStpsv ? cublasStpsv(uplo, trans, diag, n, AP, x, incx) : 0)

extern "C" void cublasDtpsv (char uplo, char trans, char diag, int n, const double *AP, double *x, int incx) __attribute__((weak));
#define cublasDtpsv(uplo, trans, diag, n, AP, x, incx) (cublasDtpsv ? cublasDtpsv(uplo, trans, diag, n, AP, x, incx) : 0)

extern "C" void cublasCtpsv (char uplo, char trans, char diag, int n, const cuComplex *AP, cuComplex *x, int incx) __attribute__((weak));
#define cublasCtpsv(uplo, trans, diag, n, AP, x, incx) (cublasCtpsv ? cublasCtpsv(uplo, trans, diag, n, AP, x, incx) : 0)

extern "C" void cublasZtpsv (char uplo, char trans, char diag, int n, const cuDoubleComplex *AP,  cuDoubleComplex *x, int incx) __attribute__((weak));
#define cublasZtpsv(uplo, trans, diag, n, AP, x, incx) (cublasZtpsv ? cublasZtpsv(uplo, trans, diag, n, AP, x, incx) : 0)

extern "C" void cublasStbsv (char uplo, char trans,  char diag, int n, int k, const float *A,  int lda, float *x, int incx) __attribute__((weak));
#define cublasStbsv(uplo, trans, diag, n, k, A, lda, x, incx) (cublasStbsv ? cublasStbsv(uplo, trans, diag, n, k, A, lda, x, incx) : 0)

extern "C" void cublasDtbsv (char uplo, char trans,  char diag, int n, int k, const double *A,  int lda, double *x, int incx) __attribute__((weak));
#define cublasDtbsv(uplo, trans, diag, n, k, A, lda, x, incx) (cublasDtbsv ? cublasDtbsv(uplo, trans, diag, n, k, A, lda, x, incx) : 0)

extern "C" void cublasCtbsv (char uplo, char trans,  char diag, int n, int k, const cuComplex *A,  int lda, cuComplex *x, int incx) __attribute__((weak));
#define cublasCtbsv(uplo, trans, diag, n, k, A, lda, x, incx) (cublasCtbsv ? cublasCtbsv(uplo, trans, diag, n, k, A, lda, x, incx) : 0)

extern "C" void cublasZtbsv (char uplo, char trans,  char diag, int n, int k, const cuDoubleComplex *A,  int lda, cuDoubleComplex *x, int incx) __attribute__((weak));
#define cublasZtbsv(uplo, trans, diag, n, k, A, lda, x, incx) (cublasZtbsv ? cublasZtbsv(uplo, trans, diag, n, k, A, lda, x, incx) : 0)

extern "C" void cublasSsymv (char uplo, int n, float alpha, const float *A, int lda, const float *x, int incx, float beta,  float *y, int incy) __attribute__((weak));
#define cublasSsymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy) (cublasSsymv ? cublasSsymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasDsymv (char uplo, int n, double alpha, const double *A, int lda, const double *x, int incx, double beta,  double *y, int incy) __attribute__((weak));
#define cublasDsymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy) (cublasDsymv ? cublasDsymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasChemv (char uplo, int n, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex beta,  cuComplex *y, int incy) __attribute__((weak));
#define cublasChemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy) (cublasChemv ? cublasChemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasZhemv (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex beta,  cuDoubleComplex *y, int incy) __attribute__((weak));
#define cublasZhemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy) (cublasZhemv ? cublasZhemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasSsbmv (char uplo, int n, int k, float alpha,  const float *A, int lda, const float *x, int incx,  float beta, float *y, int incy) __attribute__((weak));
#define cublasSsbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy) (cublasSsbmv ? cublasSsbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasDsbmv (char uplo, int n, int k, double alpha,  const double *A, int lda, const double *x, int incx,  double beta, double *y, int incy) __attribute__((weak));
#define cublasDsbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy) (cublasDsbmv ? cublasDsbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasChbmv (char uplo, int n, int k, cuComplex alpha,  const cuComplex *A, int lda, const cuComplex *x, int incx,  cuComplex beta, cuComplex *y, int incy) __attribute__((weak));
#define cublasChbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy) (cublasChbmv ? cublasChbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasZhbmv (char uplo, int n, int k, cuDoubleComplex alpha,  const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,  cuDoubleComplex beta, cuDoubleComplex *y, int incy) __attribute__((weak));
#define cublasZhbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy) (cublasZhbmv ? cublasZhbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy) : 0)

extern "C" void cublasSspmv (char uplo, int n, float alpha, const float *AP, const float *x, int incx, float beta, float *y, int incy) __attribute__((weak));
#define cublasSspmv(uplo, n, alpha, AP, x, incx, beta, y, incy) (cublasSspmv ? cublasSspmv(uplo, n, alpha, AP, x, incx, beta, y, incy) : 0)

extern "C" void cublasDspmv(char uplo, int n, double alpha, const double *AP, const double *x, int incx, double beta, double *y, int incy) __attribute__((weak));
#define cublasDspmv(uplo, n, alpha, AP, x, incx, beta, y, incy) (cublasDspmv ? cublasDspmv(uplo, n, alpha, AP, x, incx, beta, y, incy) : 0)

extern "C" void cublasChpmv (char uplo, int n, cuComplex alpha, const cuComplex *AP, const cuComplex *x, int incx, cuComplex beta, cuComplex *y, int incy) __attribute__((weak));
#define cublasChpmv(uplo, n, alpha, AP, x, incx, beta, y, incy) (cublasChpmv ? cublasChpmv(uplo, n, alpha, AP, x, incx, beta, y, incy) : 0)

extern "C" void cublasZhpmv (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *AP, const cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy) __attribute__((weak));
#define cublasZhpmv(uplo, n, alpha, AP, x, incx, beta, y, incy) (cublasZhpmv ? cublasZhpmv(uplo, n, alpha, AP, x, incx, beta, y, incy) : 0)

extern "C" void cublasSger (int m, int n, float alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) __attribute__((weak));
#define cublasSger(m, n, alpha, x, incx, y, incy, A, lda) (cublasSger ? cublasSger(m, n, alpha, x, incx, y, incy, A, lda) : 0)

extern "C" void cublasDger (int m, int n, double alpha, const double *x, int incx, const double *y, int incy, double *A, int lda) __attribute__((weak));
#define cublasDger(m, n, alpha, x, incx, y, incy, A, lda) (cublasDger ? cublasDger(m, n, alpha, x, incx, y, incy, A, lda) : 0)

extern "C" void cublasCgeru (int m, int n, cuComplex alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda) __attribute__((weak));
#define cublasCgeru(m, n, alpha, x, incx, y, incy, A, lda) (cublasCgeru ? cublasCgeru(m, n, alpha, x, incx, y, incy, A, lda) : 0)

extern "C" void cublasCgerc (int m, int n, cuComplex alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda) __attribute__((weak));
#define cublasCgerc(m, n, alpha, x, incx, y, incy, A, lda) (cublasCgerc ? cublasCgerc(m, n, alpha, x, incx, y, incy, A, lda) : 0)

extern "C" void cublasZgeru (int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) __attribute__((weak));
#define cublasZgeru(m, n, alpha, x, incx, y, incy, A, lda) (cublasZgeru ? cublasZgeru(m, n, alpha, x, incx, y, incy, A, lda) : 0)

extern "C" void cublasZgerc (int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) __attribute__((weak));
#define cublasZgerc(m, n, alpha, x, incx, y, incy, A, lda) (cublasZgerc ? cublasZgerc(m, n, alpha, x, incx, y, incy, A, lda) : 0)

extern "C" void cublasSsyr (char uplo, int n, float alpha, const float *x, int incx, float *A, int lda) __attribute__((weak));
#define cublasSsyr(uplo, n, alpha, x, incx, A, lda) (cublasSsyr ? cublasSsyr(uplo, n, alpha, x, incx, A, lda) : 0)

extern "C" void cublasDsyr (char uplo, int n, double alpha, const double *x, int incx, double *A, int lda) __attribute__((weak));
#define cublasDsyr(uplo, n, alpha, x, incx, A, lda) (cublasDsyr ? cublasDsyr(uplo, n, alpha, x, incx, A, lda) : 0)

extern "C" void cublasCher (char uplo, int n, float alpha,  const cuComplex *x, int incx, cuComplex *A, int lda) __attribute__((weak));
#define cublasCher(uplo, n, alpha, x, incx, A, lda) (cublasCher ? cublasCher(uplo, n, alpha, x, incx, A, lda) : 0)

extern "C" void cublasZher (char uplo, int n, double alpha,  const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda) __attribute__((weak));
#define cublasZher(uplo, n, alpha, x, incx, A, lda) (cublasZher ? cublasZher(uplo, n, alpha, x, incx, A, lda) : 0)

extern "C" void cublasSspr (char uplo, int n, float alpha, const float *x, int incx, float *AP) __attribute__((weak));
#define cublasSspr(uplo, n, alpha, x, incx, AP) (cublasSspr ? cublasSspr(uplo, n, alpha, x, incx, AP) : 0)

extern "C" void cublasDspr (char uplo, int n, double alpha, const double *x, int incx, double *AP) __attribute__((weak));
#define cublasDspr(uplo, n, alpha, x, incx, AP) (cublasDspr ? cublasDspr(uplo, n, alpha, x, incx, AP) : 0)

extern "C" void cublasChpr (char uplo, int n, float alpha, const cuComplex *x, int incx, cuComplex *AP) __attribute__((weak));
#define cublasChpr(uplo, n, alpha, x, incx, AP) (cublasChpr ? cublasChpr(uplo, n, alpha, x, incx, AP) : 0)

extern "C" void cublasZhpr (char uplo, int n, double alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *AP) __attribute__((weak));
#define cublasZhpr(uplo, n, alpha, x, incx, AP) (cublasZhpr ? cublasZhpr(uplo, n, alpha, x, incx, AP) : 0)

extern "C" void cublasSsyr2 (char uplo, int n, float alpha, const float *x,  int incx, const float *y, int incy, float *A,  int lda) __attribute__((weak));
#define cublasSsyr2(uplo, n, alpha, x, incx, y, incy, A, lda) (cublasSsyr2 ? cublasSsyr2(uplo, n, alpha, x, incx, y, incy, A, lda) : 0)

extern "C" void cublasDsyr2 (char uplo, int n, double alpha, const double *x,  int incx, const double *y, int incy, double *A,  int lda) __attribute__((weak));
#define cublasDsyr2(uplo, n, alpha, x, incx, y, incy, A, lda) (cublasDsyr2 ? cublasDsyr2(uplo, n, alpha, x, incx, y, incy, A, lda) : 0)

extern "C" void cublasCher2 (char uplo, int n, cuComplex alpha, const cuComplex *x,  int incx, const cuComplex *y, int incy, cuComplex *A,  int lda) __attribute__((weak));
#define cublasCher2(uplo, n, alpha, x, incx, y, incy, A, lda) (cublasCher2 ? cublasCher2(uplo, n, alpha, x, incx, y, incy, A, lda) : 0)

extern "C" void cublasZher2 (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *x,  int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A,  int lda) __attribute__((weak));
#define cublasZher2(uplo, n, alpha, x, incx, y, incy, A, lda) (cublasZher2 ? cublasZher2(uplo, n, alpha, x, incx, y, incy, A, lda) : 0)

extern "C" void cublasSspr2 (char uplo, int n, float alpha, const float *x,  int incx, const float *y, int incy, float *AP) __attribute__((weak));
#define cublasSspr2(uplo, n, alpha, x, incx, y, incy, AP) (cublasSspr2 ? cublasSspr2(uplo, n, alpha, x, incx, y, incy, AP) : 0)

extern "C" void cublasDspr2 (char uplo, int n, double alpha, const double *x, int incx, const double *y, int incy, double *AP) __attribute__((weak));
#define cublasDspr2(uplo, n, alpha, x, incx, y, incy, AP) (cublasDspr2 ? cublasDspr2(uplo, n, alpha, x, incx, y, incy, AP) : 0)

extern "C" void cublasChpr2 (char uplo, int n, cuComplex alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *AP) __attribute__((weak));
#define cublasChpr2(uplo, n, alpha, x, incx, y, incy, AP) (cublasChpr2 ? cublasChpr2(uplo, n, alpha, x, incx, y, incy, AP) : 0)

extern "C" void cublasZhpr2 (char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *AP) __attribute__((weak));
#define cublasZhpr2(uplo, n, alpha, x, incx, y, incy, AP) (cublasZhpr2 ? cublasZhpr2(uplo, n, alpha, x, incx, y, incy, AP) : 0)

extern "C" void cublasSgemm (char transa, char transb, int m, int n, int k,  float alpha, const float *A, int lda,  const float *B, int ldb, float beta, float *C,  int ldc) __attribute__((weak));
#define cublasSgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) (cublasSgemm ? cublasSgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasDgemm (char transa, char transb, int m, int n, int k, double alpha, const double *A, int lda,  const double *B, int ldb, double beta, double *C,  int ldc) __attribute__((weak));
#define cublasDgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) (cublasDgemm ? cublasDgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasCgemm (char transa, char transb, int m, int n, int k,  cuComplex alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C, int ldc) __attribute__((weak));
#define cublasCgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) (cublasCgemm ? cublasCgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasZgemm (char transa, char transb, int m, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex beta, cuDoubleComplex *C, int ldc) __attribute__((weak));
#define cublasZgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) (cublasZgemm ? cublasZgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasSsyrk (char uplo, char trans, int n, int k, float alpha,  const float *A, int lda, float beta, float *C,  int ldc) __attribute__((weak));
#define cublasSsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc) (cublasSsyrk ? cublasSsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc) : 0)

extern "C" void cublasDsyrk (char uplo, char trans, int n, int k, double alpha, const double *A, int lda, double beta, double *C, int ldc) __attribute__((weak));
#define cublasDsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc) (cublasDsyrk ? cublasDsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc) : 0)

extern "C" void cublasCsyrk (char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex *A, int lda, cuComplex beta, cuComplex *C, int ldc) __attribute__((weak));
#define cublasCsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc) (cublasCsyrk ? cublasCsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc) : 0)

extern "C" void cublasZsyrk (char uplo, char trans, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex beta, cuDoubleComplex *C, int ldc) __attribute__((weak));
#define cublasZsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc) (cublasZsyrk ? cublasZsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc) : 0)

extern "C" void cublasCherk (char uplo, char trans, int n, int k, float alpha, const cuComplex *A, int lda, float beta, cuComplex *C, int ldc) __attribute__((weak));
#define cublasCherk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc) (cublasCherk ? cublasCherk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc) : 0)

extern "C" void cublasZherk (char uplo, char trans, int n, int k, double alpha, const cuDoubleComplex *A, int lda, double beta, cuDoubleComplex *C, int ldc) __attribute__((weak));
#define cublasZherk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc) (cublasZherk ? cublasZherk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc) : 0)

extern "C" void cublasSsyr2k (char uplo, char trans, int n, int k, float alpha,  const float *A, int lda, const float *B, int ldb,  float beta, float *C, int ldc) __attribute__((weak));
#define cublasSsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc) (cublasSsyr2k ? cublasSsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasDsyr2k (char uplo, char trans, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc) __attribute__((weak));
#define cublasDsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc) (cublasDsyr2k ? cublasDsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasCsyr2k (char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C, int ldc) __attribute__((weak));
#define cublasCsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc) (cublasCsyr2k ? cublasCsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasZsyr2k (char uplo, char trans, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex beta, cuDoubleComplex *C, int ldc) __attribute__((weak));
#define cublasZsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc) (cublasZsyr2k ? cublasZsyr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasCher2k (char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, float beta, cuComplex *C, int ldc) __attribute__((weak));
#define cublasCher2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc) (cublasCher2k ? cublasCher2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasZher2k (char uplo, char trans, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, double beta, cuDoubleComplex *C, int ldc) __attribute__((weak));
#define cublasZher2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc) (cublasZher2k ? cublasZher2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasSsymm (char side, char uplo, int m, int n, float alpha,  const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) __attribute__((weak));
#define cublasSsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) (cublasSsymm ? cublasSsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasDsymm (char side, char uplo, int m, int n, double alpha,  const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc) __attribute__((weak));
#define cublasDsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) (cublasDsymm ? cublasDsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasCsymm (char side, char uplo, int m, int n, cuComplex alpha,  const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C, int ldc) __attribute__((weak));
#define cublasCsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) (cublasCsymm ? cublasCsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasZsymm (char side, char uplo, int m, int n, cuDoubleComplex alpha,  const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex beta, cuDoubleComplex *C, int ldc) __attribute__((weak));
#define cublasZsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) (cublasZsymm ? cublasZsymm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasChemm (char side, char uplo, int m, int n, cuComplex alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C, int ldc) __attribute__((weak));
#define cublasChemm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) (cublasChemm ? cublasChemm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasZhemm (char side, char uplo, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex beta, cuDoubleComplex *C, int ldc) __attribute__((weak));
#define cublasZhemm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) (cublasZhemm ? cublasZhemm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) : 0)

extern "C" void cublasStrsm (char side, char uplo, char transa, char diag, int m, int n, float alpha, const float *A, int lda, float *B, int ldb) __attribute__((weak));
#define cublasStrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) (cublasStrsm ? cublasStrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) : 0)

extern "C" void cublasDtrsm (char side, char uplo, char transa, char diag, int m, int n, double alpha, const double *A, int lda, double *B, int ldb) __attribute__((weak));
#define cublasDtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) (cublasDtrsm ? cublasDtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) : 0)

extern "C" void cublasCtrsm (char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, const cuComplex *A, int lda, cuComplex *B, int ldb) __attribute__((weak));
#define cublasCtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) (cublasCtrsm ? cublasCtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) : 0)

extern "C" void cublasZtrsm (char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb) __attribute__((weak));
#define cublasZtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) (cublasZtrsm ? cublasZtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) : 0)

extern "C" void cublasStrmm (char side, char uplo, char transa, char diag, int m, int n, float alpha, const float *A, int lda, float *B, int ldb) __attribute__((weak));
#define cublasStrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) (cublasStrmm ? cublasStrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) : 0)

extern "C" void cublasDtrmm (char side, char uplo, char transa, char diag, int m, int n, double alpha, const double *A, int lda, double *B, int ldb) __attribute__((weak));
#define cublasDtrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) (cublasDtrmm ? cublasDtrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) : 0)

extern "C" void cublasCtrmm (char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, const cuComplex *A, int lda, cuComplex *B, int ldb) __attribute__((weak));
#define cublasCtrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) (cublasCtrmm ? cublasCtrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) : 0)

extern "C" void cublasZtrmm (char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb) __attribute__((weak));
#define cublasZtrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) (cublasZtrmm ? cublasZtrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb) : 0)

extern "C" cublasStatus_t cublasSetMatrix (int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb) __attribute__((weak));
#define cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb) (cublasSetMatrix ? cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb) : 0)

extern "C" cublasStatus_t cublasGetMatrix (int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb) __attribute__((weak));
#define cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb) (cublasGetMatrix ? cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb) : 0)

extern "C" cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb, cudaStream_t stream) __attribute__((weak));
#define cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream) (cublasSetMatrixAsync ? cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream) : 0)

extern "C" cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize,  const void *A, int lda, void *B,  int ldb, cudaStream_t stream) __attribute__((weak));
#define cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream) (cublasGetMatrixAsync ? cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream) : 0)

extern "C" cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) __attribute__((weak));
#define cublasSetVector(n, elemSize, x, incx, y, incy) (cublasSetVector ? cublasSetVector(n, elemSize, x, incx, y, incy) : 0)

extern "C" cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) __attribute__((weak));
#define cublasGetVector(n, elemSize, x, incx, y, incy) (cublasGetVector ? cublasGetVector(n, elemSize, x, incx, y, incy) : 0)

extern "C" cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, cudaStream_t stream) __attribute__((weak));
#define cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream) (cublasSetVectorAsync ? cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream) : 0)

extern "C" cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, cudaStream_t stream) __attribute__((weak));
#define cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream) (cublasGetVectorAsync ? cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream) : 0)

#endif // CUBLAS_V1_WRAPPERS_H
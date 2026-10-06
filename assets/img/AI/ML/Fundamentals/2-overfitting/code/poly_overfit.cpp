// poly_overfit.cpp — Overfitting from scratch: polynomial regression + ridge.
// No std::vector, no <random>, no Eigen: raw arrays, manual new[]/delete[],
// a hand-written PRNG, and a Householder QR least-squares solver.
//
// Build: g++ -std=c++17 -O2 -Wall -Wextra -o poly_overfit poly_overfit.cpp
// Run:   ./poly_overfit            (also writes data.csv for the plotting script)
#include <cstdio>
#include <cmath>
#include <cstdint>

// ---------------- 1. Deterministic PRNG (splitmix64) + Box-Muller ----------------
static uint64_t rng_state = 42;
uint64_t next_u64() {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
double uniform01() { return (next_u64() >> 11) * (1.0 / 9007199254740992.0); } // [0,1)
double normal01() {                                    // Box-Muller transform
    double u1 = uniform01(), u2 = uniform01();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

// The "true model" the learner never sees: a cubic, as in the textbook's Figure 8.1.
double true_f(double x) { return 1.5 * x * x * x - x * x - x + 0.5; }
const double NOISE_SD = 0.25;

void make_data(double* x, double* y, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = 2.0 * uniform01() - 1.0;                // x ~ U[-1, 1]
        y[i] = true_f(x[i]) + NOISE_SD * normal01();   // y = f(x) + noise
    }
}

// ---------------- 2. Feature map: phi(x) = [1, x, x^2, ..., x^d] ----------------
// Fills a row-major (n x (d+1)) design matrix. Horner-free: just repeated products.
void design_matrix(const double* x, int n, int d, double* A) {
    for (int i = 0; i < n; ++i) {
        double p = 1.0;
        double* row = A + i * (d + 1);
        for (int j = 0; j <= d; ++j) { row[j] = p; p *= x[i]; }
    }
}

// ---------------- 3. Least squares via Householder QR ----------------
// Solves min ||A w - b||_2 for A (m x n, m >= n), row-major. A and b are overwritten.
// QR is used instead of the normal equations (A^T A) w = A^T b because forming
// A^T A squares the condition number — fatal for high-degree Vandermonde matrices.
bool lstsq_qr(double* A, double* b, int m, int n, double* w) {
    double* v = new double[m];
    for (int k = 0; k < n; ++k) {
        double norm = 0.0;
        for (int i = k; i < m; ++i) norm += A[i * n + k] * A[i * n + k];
        norm = std::sqrt(norm);
        if (norm < 1e-300) { delete[] v; return false; }
        double alpha = (A[k * n + k] > 0) ? -norm : norm;   // sign chosen to avoid cancellation
        for (int i = k; i < m; ++i) v[i] = A[i * n + k];
        v[k] -= alpha;
        double vnorm2 = 0.0;
        for (int i = k; i < m; ++i) vnorm2 += v[i] * v[i];
        if (vnorm2 < 1e-300) continue;
        // Apply H = I - 2 v v^T / (v^T v) to the remaining columns and to b.
        for (int j = k; j < n; ++j) {
            double dot = 0.0;
            for (int i = k; i < m; ++i) dot += v[i] * A[i * n + j];
            double s = 2.0 * dot / vnorm2;
            for (int i = k; i < m; ++i) A[i * n + j] -= s * v[i];
        }
        double dot = 0.0;
        for (int i = k; i < m; ++i) dot += v[i] * b[i];
        double s = 2.0 * dot / vnorm2;
        for (int i = k; i < m; ++i) b[i] -= s * v[i];
    }
    for (int k = n - 1; k >= 0; --k) {                  // back substitution R w = Q^T b
        double acc = b[k];
        for (int j = k + 1; j < n; ++j) acc -= A[k * n + j] * w[j];
        w[k] = acc / A[k * n + k];
    }
    delete[] v;
    return true;
}

// Fits degree-d polynomial with ridge penalty lambda * sum_{j>=1} w_j^2 (intercept free).
// Ridge is solved as ordinary least squares on an augmented system:
//   [ A            ]       [ y ]
//   [ sqrt(lam) I' ] w  ~  [ 0 ]
bool fit_poly(const double* x, const double* y, int n, int d, double lambda, double* w) {
    int p = d + 1;
    int extra = (lambda > 0.0) ? d : 0;
    int m = n + extra;
    double* A = new double[m * p];
    double* b = new double[m];
    design_matrix(x, n, d, A);
    for (int i = 0; i < n; ++i) b[i] = y[i];
    for (int r = 0; r < extra; ++r) {
        double* row = A + (n + r) * p;
        for (int j = 0; j < p; ++j) row[j] = 0.0;
        row[r + 1] = std::sqrt(lambda);                 // skip column 0 (intercept)
        b[n + r] = 0.0;
    }
    bool ok = lstsq_qr(A, b, m, p, w);
    delete[] A; delete[] b;
    return ok;
}

double predict(const double* w, int d, double x) {      // Horner's rule
    double acc = 0.0;
    for (int j = d; j >= 0; --j) acc = acc * x + w[j];
    return acc;
}

double mse(const double* w, int d, const double* x, const double* y, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) { double r = predict(w, d, x[i]) - y[i]; s += r * r; }
    return s / n;
}

int main() {
    const int N_TRAIN = 30, N_VAL = 15, N_TEST = 1000, MAX_D = 15;
    double* xtr = new double[N_TRAIN]; double* ytr = new double[N_TRAIN];
    double* xva = new double[N_VAL];   double* yva = new double[N_VAL];
    double* xte = new double[N_TEST];  double* yte = new double[N_TEST];
    make_data(xtr, ytr, N_TRAIN);
    make_data(xva, yva, N_VAL);
    make_data(xte, yte, N_TEST);

    double* w = new double[MAX_D + 1];

    // ----- Experiment A: model complexity (degree) vs. error -----
    std::printf("Irreducible error (noise variance) = %.4f\n\n", NOISE_SD * NOISE_SD);
    std::printf("degree  train_MSE   val_MSE    test_MSE\n");
    for (int d = 0; d <= MAX_D; ++d) {
        if (!fit_poly(xtr, ytr, N_TRAIN, d, 0.0, w)) { std::printf("%6d  singular\n", d); continue; }
        std::printf("%6d  %9.5f  %9.4f  %10.4f\n", d,
                    mse(w, d, xtr, ytr, N_TRAIN), mse(w, d, xva, yva, N_VAL), mse(w, d, xte, yte, N_TEST));
    }

    // ----- Experiment B: degree-15 model tamed by ridge (L2) regularization -----
    const int D = 15;
    const double lambdas[] = { 0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0 };
    std::printf("\nDegree %d with ridge penalty\n", D);
    std::printf("  lambda   train_MSE   val_MSE    test_MSE    ||w||_2\n");
    for (double lam : lambdas) {
        fit_poly(xtr, ytr, N_TRAIN, D, lam, w);
        double nrm = 0.0;
        for (int j = 1; j <= D; ++j) nrm += w[j] * w[j];
        std::printf("%8.0e  %9.5f  %9.4f  %10.4f  %10.2f\n", lam,
                    mse(w, D, xtr, ytr, N_TRAIN), mse(w, D, xva, yva, N_VAL),
                    mse(w, D, xte, yte, N_TEST), std::sqrt(nrm));
    }

    // ----- Dump the data so Python can plot exactly the same points -----
    FILE* f = std::fopen("data.csv", "w");
    if (f) {
        std::fprintf(f, "split,x,y\n");
        for (int i = 0; i < N_TRAIN; ++i) std::fprintf(f, "train,%.17g,%.17g\n", xtr[i], ytr[i]);
        for (int i = 0; i < N_VAL; ++i)   std::fprintf(f, "val,%.17g,%.17g\n", xva[i], yva[i]);
        for (int i = 0; i < N_TEST; ++i)  std::fprintf(f, "test,%.17g,%.17g\n", xte[i], yte[i]);
        std::fclose(f);
    }

    delete[] xtr; delete[] ytr; delete[] xva; delete[] yva; delete[] xte; delete[] yte; delete[] w;
    return 0;
}

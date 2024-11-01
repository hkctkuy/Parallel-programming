#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include <omp.h>

#define COEF 1.234 // Need to be great then 1
#define MAX_POSSIBLE_NEIB 5

enum VertexType { Square, Lower, Upper };

enum LogLevel { NoLog, TimeLog, InfoLog, ArrayLog };

/* Get cell number and vertex type by vertex number
 * # Arguments:
 * * vertex - vertex number
 * * k1 - square cells sequence length
 * * k2 - triangular cells sequence length
 * # Return values:
 * cell number
 * vertex type
 */
inline std::pair<size_t, VertexType> get_cell_n_type(
    size_t vertex,
    size_t k1,
    size_t k2
) {
    size_t div = vertex / (k1 + k2 * 2);
    size_t mod = vertex % (k1 + k2 * 2);
    size_t cell = vertex - div * k2;
    VertexType type = Square;
    if (mod >= k1) {
        cell -= (mod - k1 + 1) / 2;
        type = (mod - k1 + 1) % 2 ? Lower : Upper;
    }
    return {cell, type};
}

/* Get vertex number by cell number
 * # Arguments:
 * * cell - cell number
 * * k1 - square cells sequence length
 * * k2 - triangular cells sequence length
 * * type - expected vertex type (valuable for triangular)
 */
inline size_t get_vertex(
    size_t cell,
    size_t k1,
    size_t k2,
    VertexType type
) {
    size_t div = cell / (k1 + k2);
    size_t mod = cell % (k1 + k2);
    size_t vertex = cell + div * k2;
    if (mod >= k1) { // Triangular
        vertex += mod - k1;
        if (type == Upper) {
            vertex++;
        }
    }
    return vertex;
}

/* Generate CSR portrait by grid params
 * # Arguments:
 * * nx - grid hieght
 * * ny - grid width 
 * * k1 - square cells sequence length
 * * k2 - triangular cells sequence length
 * * ll - log level
 * # Return values:
 * * ia - row CSR array
 * * ja - col CSR array
 * * t - time
 */
auto gen(
    size_t nx,
    size_t ny,
    size_t k1,
    size_t k2,
    LogLevel ll
) {
    // Get ia/ja sizes
    size_t div = (nx * ny) / (k1 + k2);
    size_t mod = (nx * ny) % (k1 + k2);
    // Square vertices number
    size_t ns = k1 * div;
    // Triangular vertices number
    size_t nt = k2 * div * 2;
    if (mod > 0) {
        ns += k1 > mod ? mod : k1;
        mod = k1 > mod ? 0 : mod - k1;
        nt += 2 * mod;
    }
    // Vertices number
    size_t nv = ns + nt;
    // Edges number (doubled)
    size_t ne2 = 4 * ns + 3 * nt - 2 * (nx + ny);
    // IA size
    size_t size_ia = nv + 1;
    // JA size
    size_t size_ja = nv + ne2;
    if (ll >= InfoLog) {
        std::cout << "Square vertices number: " << ns << "\n";
        std::cout << "Triangular vertices number: " << nt << "\n";
        std::cout << "Vertices number: " << nv << "\n";
        std::cout << "Edges number: " << ne2 << "\n";
        std::cout << "IA size: " << size_ia << "\n";
        std::cout << "JA size: " << size_ja << "\n";
    }
    std::vector<size_t> ia(size_ia);
    ia[0] = 0;
    // Memory: 4|V|, instead |E| for ja :(
    std::vector<std::vector<size_t>> dist_ja(nv);
    auto start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for
        for (auto& dist_val : dist_ja) {
            dist_val = std::vector<size_t>(MAX_POSSIBLE_NEIB);
        }
    }
    // Fill vecs
    // Go through vertices nums
    #pragma omp parallel
    {
        #pragma omp for
        for (size_t v = 0; v < nv; v++) {
            // NOTE: Define and count this here by parallel issues
            // NOTE: In start of iteration we are always on square vertex
            // Cur cell number and vertex type
            auto [c, t] = get_cell_n_type(v, k1, k2);
            size_t i = c / ny; // Cur cell row number
            size_t j = c % ny; // Cur cell col number
            unsigned char neibs = 0; // Vertex neighbor number
            switch (t) {
                case Square:
                    if (i > 0) { // Upper neighbor
                        size_t neighbor = get_vertex(c - ny, k1, k2, Lower);
                        dist_ja[v][neibs++] = neighbor;
                    }
                    if (j > 0) { // Left neighbor
                        dist_ja[v][neibs++] = v - 1;
                    }
                    dist_ja[v][neibs++] = v; // Self
                    if (j < ny - 1) { // Right neighbor
                        dist_ja[v][neibs++] = v + 1;
                    }
                    if (i < nx - 1) { // Lower neighbor
                        size_t neighbor = get_vertex(c + ny, k1, k2, Upper);
                        dist_ja[v][neibs++] = neighbor;
                    }
                    break;
                case Lower:
                    if (j > 0) { // Left neighbor
                        dist_ja[v][neibs++] = v - 1;
                    }
                    dist_ja[v][neibs++] = v; // Self
                    dist_ja[v][neibs++] = v + 1; // Pair upper triangular neighbor
                    if (i < nx - 1) { // Lower neighbor
                        size_t neighbor = get_vertex(c + ny, k1, k2, Upper);
                        dist_ja[v][neibs++] = neighbor;
                    }
                    break;
                case Upper:
                    if (i > 0) { // Upper neighbor
                        size_t neighbor = get_vertex(c - ny, k1, k2, Lower);
                        dist_ja[v][neibs++] = neighbor;
                    }
                    dist_ja[v][neibs++] = v - 1; // Pair lower triangular neighbor
                    dist_ja[v][neibs++] = v; // Self
                    if (j < ny - 1) { // Right neighbor
                        dist_ja[v][neibs++] = v + 1;
                    }
                    break;
            }
            ia[v + 1] = neibs;
        }
    }
    // Adjust ia vals
    int prev = 0;
    for (int i = 1; i < size_ia; i++) {
        ia[i] += prev;
        prev = ia[i];
    }
    // Concat dist_ja to ja
    std::vector<size_t> ja(size_ja);
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < nv; i++) {
            auto pos = ja.begin() + ia[i];
            auto begin = dist_ja[i].begin();
            auto end = begin + (ia[i + 1] - ia[i]);
            std::copy(begin, end, pos);
        }
    }
    auto end = omp_get_wtime();
    auto t = end - start;
    return std::tuple(std::move(ia), std::move(ja), t);
}

inline float filler_i(size_t i) {
    return std::sin(i);
}

inline float filler_ij(size_t i, size_t j) {
    return std::cos(i * j + i + j);
}

/* Fill val CSR array by given row/col arrays and right side array
 * # Arguments:
 * * ia - row CSR array
 * * ja - col CSR array
 * # Return values:
 * * a - val CSR array
 * * b - right side array
 * * t - time
 */
auto fill(
    std::vector<size_t>& ia,
    std::vector<size_t>& ja
) {
    std::vector<float> a(ja.size());
    std::vector<float> b(ia.size() - 1);
    auto start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for
        for (size_t i = 0; i < ia.size() - 1; i++) {
            size_t diag_ind;
            float sum = 0;
            for (size_t ind = ia[i]; ind < ia[i + 1]; ind++) {
                size_t j = ja[ind];
                if (j != i) {
                    a[ind] = filler_ij(i, j);
                    sum += std::abs(a[ind]);
                } else {
                    diag_ind = ind;
                }
            }
            a[diag_ind] = COEF * sum;
            b[i] = filler_i(i);
        }
    }
    auto end = omp_get_wtime();
    auto t = end - start;
    return std::tuple(std::move(a), std::move(b), t);
}

/* Get CSR reverse diagonal matrix of given one
 * # Arguments:
 * * ia - row CSR array
 * * ja - col CSR array
 * * a - val CSR array
 * # Return values:
 * * im - row CSR result array
 * * jm - col CSR result array
 * * m - val CSR result array
 */
auto get_inverse_diag_m(
    std::vector<size_t>& ia,
    std::vector<size_t>& ja,
    std::vector<float>& a
) {
    size_t size = ia.size() - 1;
    std::vector<size_t> im(size + 1);
    std::vector<size_t> jm(size);
    std::vector<float> m(size);
    #pragma omp parallel
    {
        #pragma omp for
        for (size_t i = 0; i < size; i++) {
            im[i] = i;
            jm[i] = i;
            for (size_t ind = ia[i]; ind < ia[i + 1]; ind++) {
                size_t j = ja[ind];
                if (j == i) {
                    m[i] = 1 / a[ind];
                    break;
                }
            }
        }
    }
    im[size] = size;
    return std::tuple(std::move(im), std::move(jm), std::move(m));
}

/* Compute scalar product of two vectors with equal sizes
 * # Arguments:
 * * a - first vector
 * * b - second vector
 * * n - size
 * # Return value:
 * * scalar product value
 * * computing time
 */
auto scalar(
    std::vector<float>& a,
    std::vector<float>& b,
    size_t n
) {
    auto start = omp_get_wtime();
    float sum = 0;
    #pragma omp parallel
    {
        float sum_private = 0;
        #pragma omp for
        for (size_t i = 0; i < n; i++) {
            sum_private += a[i] * b[i];
        }
        #pragma omp critical
        sum += sum_private;
    }
    auto end = omp_get_wtime();
    return std::pair(sum, end - start);
}

/* Store sum of two vectors with equal sizes (second with coef)
 * to preallocated vector
 * # Arguments:
 * * res - allocated result vector
 * * a - first vector
 * * b - second vector
 * * c - second vector coefficient
 * * n - size
 * # Return value:
 * * computing time
 */
auto sum_vvc(
    std::vector<float>& res,
    std::vector<float>& a,
    std::vector<float>& b,
    float c,
    size_t n
) {
    auto start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for
        for (size_t i = 0; i < n; i++) {
            res[i] = a[i] + c * b[i];
        }
    }
    auto end = omp_get_wtime();
    return end - start;
}

/* Store multiply square CSR matrix by vector to preallocated vector
 * # Arguments:
 * * res - allocated result vector
 * * ia - row CSR array
 * * ja - col CSR array
 * * a - val CSR array
 * * v - vector
 * * n - size
 * # Return value:
 * * computing time
 */
auto mul_mv(
    std::vector<float>& res,
    std::vector<size_t>& ia,
    std::vector<size_t>& ja,
    std::vector<float>& a,
    std::vector<float>& v,
    size_t n
) {
    auto start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for
        for (size_t i = 0; i < n; i++) {
            res[i] = 0;
            for (size_t ind = ia[i]; ind < ia[i + 1]; ind++) {
                size_t j = ja[ind];
                res[i] += a[ind] * v[j];
            }
        }
    }
    auto end = omp_get_wtime();
    return end - start;
}

/* Compute L2 norm of given vector
 * # Arguments:
 * * a - first vector
 * * n - size
 */
float L2norm(
    std::vector<float>& a,
    size_t n
) {
    auto [scal, _] = scalar(a, a, n);
    return std::sqrt(scal);
}

/* Solve Ax=b system
 * # Arguments:
 * * n - size
 * * ia - A row CSR array
 * * ja - A col CSR array
 * * a - A val CSR array
 * * b - right side array
 * * eps - accuracy
 * * maxit = maximum iteration number
 * * ll - log level
 * # Return values:
 * * x - solution
 * * k - iteration number
 * * r - residual
 * * t - operation time tuple
 */
auto solve(
    size_t n,
    std::vector<size_t>& ia,
    std::vector<size_t>& ja,
    std::vector<float>& a,
    std::vector<float>& b,
    float eps,
    size_t maxit,
    LogLevel ll
) {
    double total_scal_time = 0;
    double total_add_time = 0;
    double total_mul_time = 0;
    std::vector<float> x(n, 0);
    std::vector<float> r(b);
    std::vector<float> z(n);
    std::vector<float> p(n);
    std::vector<float> q(n);
    auto [im, jm, m] = get_inverse_diag_m(ia, ja, a);
    float ro_prev = eps * eps + 1;
    size_t k = 1;
    auto start = omp_get_wtime();
    for(; ro_prev > eps * eps && k < maxit; k++) {
        auto z_mul_time = mul_mv(z, im, jm, m, r, n); // z = M^(-1) * r
        auto [ro, ro_scal_time] = scalar(r, z, n);
        if (k == 1) {
            p = z; // NOTE: not move!!!
        } else {
            float beta = ro / ro_prev;
            sum_vvc(p, z, p, beta, n);
        }
        auto q_mul_time = mul_mv(q, ia, ja, a, p, n); // q = Ap
        auto [scal, scal_time] = scalar(p, q, n);
        auto alpha = ro / scal;
        auto x_add_time = sum_vvc(x, x, p, alpha, n);
        auto r_add_time = sum_vvc(r, r, q, -alpha, n);
        ro_prev = ro;
        if (ll >= TimeLog) {
            total_scal_time += ro_scal_time + scal_time;
            total_add_time += x_add_time + r_add_time;
            total_mul_time += z_mul_time + z_mul_time;
        }
        if (ll >= InfoLog) {
            auto norm = L2norm(r, n);
            std::cout << "Iteration: " << k << "\t";
            std::cout << "Residual L2 norm: " << norm << std::endl;
        }
    }
    auto end = omp_get_wtime();
    auto time = end - start;
    auto t = std::tuple(time, total_scal_time, total_add_time, total_mul_time);
    return std::tuple(std::move(x), k, std::move(r), t);
}

int main(int argc, char** argv) {
    if (argc != 9) {
        // Help
        std::cout << "Usage: " << argv[0] << " Nx Ny K1 K2 Maxit Eps Tn Ll\n";
        std::cout << "Where:\n";
        std::cout << "Nx is positive int that represents grid hieght\n";
        std::cout << "Ny is positive int that represents grid width\n";
        std::cout << "K1 is positive int that represents square cells sequence length\n";
        std::cout << "K2 is positive int that represents triangular cells sequence length\n";
        std::cout << "Maxit is positive int that represents maximum iteration number\n";
        std::cout << "Eps is positive float that represents accuracy\n";
        std::cout << "Tn is tread number";
        std::cout << "Ll is log level:\n";
        std::cout << "\t<=" << NoLog << " - no logs\n";
        std::cout << "\t>=" << TimeLog << " - show time\n";
        std::cout << "\t>=" << InfoLog << " - show info\n"; // TODO: change description
        std::cout << "\t>=" << ArrayLog << " - show arrays\n";
        return 1;
    }
    size_t nx = std::stoll(argv[1]);
    size_t ny = std::stoll(argv[2]);
    size_t k1 = std::stoll(argv[3]);
    size_t k2 = std::stoll(argv[4]);
    size_t maxit = std::stoll(argv[5]);
    float eps = std::stof(argv[6]);
    size_t tn = std::stoll(argv[7]);
    LogLevel ll = (LogLevel) std::stoi(argv[8]);

    omp_set_num_threads(tn);

    // Gen ia/ja
    if (ll >= InfoLog) {
        std::cout << "Generating IA/JA..." << std::endl;
    }
    auto [ia, ja, gen_time] = gen(nx, ny, k1, k2, ll);
    if (ll >= ArrayLog) {
        std::cout << "IA:\t";
        for (const auto val: ia) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        std::cout << "JA:\t";
        for (const auto val: ja) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    if (ll >= TimeLog) {
        std::cout << "Gen time:\t" << gen_time << std::endl;
    }

    // Fill a
    if (ll >= InfoLog) {
        std::cout << "Filling A/b..." << std::endl;
    }
    auto [a, b, fill_time] = fill(ia, ja);
    if (ll >= ArrayLog) {
        std::cout << "A:\t";
        for (const auto val: a) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        std::cout << "b:\t";
        for (const auto val: b) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    if (ll >= TimeLog) {
        std::cout << "Fill time:\t" << fill_time << std::endl;
    }

    // Solve
    if (ll >= InfoLog) {
        std::cout << "Solving..." << std::endl;
    }
    size_t n = ia.size() - 1;
    auto [x, k, r, t] = solve(n, ia, ja, a, b, eps, maxit, ll);
    if (ll >= ArrayLog) {
        std::cout << "x:\t";
        for (const auto val: x) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    if (ll >= TimeLog) {
        auto [solve_time, scal_time, add_time, mul_time] = t;
        std::cout << "Solve time:\t" << solve_time << std::endl;
        std::cout << "Scalar total time:\t" << scal_time << "\t\t";
        std::cout << "Scalar avarage time:\t" << scal_time / k << "\n";
        std::cout << "Add total time:\t\t" << add_time << "\t\t";
        std::cout << "Add avarage time:\t" << add_time / k << "\n";
        std::cout << "Mul total time:\t\t" << mul_time << "\t\t";
        std::cout << "Mul avarage time:\t" << mul_time / k << "\n";
    }
    return 0;
}

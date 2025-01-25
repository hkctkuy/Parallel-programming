#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <limits.h>

#include <mpi.h>
#include <omp.h>

#define COEF 1.234 // Need to be great then 1
#define MAX_POSSIBLE_NEIB_NUM 5
#define HALO_NEIB_NUM 2

// define MPI_Datatype for size_t
#if SIZE_MAX == UCHAR_MAX
    #define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
    #define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
    #define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
    #define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
    #define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
    // I'll be lucky
    #define MPI_SIZE_T MPI_UINT64_T
#endif

// Print template start
template <template <typename... Args> class ContainerT, typename... Args>
std::ostream& operator << (std::ostream& stream, const ContainerT <Args...> & container) {
    stream << "{ ";
    for (auto&& elem : container) stream << elem << " ";
    stream << "}";
    return stream;
}

template <template <typename T, size_t N, typename... Args> class ContainerT, typename T, size_t N, typename... Args>
std::ostream& operator << (std::ostream& stream, const ContainerT <T, N, Args...> & container) {
    stream << "{ ";
    for (auto&& elem : container) stream << elem << " ";
    stream << "}";
    return stream;
}

template<typename T1, typename T2>
std::ostream& operator << (std::ostream& stream, const std::pair <T1, T2> & val) { 
    stream << "{" << val.first << " " << val.second << "}";
    return stream;
}

std::ostream& operator << (std::ostream& stream, const std::string& str) {
    stream << str.c_str();
    return stream;
}

template<typename T>
void print(std::string literal, T const &data) {
    std::cout << literal << ": " << data << "\n";
}

void print(std::string literal) {
    std::cout << literal << "\n";
}

/* Print given data
 * # Arguments:
 * * pr - process number (optional)
 * * literal - string literal
 * * cont - container
 */
template<typename T>
void print(int pr, std::string literal, T const &data) {
     std::cout << "[" << pr << "] ";
     print(literal, data);
}

enum VertexType { Square, Lower, Upper };

enum LogLevel { NoLog, TimeLog, ArrayLog, InfoLog };

// NOTE: in the local graph vectors we have following vertices order:
// Inner, interface vertices, upper, left, right and lower halo vertices

// Base section structure
struct Section {
    size_t offset;
    size_t size;

    Section(size_t offset = 0, size_t size = 0): offset(offset), size(size) {}
};

// Own section: has inner and inter subsections
struct Own: Section {
    Section inner;
    Section inter;

    Own(Section inner, Section inter):
        Section(inner.offset, inner.size + inter.size),
        inner(inner), inter(inter) {}
};

// Halo section: has upper, left, right and lower subsections
struct Halo: Section {
    Section upper;
    Section left_; // _ for text align
    Section right;
    Section lower;

    Halo(Section upper, Section left_, Section right, Section lower):
        Section(upper.offset, upper.size + left_.size + right.size + lower.size),
        upper(upper), left_(left_), right(right), lower(lower) {}
};

// Sections structure: has own and halo sections
struct Sections: Section {
    Own own;
    Halo halo;
    bool first_is_square;
    bool last_is_square;

    Sections(Own own, Halo halo, bool f, bool l):
        Section(own.offset, own.size + halo.size),
        own(own), halo(halo), first_is_square(f), last_is_square(l) {}
};

// Interface vertex send info
struct Sends {
    std::vector<size_t> upper;
    std::vector<size_t> left_;
    std::vector<size_t> right;
    std::vector<size_t> lower;

    Sends(
        size_t upper_size,
        size_t left__size,
        size_t right_size,
        size_t lower_size
    ):
        upper(upper_size),
        left_(left__size),
        right(right_size),
        lower(lower_size)
    {}
};

// Interprocess communication structure
struct Comm {
    int pr;
    std::vector<size_t> send;
    std::vector<size_t> recv;
};

/* Get cell number and vertex type by vertex number
 * # Arguments:
 * * vertex - vertex number
 * * k1 - square cells sequence length
 * * k2 - triangular cells sequence length
 * # Return values:
 * cell number
 * vertex type
 */
std::pair<size_t, VertexType> get_cell_n_type(
    size_t vertex,
    size_t k1,
    size_t k2
) {
    size_t div = vertex / (k1 + k2 * 2);
    size_t mod = vertex % (k1 + k2 * 2);
    size_t cell = vertex - div * k2;
    VertexType type = Square;
    if (mod >= k1) { // Triangular
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
 * # Return value:
 * vertex number
 */
size_t get_vertex(
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

/* Check if given cell number corresponds to a square vertex
 * # Arguments:
 * * cell - cell number
 * * k1 - square cells sequence length
 * * k2 - triangular cells sequence length
 */
bool is_square(
    size_t cell,
    size_t k1,
    size_t k2
) {
    size_t mod = cell % (k1 + k2);
    return mod < k1;
}

/* Fill partition array
 * # Arguments:
 * * l2g - local to global vertices numbers array
 * * nv - vertices number
 * * no - own vertices number
 * * px - x axis decomposition param
 * * py - y axis decomposition param
 * * pr - process rank
 * # Return value:
 * * part - partition array
 */
auto fill_part(
    std::vector<size_t>& l2g,
    size_t nv,
    size_t no,
    int px,
    int py,
    int pr
) {
    std::vector<int> part(nv);
    MPI_Status status;
    // Stage 1: send part inside rows from left to right
    if (pr % py != 0) {
        MPI_Recv(part.data(), nv, MPI_INT, pr - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
    if (pr % py != py - 1) {
        // Fill part with self values
        #pragma omp parallel for
        for (size_t i = 0; i < no; i++) {
            part[l2g[i]] = pr;
        }
        MPI_Send(part.data(), nv, MPI_INT, pr + 1, 0, MPI_COMM_WORLD);
    }
    // Stage 2: send part in last rows processes from top to bottom
    // NOTE: need to merge vectors received from diff senders
    if (pr % py == py - 1) {
        if (pr / py != 0) {
            std::vector<int> tmp(nv);
            MPI_Recv(tmp.data(), nv, MPI_INT, pr - py, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            // Merge received vectors
            #pragma omp parallel for
            for (size_t i = 0; i < nv; i++) {
                if (tmp[i] != 0) {
                    part[i] = tmp[i];
                }
            }
        }
        // Fill part with self values
        #pragma omp parallel for
        for (size_t i = 0; i < no; i++) {
            part[l2g[i]] = pr;
        }
        if (pr / py != px - 1) {
            MPI_Send(part.data(), nv, MPI_INT, pr + py, 0, MPI_COMM_WORLD);
        }
        // Stage 3: return part in last rows processes from bottom to top
        if (pr / py != px - 1) {
            MPI_Recv(part.data(), nv, MPI_INT, pr + py, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
        if (pr / py != 0) {
            MPI_Send(part.data(), nv, MPI_INT, pr - py, 0, MPI_COMM_WORLD);
        }
    }
    // Stage 4: return part inside rows from right to left
    if (pr % py != py - 1) {
        MPI_Recv(part.data(), nv, MPI_INT, pr + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
    if (pr % py != 0) {
        MPI_Send(part.data(), nv, MPI_INT, pr - 1, 0, MPI_COMM_WORLD);
    }
    return part;
}

/* Generate CSR portrait by grid params
 * # Arguments:
 * * nx - grid hieght
 * * ny - grid width 
 * * k1 - square cells sequence length
 * * k2 - triangular cells sequence length
 * * px - x axis decomposition param
 * * py - y axis decomposition param
 * * pr - MPI process rank
 * * ll - log level
 * # Return values:
 * * ia - row CSR array
 * * ja - col CSR array
 * * l2g - local to global vertices numbers array
 * * part - partition array
 * * sections - vertex sections info
 * * sends - interface vertex send info
 * * t - time
 */
auto gen(
    size_t nx,
    size_t ny,
    size_t k1,
    size_t k2,
    int px,
    int py,
    int pr,
    LogLevel ll
) {
    // Count total vertices number
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
    // Count process area ranges
    size_t pi = pr / py;
    size_t pj = pr % py;
    div = nx / px;
    mod = nx % px;
    size_t ib = div * pi + std::min<size_t>(mod, pi);
    size_t ie = div * (pi + 1) + std::min<size_t>(mod, pi + 1);
    size_t in = ie - ib;
    div = ny / py;
    mod = ny % py;
    size_t jb = div * pj + std::min<size_t>(mod, pj);
    size_t je = div * (pj + 1) + std::min<size_t>(mod, pj + 1);
    size_t jn = je - jb;
    // Collect info about rows vertices number and offsets
    std::vector<size_t> inner_offsets(in + 1);
    std::vector<size_t> inter_offsets(in + 1);
    #pragma omp parallel for
    for (size_t i = ib; i < ie; i++) {
        // Row vertices numbers
        auto& row_inner_nv = inner_offsets[i - ib + 1];
        auto& row_inter_nv = inter_offsets[i - ib + 1];
        for (size_t j = jb; j < je; j++) {
            size_t c = i * ny + j;
            if(is_square(c, k1, k2)) {
                bool is_inter = i == ib && i > 0 || i == ie - 1 && i < nx - 1
                    || j == jb && j > 0 || j == je - 1 && j < ny - 1;
                (is_inter ? row_inter_nv : row_inner_nv) += 1;
            } else {
                bool is_inter = i == ie - 1 && i < nx - 1 || j == jb && j > 0;
                (is_inter ? row_inter_nv : row_inner_nv) += 1;
                is_inter = i == ib && i > 0 || j == je - 1 && j < ny - 1;
                (is_inter ? row_inter_nv : row_inner_nv) += 1;
            }
        }
    }
    inner_offsets[0] = 0;
    for (size_t i = 0; i < in; i++) {
        inner_offsets[i + 1] += inner_offsets[i];
    }
    inter_offsets[0] = inner_offsets.back();
    for (size_t i = 0; i < in; i++) {
        inter_offsets[i + 1] += inter_offsets[i];
    }
    // Fill sections
    // Collect offsets and size info
    Section inner;
    inner.size = inner_offsets.back();
    Section inter;
    inter.offset = inter_offsets[0];
    inter.size = inter_offsets.back() - inter_offsets[0];
    Own own {inner, inter};
    Section upper {own.size, ib != 0 ? jn : 0};
    Section left_ {upper.offset + upper.size, jb != 0 ? in : 0};
    Section right {left_.offset + left_.size, je != ny ? in : 0};
    Section lower {right.offset + right.size, ie != nx ? jn : 0};
    Halo halo {upper, left_, right, lower};
    Sections sections {
        own, halo,
        is_square(ib * ny + jb, k1, k2),
        is_square((ie - 1) * ny + je - 1, k1, k2)
    };
    // Process own vertices number
    auto& no = sections.own.size;
    // Process vertices number
    auto& pnv = sections.size;
    if (ll >= InfoLog) {
        print(pr, "pi", pi);
        print(pr, "pj", pj);
        print(pr, "ib", ib);
        print(pr, "ie", ie);
        print(pr, "in", in);
        print(pr, "jb", jb);
        print(pr, "je", je);
        print(pr, "jn", jn);
        print(pr, "Interface offset", inter.offset);
        print(pr, "Interface size", inter.size);
        print(pr, "Halo upper size", upper.size);
        print(pr, "Halo right size", right.size);
        print(pr, "Halo left size", left_.size);
        print(pr, "Halo lower size", lower.size);
        print(pr, "Own vertices number", no);
        print(pr, "Process vertices number", pnv);
        print(pr, "First is square", sections.first_is_square);
        print(pr, "Last is square", sections.last_is_square);
    }
    // Target vectors
    std::vector<size_t> ia(pnv + 1);
    std::vector<size_t> l2g(pnv);
    std::vector<std::vector<size_t>> dist_ja(pnv);
    auto start = omp_get_wtime();
    #pragma omp parallel for
    for (size_t i = 0; i < no; i++) {
        dist_ja[i] = std::vector<size_t>(MAX_POSSIBLE_NEIB_NUM);
    }
    #pragma omp parallel for
    for (size_t i = no; i < pnv; i++) {
        dist_ja[i] = std::vector<size_t>(HALO_NEIB_NUM);
    }
    // Rows edges number
    std::vector<size_t> rne(in);
    // NOTE: We can simultaneously collect send info
    Sends sends(upper.size, left_.size, right.size, lower.size);
    // Fill vecs
    #pragma omp parallel for    
    for (size_t i = ib; i < ie; i++) {
        // Row edge number
        auto& ne = rne[i - ib];
        // Local inNer/inTer Offsets
        auto lnn = inner_offsets[i - ib];
        auto ltn = inter_offsets[i - ib];
        for (size_t j = jb; j < je; j++) {
            // NOTE: Define and count this here by parallel issues
            size_t c = i * ny + j; // Cur cell number
            size_t v = get_vertex(c, k1, k2, Lower); // Cur vertex number
            unsigned char neibs = 0; // Vertex neighbor number
            if (is_square(c, k1, k2)) {
                // Square
                bool is_inter = i == ib && i > 0 || i == ie - 1 && i < nx - 1
                    || j == jb && j > 0 || j == je - 1 && j < ny - 1;
                // Local vertex number
                size_t n = is_inter ? ltn++ : lnn++;
                if (i > 0) { // Upper neighbor
                    size_t neighbor = get_vertex(c - ny, k1, k2, Lower);
                    dist_ja[n][neibs++] = neighbor;
                    if (i == ib) {
                        dist_ja[upper.offset + j - jb][0] = v;
                        dist_ja[upper.offset + j - jb][1] = neighbor;
                        l2g[upper.offset + j - jb] = neighbor;
                        sends.upper[j - jb] = v;
                    }
                }
                if (j > 0) { // Left neighbor
                    size_t neighbor = v - 1;
                    dist_ja[n][neibs++] = neighbor;
                    if (j == jb) {
                        dist_ja[left_.offset + i - ib][0] = v;
                        dist_ja[left_.offset + i - ib][1] = neighbor;
                        l2g[left_.offset + i - ib] = neighbor;
                        sends.left_[i - ib] = v;
                    }
                }
                dist_ja[n][neibs++] = v; // Self
                if (j < ny - 1) { // Right neighbor
                    size_t neighbor = v + 1;
                    dist_ja[n][neibs++] = neighbor;
                    if (j == je - 1) {
                        dist_ja[right.offset + i - ib][0] = v;
                        dist_ja[right.offset + i - ib][1] = neighbor;
                        l2g[right.offset + i - ib] = neighbor;
                        sends.right[i - ib] = v;
                    }
                }
                if (i < nx - 1) { // Lower neighbor
                    size_t neighbor = get_vertex(c + ny, k1, k2, Upper);
                    dist_ja[n][neibs++] = neighbor;
                    if (i == ie - 1) {
                        dist_ja[lower.offset + j - jb][0] = v;
                        dist_ja[lower.offset + j - jb][1] = neighbor;
                        l2g[lower.offset + j - jb] = neighbor;
                        sends.lower[j - jb] = v;
                    }
                }
                ia[n + 1] = neibs;
                l2g[n] = v;
                ne += neibs;
            } else {
                // Lower
                bool is_inter = i == ie - 1 && i < nx - 1 || j == jb && j > 0;
                // Local vertex number
                size_t n = is_inter ? ltn++ : lnn++;
                if (j > 0) { // Left neighbor
                    size_t neighbor = v - 1;
                    dist_ja[n][neibs++] = neighbor;
                    if (j == jb) {
                        dist_ja[left_.offset + i - ib][0] = v;
                        dist_ja[left_.offset + i - ib][1] = neighbor;
                        l2g[left_.offset + i - ib] = neighbor;
                        sends.left_[i - ib] = v;
                    }
                }
                dist_ja[n][neibs++] = v; // Self
                dist_ja[n][neibs++] = v + 1; // Pair upper triangular neighbor
                if (i < nx - 1) { // Lower neighbor
                    size_t neighbor = get_vertex(c + ny, k1, k2, Upper);
                    dist_ja[n][neibs++] = neighbor;
                    if (i == ie - 1) {
                        dist_ja[lower.offset + j - jb][0] = v;
                        dist_ja[lower.offset + j - jb][1] = neighbor;
                        l2g[lower.offset + j - jb] = neighbor;
                        sends.lower[j - jb] = v;
                    }
                }
                ia[n + 1] = neibs;
                l2g[n] = v;
                ne += neibs;
                // Upper
                v++;
                neibs = 0;
                is_inter = i == ib && i > 0 || j == je - 1 && j < ny - 1;
                // Local vertex number
                n = is_inter ? ltn++ : lnn++;
                if (i > 0) { // Upper neighbor
                    size_t neighbor = get_vertex(c - ny, k1, k2, Lower);
                    dist_ja[n][neibs++] = neighbor;
                    if (i == ib) {
                        dist_ja[upper.offset + j - jb][0] = v;
                        dist_ja[upper.offset + j - jb][1] = neighbor;
                        l2g[upper.offset + j - jb] = neighbor;
                        sends.upper[j - jb] = v;
                    }
                }
                dist_ja[n][neibs++] = v - 1; // Pair lower triangular neighbor
                dist_ja[n][neibs++] = v; // Self
                if (j < ny - 1) { // Right neighbor
                    size_t neighbor = v + 1;
                    dist_ja[n][neibs++] = neighbor;
                    if (j == je - 1) {
                        dist_ja[right.offset + i - ib][0] = v;
                        dist_ja[right.offset + i - ib][1] = neighbor;
                        l2g[right.offset + i - ib] = neighbor;
                        sends.right[i - ib] = v;
                    }
                }
                ia[n + 1] = neibs;
                l2g[n] = v;
                ne += neibs;
            }
        }
    }
    // Adjust ia vals
    ia[0] = 0;
    for (int i = 1; i < no + 1; i++) {
        ia[i] += ia[i - 1];
    }
    for (int i = no + 1; i < pnv + 1; i++) {
        ia[i] = ia[i - 1] + HALO_NEIB_NUM;
    }
    // Count edges number
    size_t ne = HALO_NEIB_NUM * halo.size;
    #pragma omp parallel
    {
        size_t private_sum = 0;
        #pragma omp for
        for (size_t i = 0; i < in; i++) {
            private_sum += rne[i];
        }
        #pragma omp critical
        {
            ne += private_sum;
        }
    }
    // Concat dist_ja to ja
    std::vector<size_t> ja(ne);
    #pragma omp for
    for (size_t i = 0; i < pnv; i++) {
        auto pos = ja.begin() + ia[i];
        auto begin = dist_ja[i].begin();
        auto end = begin + (ia[i + 1] - ia[i]);
        std::copy(begin, end, pos);
    }
    // Fill local to global map
    std::unordered_map<size_t, size_t> g2l;
    g2l.reserve(l2g.size());
    for (int i = 0; i < l2g.size(); i++) {
        g2l[l2g[i]] = i;
    }
    // Convert global to local in ja
    #pragma omp parallel for
    for (int i = 0; i < ja.size(); i++) {
        ja[i] = g2l[ja[i]];
    }
    // Covert global to local in sends
    #pragma omp parallel for
    for (int i = 0; i < upper.size; i++) {
        sends.upper[i] = g2l[sends.upper[i]];
    }
    #pragma omp parallel for
    for (int i = 0; i < left_.size; i++) {
        sends.left_[i] = g2l[sends.left_[i]];
    }
    #pragma omp parallel for
    for (int i = 0; i < right.size; i++) {
        sends.right[i] = g2l[sends.right[i]];
    }
    #pragma omp parallel for
    for (int i = 0; i < lower.size; i++) {
        sends.lower[i] = g2l[sends.lower[i]];
    }
    // Fill part
    auto part = fill_part(l2g, nv, no, px, py, pr);
    auto end = omp_get_wtime();
    auto t = end - start;
    return std::tuple{ia, ja, l2g, part, sections, sends, t};
}

inline double filler_i(size_t i) {
    return std::sin(i);
}

inline double filler_ij(size_t i, size_t j) {
    return std::cos(i * j + i + j);
}

/* Fill val CSR array by given row/col arrays and right side array
 * # Arguments:
 * * ia - row CSR array
 * * ja - col CSR array
 * * l2g - local to global vertices numbers array
 * * no - own vertices number
 * # Return values:
 * * a - val CSR array
 * * b - right side array
 * * t - time
 */
auto fill(
    std::vector<size_t>& ia,
    std::vector<size_t>& ja,
    std::vector<size_t>& l2g,
    size_t no
) {
    auto n = ia.size() - 1;
    std::vector<double> a(ia[no]);
    std::vector<double> b(n);
    auto start = omp_get_wtime();
    #pragma omp parallel for
    for (size_t i = 0; i < no; i++) {
        size_t diag_ind;
        double sum = 0;
        for (size_t ind = ia[i]; ind < ia[i + 1]; ind++) {
            size_t j = ja[ind];
            if (j != i) {
                a[ind] = filler_ij(l2g[i], l2g[j]);
                sum += std::abs(a[ind]);
            } else {
                diag_ind = ind;
            }
        }
        a[diag_ind] = COEF * sum;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        b[i] = filler_i(l2g[i]);
    }
    auto end = omp_get_wtime();
    auto t = end - start;
    return std::tuple{a, b, t};
}

/* Get interprocess communication receive vector
 * # Arguments
 * * offset - halo section offset
 * * size - halo section size
 * # Return value
 * * recv - receive vector
 */
auto get_recv(size_t offset, size_t size) {
    std::vector<size_t> recv(size);
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        recv[i] = i + offset;
    }
    return recv;
}

/* Get interprocess communication
 * # Arguments
 * * halo - halo section
 * * send - send vector
 * * l2g - local to global vertices numbers array
 * * part - partition array
 * # Return value
 * * Comm struct
 */
auto get_comm(
    Section halo,
    std::vector<size_t>& send,
    std::vector<size_t>& l2g,
    std::vector<int>& part
) {
    std::sort(send.begin(), send.end()); // Just for beauty
    return Comm{
        part[l2g[halo.offset]],
        send,
        get_recv(halo.offset, halo.size)
    };
}

/* Build interprocess communication scheme
 * # Arguments
 * * sends - interface vertex send info
 * * sections - vertex sections info
 * * part - partition array
 * * l2g - local to global vertices numbers array
 * # Return values
 * * comms - interprocess communication array
 * * t - time
 */
auto build_comm(
    Sends& sends,
    Sections& sections,
    std::vector<int>& part,
    std::vector<size_t>& l2g
) {
    std::vector<Comm> comms;
    auto& inter = sections.own.inter;
    auto& halo = sections.halo;
    auto start = omp_get_wtime();
    if (halo.upper.size) {
        comms.push_back(get_comm(halo.upper, sends.upper, l2g, part));
    }
    if (halo.left_.size) {
        comms.push_back(get_comm(halo.left_, sends.left_, l2g, part));
    }
    if (halo.right.size) {
        comms.push_back(get_comm(halo.right, sends.right, l2g, part));
    }
    if (halo.lower.size) {
        comms.push_back(get_comm(halo.lower, sends.lower, l2g, part));
    }
    auto end = omp_get_wtime();
    auto t = end - start;
    return std::tuple{comms, t};
}

/* Update process halo values by communication with the others
 * # Arguments:
 * * v - vector to update
 * * comms - interprocess communication array
 */
void update(
    std::vector<double>& v,
    std::vector<Comm>& comms
) {
    auto cn = comms.size(); // Communications number
    std::vector<std::vector<double>> sends(cn);
    std::vector<std::vector<double>> recvs(cn);
    MPI_Request requests[2 * cn];
    MPI_Status statuses[2 * cn];
    for (size_t n = 0; n < cn; n++) {
        auto& comm = comms[n];
        auto size = comm.send.size();
        auto& send = sends[n];
        auto& recv = recvs[n];
        send = std::vector<double>(size);
        recv = std::vector<double>(size);
        #pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            send[i] = v[comm.send[i]];
        }
        MPI_Isend(send.data(), size, MPI_DOUBLE, comm.pr, 0, MPI_COMM_WORLD, &requests[2 * n]);
        MPI_Irecv(recv.data(), size, MPI_DOUBLE, comm.pr, 0, MPI_COMM_WORLD, &requests[2 * n + 1]);
    }
    MPI_Waitall(2 * cn, requests, statuses);
    for (size_t n = 0; n < cn; n++) {
        auto& comm = comms[n];
        auto size = comm.send.size();
        auto& recv = recvs[n];
        #pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            v[comm.recv[i]] = recv[i];
        }
    }
}

/* Get CSR reverse diagonal matrix of given one
 * # Arguments:
 * * ia - row CSR array
 * * ja - col CSR array
 * * a - val CSR array
 * * no - own vertices number
 * # Return values:
 * * im - row CSR result array
 * * jm - col CSR result array
 * * m - val CSR result array
 */
auto get_inverse_diag_m(
    std::vector<size_t>& ia,
    std::vector<size_t>& ja,
    std::vector<double>& a,
    size_t no
) {
    std::vector<size_t> im(no + 1);
    std::vector<size_t> jm(no);
    std::vector<double> m(no);
    #pragma omp parallel for
    for (size_t i = 0; i < no; i++) {
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
    im[no] = no;
    return std::tuple{std::move(im), std::move(jm), std::move(m)};
}

/* Compute scalar product of two vectors with equal sizes
 * # Arguments:
 * * a - first vector
 * * b - second vector
 * * no - own vertices number
 * # Return values:
 * * scalar product value
 * * computing time
 */
auto scalar(
    std::vector<double>& a,
    std::vector<double>& b,
    size_t no
) {
    auto start = omp_get_wtime();
    double local_sum = 0;
    #pragma omp parallel
    {
        double sum_private = 0;
        #pragma omp for
        for (size_t i = 0; i < no; i++) {
            sum_private += a[i] * b[i];
        }
        #pragma omp critical
        local_sum += sum_private;
    }
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    auto end = omp_get_wtime();
    return std::pair(global_sum, end - start);
}

/* Store sum of two vectors with equal sizes (second with coef)
 * to preallocated vector
 * # Arguments:
 * * res - allocated result vector
 * * a - first vector
 * * b - second vector
 * * c - second vector coefficient
 * # Return value:
 * * computing time
 */
auto sum_vvc(
    std::vector<double>& res,
    std::vector<double>& a,
    std::vector<double>& b,
    double c
) {
    auto n = a.size();
    auto start = omp_get_wtime();
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        res[i] = a[i] + c * b[i];
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
 * * comms - interprocess communication array
 * * no - own vertices number
 * # Return value:
 * * computing time
 */
auto mul_mv(
    std::vector<double>& res,
    std::vector<size_t>& ia,
    std::vector<size_t>& ja,
    std::vector<double>& a,
    std::vector<double>& v,
    std::vector<Comm>& comms,
    size_t no
) {
    auto start = omp_get_wtime();
    #pragma omp parallel for
    for (size_t i = 0; i < no; i++) {
        double sum = 0;
        for (size_t ind = ia[i]; ind < ia[i + 1]; ind++) {
            auto j = ja[ind];
            sum += a[ind] * v[j];
        }
        res[i] = sum;
    }
    auto end = omp_get_wtime();
    update(res, comms);
    return end - start;
}

/* Compute L2 norm of given vector
 * # Arguments:
 * * a - first vector
 * * no - own vertices number
 */
double L2norm(
    std::vector<double>& a,
    size_t no
) {
    auto [scal, _] = scalar(a, a, no);
    return std::sqrt(scal);
}

/* Solve Ax=b system
 * # Arguments:
 * * no - own vertices number
 * * ia - A row CSR array
 * * ja - A col CSR array
 * * a - A val CSR array
 * * b - right side array
 * * comms - interprocess communication array
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
    std::vector<size_t>& ia,
    std::vector<size_t>& ja,
    std::vector<double>& a,
    std::vector<double>& b,
    std::vector<Comm>& comms,
    size_t no,
    double eps,
    size_t maxit,
    LogLevel ll
) {
    double total_scal_time = 0;
    double total_add_time = 0;
    double total_mul_time = 0;
    size_t n = ia.size() - 1;
    std::vector<double> x(n);
    std::vector<double> r(b);
    std::vector<double> z(n);
    std::vector<double> p(n);
    std::vector<double> q(n);
    auto [im, jm, m] = get_inverse_diag_m(ia, ja, a, no);
    double ro_prev = eps * eps + 1;
    size_t k = 1;
    auto start = omp_get_wtime();
    for(; ro_prev > eps * eps && k < maxit; k++) {
        auto z_mul_time = mul_mv(z, im, jm, m, r, comms, no); // z = M^(-1) * r
        auto [ro, ro_scal_time] = scalar(r, z, no);
        if (k == 1) {
            p = z; // NOTE: not move!!!
        } else {
            double beta = ro / ro_prev;
            sum_vvc(p, z, p, beta);
        }
        auto q_mul_time = mul_mv(q, ia, ja, a, p, comms, no); // q = Ap
        auto [scal, scal_time] = scalar(p, q, no);
        auto alpha = ro / scal;
        auto x_add_time = sum_vvc(x, x, p, alpha);
        auto r_add_time = sum_vvc(r, r, q, -alpha);
        ro_prev = ro;
        if (ll >= TimeLog) {
            total_scal_time += ro_scal_time + scal_time;
            total_add_time += x_add_time + r_add_time;
            total_mul_time += z_mul_time + q_mul_time;
        }
        if (ll >= InfoLog) {
            auto norm = L2norm(r, no);
            print("Iteration", k);
            print("Residual L2 norm", norm);
        }
    }
    auto end = omp_get_wtime();
    auto time = end - start;
    auto t = std::tuple(time, total_scal_time, total_add_time, total_mul_time);
    return std::tuple(x, r, k, t);
}

int main(int argc, char** argv) {
    if (argc != 11) {
        // Help
        std::cout << "Usage: " << argv[0] << " Nx Ny K1 K2 Px Py Maxit Eps Tn Ll\n";
        std::cout << "Where:\n";
        std::cout << "Nx is positive int that represents grid hieght\n";
        std::cout << "Ny is positive int that represents grid width\n";
        std::cout << "K1 is positive int that represents square cells sequence length\n";
        std::cout << "K2 is positive int that represents triangular cells sequence length\n";
        std::cout << "Px is positive int that represents x axis decomposition param\n";
        std::cout << "Py is positive int that represents y axis decomposition param\n";
        std::cout << "Maxit is positive int that represents maximum iteration number\n";
        std::cout << "Eps is positive double that represents accuracy\n";
        std::cout << "Tn is tread number";
        std::cout << "Ll is log level:\n";
        std::cout << "\t<=" << NoLog << " - no logs\n";
        std::cout << "\t>=" << TimeLog << " - show time\n";
        std::cout << "\t>=" << ArrayLog << " - show arrays\n";
        std::cout << "\t>=" << InfoLog << " - show info\n";
        return 1;
    }
    
    // Init MPI
    int status, pn, pr;
    status = MPI_Init(&argc, &argv);
    if (status != MPI_SUCCESS) {
        std::cerr << "Failed to init MPI: " << status << std::endl;
        return 1;
    }
    status = MPI_Comm_size(MPI_COMM_WORLD, &pn);
    if (status != MPI_SUCCESS) {
        std::cerr << "Failed to get process number: " << status << std::endl;
        return 1;
    }
    status = MPI_Comm_rank(MPI_COMM_WORLD, &pr);
    if (status != MPI_SUCCESS) {
        std::cerr << "Failed to get process rank: " << status << std::endl;
        return 1;
    }

    // Get args
    auto nx = std::stoull(argv[1]);
    auto ny = std::stoull(argv[2]);
    auto k1 = std::stoull(argv[3]);
    auto k2 = std::stoull(argv[4]);
    auto px = std::stoi(argv[5]);
    auto py = std::stoi(argv[6]);
    auto maxit = std::stoull(argv[7]);
    auto eps = std::stod(argv[8]);
    auto tn = std::stoull(argv[9]);
    auto ll = LogLevel(std::stoi(argv[10]));

    if (px < 0 || py < 0) {
        std::cerr << "Negative decomposition params are not allowed\n";
        return 1;
    }
    if (pn != px * py) {
        std::cerr << "Decomposition params are not suit process number\n";
        return 1;
    }

    omp_set_num_threads(tn);
    
    // Gen ia/ja
    if (ll >= InfoLog && pr == 0) {
        print("Generating IA/JA...");
    }
    auto [ia, ja, l2g, part, sections, sends, gen_time] = gen(nx, ny, k1, k2, px, py, pr, ll);
    auto no = sections.own.size;
    if (ll >= ArrayLog) {
        print(pr, "IA", ia);
        print(pr, "JA", ja);
        print(pr, "L2G", l2g);
        print(pr, "PART", part);
        print(pr, "No", no);
    }
    if (ll >= TimeLog) {
        print(pr, "Gen time", gen_time);
    }

    // Fill a
    if (ll >= InfoLog && pr == 0) {
        print("Filling A/b...");
    }
    auto [a, b, fill_time] = fill(ia, ja, l2g, no);
    if (ll >= ArrayLog) {
        print(pr, "A", a);
        print(pr, "b", b);
    }
    if (ll >= TimeLog) {
        print(pr, "Fill time", fill_time);
    }

    // Build comms
    if (ll >= InfoLog && pr == 0) {
        print("Build comms");
    }
    auto [comms, comm_time] = build_comm(sends, sections, part, l2g);
    if (ll >= ArrayLog) {
        print(pr, "Comms", comms.size());
        for (const auto& comm: comms) {
            print(pr, "Comm", comm.pr);
            print(pr, "Send", comm.send);
            print(pr, "Recv", comm.recv);
        }
    }
    if (ll >= TimeLog) {
        print(pr, "Comm time", comm_time);
    }

    // Solve
    if (ll >= InfoLog && pr == 0) {
        print("Solving...");
    }
    auto [x, r, k, t] = solve(ia, ja, a, b, comms, no, eps, maxit, ll);
    if (ll >= ArrayLog) {
        print(pr, "x", x);
    }
    if (ll >= TimeLog) {
        auto [solve_time, scal_time, add_time, mul_time] = t;
        print(pr, "Solve time", solve_time);
        print(pr, "Scalar time", scal_time);
        print(pr, "Add time", add_time);
        print(pr, "Mul time", mul_time);

        size_t flop_per_iter = 8 * (ia.size() - 1) + 4 * ja.size();
        size_t local_flop = flop_per_iter * k;
        size_t global_flop;
        MPI_Reduce(&local_flop, &global_flop, 1, MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
        print(pr, "Local FLOP", local_flop);
        if (pr == 0) {
            print(pr, "Global FLOP", global_flop);
            print("Iterations", k);
        }
    }

    MPI_Finalize();
    return 0;
}

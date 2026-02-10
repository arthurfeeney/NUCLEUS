#include <torch/extension.h>
#include <queue>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace py = pybind11;

// Status codes for the fast marching method
const int KNOWN = 2;
const int TRIAL = 1;
const int FAR = 0;

struct TrialPoint {
    double value;
    int i;
    int j;
    
    bool operator>(const TrialPoint& other) const {
        return std::abs(value) > std::abs(other.value);
    }
};

double solve_eikonal(
    const torch::Tensor& phi,
    const torch::Tensor& phi_init,
    const torch::Tensor& status,
    int i, int j, double dx) {
    
    int ny = phi.size(0);
    int nx = phi.size(1);
    
    auto phi_a = phi.accessor<double, 2>();
    auto phi_init_a = phi_init.accessor<double, 2>();
    auto status_a = status.accessor<int, 2>();
    
    double phi_val = phi_a[i][j];
    //double phi_init_val = phi_init_a[i][j];

    // sign is determined using the original SDF, not the "in progress" SDF
    int sign = (phi_val > 0) ? 1 : ((phi_val < 0) ? -1 : 0);
    
    // Get upwind values in y-direction
    std::vector<double> phi_x;
    if (i > 0 && status_a[i-1][j] == KNOWN) {
        phi_x.push_back(phi_a[i-1][j]);
    }
    if (i < ny-1 && status_a[i+1][j] == KNOWN) {
        phi_x.push_back(phi_a[i+1][j]);
    }
    
    // Get upwind values in x-direction
    std::vector<double> phi_y;
    if (j > 0 && status_a[i][j-1] == KNOWN) {
        phi_y.push_back(phi_a[i][j-1]);
    }
    if (j < nx-1 && status_a[i][j+1] == KNOWN) {
        phi_y.push_back(phi_a[i][j+1]);
    }
    
    if (phi_x.empty() && phi_y.empty()) {
        return phi_val;  // No known neighbors
    }
    
    // Choose closest known value in each direction
    double a = std::numeric_limits<double>::infinity();
    double b = std::numeric_limits<double>::infinity();
    if (!phi_x.empty()) {
        a = *std::min_element(phi_x.begin(), phi_x.end(),
            [](double x, double y) { return std::abs(x) < std::abs(y); });
    }
    if (!phi_y.empty()) {
        b = *std::min_element(phi_y.begin(), phi_y.end(),
            [](double x, double y) { return std::abs(x) < std::abs(y); });
    }
    
    // Solve quadratic: (phi-a)^2 + (phi-b)^2 = dx^2
    if (std::isinf(a) && std::isinf(b)) {
        return phi_val;
    } else if (std::isinf(a)) {
        return b + sign * dx;
    } else if (std::isinf(b)) {
        return a + sign * dx;
    } else {
        // Check if 2D solution is valid
        double discriminant = 2 * dx * dx - (a - b) * (a - b);
        if (discriminant >= 0) {
            // Full 2D solve
            double phi_new = (a + b + sign * std::sqrt(discriminant)) / 2.0;
            // Check if solution satisfies upwind condition
            if (std::abs(phi_new) > std::max(std::abs(a), std::abs(b))) {
                return phi_new;
            }
        }
        // 1D solution
        return (std::abs(a) < std::abs(b) ? a : b) + sign * dx;
    }
}

torch::Tensor fast_marching_2d(const torch::Tensor& phi_init, double dx) {
    int ny = phi_init.size(0);
    int nx = phi_init.size(1);
    
    auto phi_init_a = phi_init.accessor<double, 2>();
    
    auto phi = phi_init.clone();
    auto status = torch::zeros({ny, nx}, torch::dtype(torch::kInt32));
    
    auto phi_a = phi.accessor<double, 2>();
    auto status_a = status.accessor<int, 2>();
    
    // Find the interface (zero crossings)
    // Mark points near interface as known
    for (int i = 1; i < ny-1; i++) {
        for (int j = 1; j < nx-1; j++) {
            // Check if sign changes in neighborhood
            double center = phi_init_a[i][j];
            double neighbors[4] = {
                phi_init_a[i-1][j],
                phi_init_a[i+1][j],
                phi_init_a[i][j-1],
                phi_init_a[i][j+1]
            };
            
            bool sign_change = false;
            for (int k = 0; k < 4; k++) {
                if (center * neighbors[k] < 0) {
                    sign_change = true;
                    break;
                }
            }
            
            if (sign_change) {
                // Near interface, keep distance unchanged,
                // but mark it as known.
                status_a[i][j] = KNOWN;
            }
        }
    }
    
    std::priority_queue<TrialPoint, std::vector<TrialPoint>, std::greater<TrialPoint>> heap;
    
    int di[4] = {-1, 1, 0, 0};
    int dj[4] = {0, 0, -1, 1};
    
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            if (status_a[i][j] == KNOWN) {
                // Add neighbors to trial
                for (int k = 0; k < 4; k++) {
                    int ni = i + di[k];
                    int nj = j + dj[k];
                    if (ni >= 0 && ni < ny && nj >= 0 && nj < nx && status_a[ni][nj] == FAR) {
                        status_a[ni][nj] = TRIAL;
                        double phi_new = solve_eikonal(phi, phi_init, status, ni, nj, dx);
                        phi_a[ni][nj] = phi_new;
                        heap.push({phi_new, ni, nj});
                    }
                }
            }
        }
    }
    
    while (!heap.empty()) {
        TrialPoint point = heap.top();
        heap.pop();
        
        int i = point.i;
        int j = point.j;
        
        if (status_a[i][j] == KNOWN) {
            continue;
        }
        
        status_a[i][j] = KNOWN;
        
        // Update neighbors
        for (int k = 0; k < 4; k++) {
            int ni = i + di[k];
            int nj = j + dj[k];
            if (ni >= 0 && ni < ny && nj >= 0 && nj < nx && status_a[ni][nj] != KNOWN) {
                double phi_new = solve_eikonal(phi, phi_init, status, ni, nj, dx);
                if (status_a[ni][nj] == FAR) {  // Far -> Trial
                    status_a[ni][nj] = TRIAL;
                    phi_a[ni][nj] = phi_new;
                    heap.push({phi_new, ni, nj});
                } else if (std::abs(phi_new) < std::abs(phi_a[ni][nj])) {  // Already trial, update if better
                    phi_a[ni][nj] = phi_new;
                    heap.push({phi_new, ni, nj});
                }
            }
        }
    }
    
    return phi;
}

torch::Tensor upsample(const torch::Tensor& tensor, int scale_factor) {
    auto t = tensor;
    if (t.dim() == 2) {
        t = t.unsqueeze(0).unsqueeze(0);
    } else if (t.dim() == 3) {
        t = t.unsqueeze(0);
    }
    
    int H = t.size(-2);
    int W = t.size(-1);
    int64_t new_H = H * scale_factor;
    int64_t new_W = W * scale_factor;
    
    auto upsampled = at::upsample_bicubic2d(
        t,
        at::IntArrayRef{new_H, new_W},
        false, // align_corners
        std::nullopt // scale_factor
    );
    
    // Remove batch/channel dimensions if they were added
    while (upsampled.dim() > tensor.dim()) {
        upsampled = upsampled.squeeze();
    }
    
    return upsampled;
}

torch::Tensor downsample(const torch::Tensor& tensor, int scale_factor) {
    auto t = tensor;
    if (t.dim() == 2) {
        t = t.unsqueeze(0).unsqueeze(0);
    } else if (t.dim() == 3) {
        t = t.unsqueeze(0);
    }
    
    // Get dimensions
    int H = t.size(-2);
    int W = t.size(-1);
    int new_H = H / scale_factor;
    int new_W = W / scale_factor;
    
    auto downsampled = at::upsample_bicubic2d(
        t,
        at::IntArrayRef{new_H, new_W},
        false, // align_corners
        std::nullopt // scale_factor
    );
    
    // Remove batch/channel dimensions if they were added
    while (downsampled.dim() > tensor.dim()) {
        downsampled = downsampled.squeeze(0);
    }
    
    return downsampled;
}

// Main SDF reinitialization function
torch::Tensor sdf_reinit(
    torch::Tensor sdf_init,
    float dx,
    int scale_factor,
    float far_threshold
) {
    
    // Handle 2D case by adding temporal dimension
    if (sdf_init.dim() == 2) {
        sdf_init = sdf_init.unsqueeze(0);
    }
    
    TORCH_CHECK(sdf_init.dim() == 3, "SDF must be of shape (T, H, W) or (H, W)");
    
    auto dtype = sdf_init.dtype();
    
    auto reinitialized_sdf = sdf_init.clone();
    int T = sdf_init.size(0);
    
    #pragma omp parallel for
    for (int t = 0; t < T; t++) {
        auto frame = sdf_init[t];
        auto frame_f64 = frame.to(torch::kFloat64);        
        auto upsample_sdf_init = upsample(frame_f64, scale_factor);
        
        // Apply fast marching method to upsampled SDF
        double dx_scaled = dx / static_cast<double>(scale_factor);
        auto upsample_sdf_corrected = fast_marching_2d(upsample_sdf_init, dx_scaled);
        
        auto sdf_corrected = downsample(upsample_sdf_corrected, scale_factor);
        
        // Only reinitialize the SDF at points sufficiently far from the interfaces
        auto far_mask = sdf_corrected < far_threshold;
        auto sdf_corrected_typed = sdf_corrected.to(dtype);
        reinitialized_sdf[t].masked_scatter_(far_mask, sdf_corrected_typed.masked_select(far_mask));
    }
    
    // Remove temporal dimension if it was added
    if (reinitialized_sdf.dim() == 3 && reinitialized_sdf.size(0) == 1) {
        reinitialized_sdf = reinitialized_sdf.squeeze(0);
    }
    return reinitialized_sdf;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SDF Reinitialization using Fast Marching Method";
    m.def("sdf_reinit", &sdf_reinit, "SDF Reinitialization using Fast Marching Method",
        py::arg("sdf_init"),
        py::arg("dx"),
        py::arg("scale_factor") = 8,
        py::arg("far_threshold") = 4.0f
    );
}

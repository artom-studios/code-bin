#include "zenoise/sim/erode2.hpp"
#include "zenoise/sim/erode2_gpu.hpp"
#include "zenoise/sim/gpu_compute.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace zenoise::sim {

namespace {

constexpr float kDepositionRate = 0.50F;
constexpr float kCapacityScale = 0.75F;
constexpr float kFlowDischargeIncrement = 0.05F;
constexpr float kMinimumWaterThreshold = 0.01F;
constexpr float kPitSlopeEpsilon = 1e-6F;
constexpr float kSlopeLengthEpsilon = 1e-7F;

/// Pseudo-random number generator implementing the Park-Miller minimal standard
/// Linear Congruential Generator (LCG).
/// Formula: X_{n+1} = (16807 * X_n) mod (2^31 - 1)
class ParkMillerRng {
public:
    explicit constexpr ParkMillerRng(std::uint32_t seed = 1337U) noexcept
        : state_(seed == 0U ? 1337U : (seed & 0x7FFFFFFFU)) {}

    constexpr std::uint32_t next() noexcept {
        const std::uint32_t hi = (state_ >> 16U) * 0x41A7U;
        const std::uint32_t lo = (state_ & 0xFFFFU) * 0x41A7U;
        std::uint32_t val = (hi & 0x7FFFU) * 0x10000U + lo + (hi >> 15U);
        if (val >= 0x80000000U) {
            val += 0x80000001U; // modulo (2^31 - 1)
        }
        state_ = val;
        return val;
    }

private:
    std::uint32_t state_{1337U};
};

[[nodiscard]] inline float sample_clamped(std::span<const float> grid, std::uint32_t width, std::uint32_t height,
                                          std::int64_t x, std::int64_t z) noexcept {
    const std::uint32_t cx =
        static_cast<std::uint32_t>(std::clamp<std::int64_t>(x, 0, static_cast<std::int64_t>(width - 1U)));
    const std::uint32_t cz =
        static_cast<std::uint32_t>(std::clamp<std::int64_t>(z, 0, static_cast<std::int64_t>(height - 1U)));
    return grid[static_cast<std::size_t>(cz) * width + cx];
}

[[nodiscard]] inline float effective_erosion_scale_meters(float erosion_scale_meters,
                                                          float world_size_meters) noexcept {
    const float ratio = erosion_scale_meters / world_size_meters;
    if (ratio <= 0.2F) {
        return erosion_scale_meters;
    }

    const float adjusted_ratio = 0.2F + (1.0F - 1.0F / ((ratio - 0.2F) * 6.0F + 1.0F)) / 6.0F;
    return world_size_meters * adjusted_ratio;
}

[[nodiscard]] float compute_scale_level(float erosion_scale_meters, float world_size_x, int width) noexcept {
    const float effective_scale = effective_erosion_scale_meters(erosion_scale_meters, world_size_x);
    const float cell_size = world_size_x / static_cast<float>(width);
    const float scale_cells = std::max(1.0F, (effective_scale * 0.5F) / std::max(cell_size, 0.000001F));
    return std::log2(scale_cells);
}

/// Computes an exponential decaying response curve for multiscale pyramid octave weighting.
/// Distributes erosion energy nonlinearly across frequency octaves based on multiscale_bias.
/// Formula: W(t) = exp(-decay * |t - t_peak|) * (1.0 + 0.5 * (1.0 - |t - t_peak|))
[[nodiscard]] inline float erode2_exponential_stage_response(std::size_t level_index, std::size_t num_levels,
                                                             float multiscale_bias) noexcept {
    if (num_levels <= 1U) {
        return 1.0F;
    }
    // level_index 0 = finest resolution (high frequency), num_levels - 1 = coarsest (low frequency)
    const float t = static_cast<float>(level_index) / static_cast<float>(num_levels - 1U);
    // multiscale_bias: 0.0 = low-frequency (coarsest, t=1), 1.0 = high-frequency (finest, t=0), 0.5 = balanced middle
    const float t_peak = 1.0F - std::clamp(multiscale_bias, 0.0F, 1.0F);
    constexpr float kDecayRate = 1.75F;
    const float dist = std::abs(t - t_peak);
    return std::exp(-kDecayRate * dist) * (1.0F + 0.5F * (1.0F - dist));
}

// 2x average downsampling for pyramid multiscale decomposition
[[nodiscard]] std::vector<float> downsample_half(std::span<const float> src, std::uint32_t src_w, std::uint32_t src_h,
                                                 std::uint32_t dst_w, std::uint32_t dst_h) {
    std::vector<float> dst(static_cast<std::size_t>(dst_w) * dst_h, 0.0F);
    for (std::uint32_t y = 0U; y < dst_h; ++y) {
        const std::uint32_t sy0 = y * 2U;
        const std::uint32_t sy1 = std::min(sy0 + 1U, src_h - 1U);
        for (std::uint32_t x = 0U; x < dst_w; ++x) {
            const std::uint32_t sx0 = x * 2U;
            const std::uint32_t sx1 = std::min(sx0 + 1U, src_w - 1U);

            const float v00 = src[static_cast<std::size_t>(sy0) * src_w + sx0];
            const float v10 = src[static_cast<std::size_t>(sy0) * src_w + sx1];
            const float v01 = src[static_cast<std::size_t>(sy1) * src_w + sx0];
            const float v11 = src[static_cast<std::size_t>(sy1) * src_w + sx1];

            dst[static_cast<std::size_t>(y) * dst_w + x] = 0.25F * (v00 + v10 + v01 + v11);
        }
    }
    return dst;
}

// Bilinear 2x upsampling for multiscale displacement interpolation
[[nodiscard]] std::vector<float> upsample_double(std::span<const float> src, std::uint32_t src_w, std::uint32_t src_h,
                                                 std::uint32_t dst_w, std::uint32_t dst_h) {
    std::vector<float> dst(static_cast<std::size_t>(dst_w) * dst_h, 0.0F);
    if (src_w == 0U || src_h == 0U || dst_w == 0U || dst_h == 0U) {
        return dst;
    }

    const float sx_scale = (src_w > 1U) ? static_cast<float>(src_w - 1U) / static_cast<float>(dst_w - 1U) : 0.0F;
    const float sy_scale = (src_h > 1U) ? static_cast<float>(src_h - 1U) / static_cast<float>(dst_h - 1U) : 0.0F;

    for (std::uint32_t y = 0U; y < dst_h; ++y) {
        const float fy = static_cast<float>(y) * sy_scale;
        const auto y0 = static_cast<std::uint32_t>(std::floor(fy));
        const std::uint32_t y1 = std::min(y0 + 1U, src_h - 1U);
        const float ty = fy - static_cast<float>(y0);

        for (std::uint32_t x = 0U; x < dst_w; ++x) {
            const float fx = static_cast<float>(x) * sx_scale;
            const auto x0 = static_cast<std::uint32_t>(std::floor(fx));
            const std::uint32_t x1 = std::min(x0 + 1U, src_w - 1U);
            const float tx = fx - static_cast<float>(x0);

            const float v00 = src[static_cast<std::size_t>(y0) * src_w + x0];
            const float v10 = src[static_cast<std::size_t>(y0) * src_w + x1];
            const float v01 = src[static_cast<std::size_t>(y1) * src_w + x0];
            const float v11 = src[static_cast<std::size_t>(y1) * src_w + x1];

            const float top = v00 + tx * (v10 - v00);
            const float bot = v01 + tx * (v11 - v01);
            dst[static_cast<std::size_t>(y) * dst_w + x] = top + ty * (bot - top);
        }
    }
    return dst;
}

struct Erode2PyramidLevel {
    std::uint32_t width = 0U;
    std::uint32_t height = 0U;
    std::vector<float> base;
    std::vector<float> residual_to_child;
};

struct Erode2ProjectedParent {
    std::vector<float> heightmap;
    std::vector<float> wear;
    std::vector<float> deposition;
};

[[nodiscard]] std::vector<Erode2PyramidLevel> build_erode2_residual_pyramid(const std::vector<float>& input_pixels,
                                                                            std::uint32_t width, std::uint32_t height,
                                                                            std::uint32_t octaves) {
    std::vector<Erode2PyramidLevel> levels;
    levels.reserve(octaves);

    Erode2PyramidLevel full;
    full.width = width;
    full.height = height;
    full.base = input_pixels;
    levels.push_back(std::move(full));

    for (std::uint32_t level = 1U; level < octaves; ++level) {
        Erode2PyramidLevel& parent = levels.back();
        const std::uint32_t child_width = std::max(16U, parent.width / 2U);
        const std::uint32_t child_height = std::max(16U, parent.height / 2U);
        if (child_width == parent.width || child_height == parent.height) {
            break;
        }

        Erode2PyramidLevel child;
        child.width = child_width;
        child.height = child_height;
        child.base = downsample_half(parent.base, parent.width, parent.height, child_width, child_height);

        const std::vector<float> reconstructed =
            upsample_double(child.base, child_width, child_height, parent.width, parent.height);
        parent.residual_to_child.assign(parent.base.size(), 0.0F);
        for (std::size_t i = 0U; i < parent.base.size(); ++i) {
            parent.residual_to_child[i] = parent.base[i] - reconstructed[i];
        }

        levels.push_back(std::move(child));
    }

    return levels;
}

[[nodiscard]] Erode2ProjectedParent
project_erode2_child_to_parent(std::span<const float> child_height, std::span<const float> child_wear,
                               std::span<const float> child_deposition, std::uint32_t child_width,
                               std::uint32_t child_height_px, const Erode2PyramidLevel& parent,
                               float child_cell_height_ratio) {
    const std::vector<float> projected_height =
        upsample_double(child_height, child_width, child_height_px, parent.width, parent.height);
    std::vector<float> projected_deposition =
        upsample_double(child_deposition, child_width, child_height_px, parent.width, parent.height);
    const std::vector<float> projected_wear =
        upsample_double(child_wear, child_width, child_height_px, parent.width, parent.height);

    // Enforce strict physical mass conservation across discrete boundary interpolations
    float sum_proj_wear = 0.0F;
    float sum_proj_dep = 0.0F;
    for (std::size_t i = 0U; i < parent.base.size(); ++i) {
        sum_proj_wear += projected_wear[i];
        sum_proj_dep += projected_deposition[i];
    }
    if (sum_proj_dep > sum_proj_wear && sum_proj_dep > 0.0F) {
        const float fix_factor = sum_proj_wear / sum_proj_dep;
        for (float& d : projected_deposition) {
            d *= fix_factor;
        }
    }

    Erode2ProjectedParent result;
    result.heightmap.resize(parent.base.size(), 0.0F);
    result.wear = projected_wear;
    result.deposition = std::move(projected_deposition);

    for (std::size_t i = 0U; i < parent.base.size(); ++i) {
        const float residual = parent.residual_to_child.empty() ? 0.0F : parent.residual_to_child[i];
        float restored_height = projected_height[i];
        const float carried_deposition = result.deposition[i];

        // Alluvial sediment dampens high-frequency rocky roughness (burying under sediment)
        if (carried_deposition > 0.0F && residual > 0.0F) {
            const float dampening = std::clamp(child_cell_height_ratio / (carried_deposition + 1e-5F), 0.0F, 1.0F);
            restored_height += residual * dampening;
        } else {
            restored_height += residual;
        }

        result.heightmap[i] = restored_height;
    }

    return result;
}

// Thread Safety: Pure functional pass operating on privately owned buffers.
bool run_droplet_pass(std::span<float> elevation, std::span<float> wear_mask, std::span<float> deposition_mask,
                      std::span<float> flow_mask, const float* hardness_mask, std::uint32_t width, std::uint32_t height,
                      const Erode2Settings& settings, const terrain::WorldSpec& world, std::uint32_t seed_offset,
                      const std::stop_token& stop_token) {
    if (width < 2U || height < 2U || settings.amount <= 0.0F) {
        return true;
    }

    // Maximum steps per droplet: (2 * width) / 3
    const std::int32_t max_steps = std::max<std::int32_t>(8, static_cast<std::int32_t>((2U * width) / 3U));

    // Resolution-independent scaling against 512x512 reference grid
    float resolution_scale = 1.0F;
    if (settings.scale_independence) {
        const float cell_dx = width > 1U ? world.size_x_meters / static_cast<float>(width - 1U) : 1.0F;
        const float ref_scale = std::max(1.0F, settings.erosion_scale_meters * 0.02F);
        resolution_scale = std::clamp(cell_dx / ref_scale, 0.25F, 4.0F);
    }

    const std::size_t total_cells = static_cast<std::size_t>(width) * height;
    const auto droplet_count =
        static_cast<std::size_t>(static_cast<float>(total_cells) * settings.amount * 2.5F * resolution_scale);

    ParkMillerRng rng(settings.seed + seed_offset);

    for (std::size_t d = 0U; d < droplet_count; ++d) {
        if ((d & 0xFFU) == 0U && stop_token.stop_requested()) {
            return false;
        }

        auto px = static_cast<float>(rng.next() % width);
        auto py = static_cast<float>(rng.next() % height);

        float sediment = 0.0F;
        float water = 1.0F;
        float dir_x = 0.0F;
        float dir_y = 0.0F;
        float angular_momentum = 0.0F;

        for (std::int32_t step = 0; step < max_steps; ++step) {
            const auto ix = static_cast<std::int64_t>(px);
            const auto iy = static_cast<std::int64_t>(py);

            if (ix < 0 || ix >= static_cast<std::int64_t>(width) || iy < 0 || iy >= static_cast<std::int64_t>(height)) {
                break;
            }

            const auto uix = static_cast<std::uint32_t>(ix);
            const auto uiy = static_cast<std::uint32_t>(iy);

            if (settings.preserve_edges) {
                if (uix == 0U || uix == width - 1U || uiy == 0U || uiy == height - 1U) {
                    break;
                }
            }

            const std::size_t cell_idx = static_cast<std::size_t>(uiy) * width + uix;
            const float current_elev = elevation[cell_idx];

            // Central finite differences along cardinal directions
            const float z_west = sample_clamped(elevation, width, height, ix - 1, iy);
            const float z_east = sample_clamped(elevation, width, height, ix + 1, iy);
            const float z_north = sample_clamped(elevation, width, height, ix, iy - 1);
            const float z_south = sample_clamped(elevation, width, height, ix, iy + 1);

            const float grad_west = current_elev - z_west;
            const float grad_east = current_elev - z_east;
            const float grad_north = current_elev - z_north;
            const float grad_south = current_elev - z_south;

            // Pit depression detection
            if (grad_west <= kPitSlopeEpsilon && grad_east <= kPitSlopeEpsilon && grad_north <= kPitSlopeEpsilon &&
                grad_south <= kPitSlopeEpsilon) {
                if (sediment > 0.0F) {
                    elevation[cell_idx] += sediment;
                    deposition_mask[cell_idx] += sediment;
                }
                break;
            }

            float f11 = grad_east;
            float f17 = grad_west;
            if (f11 < f17) {
                f11 = -f17;
            }

            float f13 = grad_south;
            float f16 = grad_north;
            if (f13 < f16) {
                f13 = -f16;
            }

            float slope_len = std::sqrt(f13 * f13 + f11 * f11);
            if (slope_len == 0.0F) {
                f11 = 0.0F;
                f13 = 0.0F;
            } else {
                f11 /= slope_len;
                f13 /= slope_len;
            }

            // Directional jitter breakup using isotropic 8-neighbor Sobel and stochastic lateral deflection
            if (settings.jitter > 0.0F) {
                const float z_nw = sample_clamped(elevation, width, height, ix - 1, iy - 1);
                const float z_ne = sample_clamped(elevation, width, height, ix + 1, iy - 1);
                const float z_sw = sample_clamped(elevation, width, height, ix - 1, iy + 1);
                const float z_se = sample_clamped(elevation, width, height, ix + 1, iy + 1);

                // Isotropic 3x3 Sobel gradient operator (downhill direction = -gradient)
                const float gx = (z_ne + 2.0F * z_east + z_se) - (z_nw + 2.0F * z_west + z_sw);
                const float gy = (z_sw + 2.0F * z_south + z_se) - (z_nw + 2.0F * z_north + z_ne);
                const float sobel_len = std::sqrt(gx * gx + gy * gy);
                if (sobel_len > 1e-7F) {
                    f11 = -gx / sobel_len;
                    f13 = -gy / sobel_len;
                    slope_len = sobel_len * 0.125F; // Normalization factor for 1/8 Sobel kernel
                }

                // Lateral Brownian perturbation perpendicular to gradient: u_perp = (-f13, f11)
                const float rand_val = (static_cast<float>(rng.next() % 2001U) - 1000.0F) * 0.001F;
                const float perp_x = -f13;
                const float perp_y = f11;
                f11 += perp_x * (settings.jitter * rand_val * 0.5F);
                f13 += perp_y * (settings.jitter * rand_val * 0.5F);
                const float perturbed_len = std::sqrt(f11 * f11 + f13 * f13);
                if (perturbed_len > 1e-7F) {
                    f11 /= perturbed_len;
                    f13 /= perturbed_len;
                }
            }

            // Clean-room centrifugal sinuosity meander dynamics
            if (settings.sinuosity > 0.0F && step > 0) {
                // Instantaneous streamline turning torque (2D cross product: u_prev x g)
                const float turn = dir_x * f13 - dir_y * f11;
                angular_momentum = angular_momentum * 0.80F + turn * 0.80F;
                const float perp_x = -f13;
                const float perp_y = f11;
                f11 += perp_x * (angular_momentum * settings.sinuosity * 1.5F);
                f13 += perp_y * (angular_momentum * settings.sinuosity * 1.5F);
                const float curved_len = std::sqrt(f11 * f11 + f13 * f13);
                if (curved_len > 1e-7F) {
                    f11 /= curved_len;
                    f13 /= curved_len;
                }
            }

            // Droplet momentum blending (inertia):
            // formula: d = v_prev * inertia + g * (1.0 - inertia)
            if (settings.inertia > 0.0F && step > 0) {
                const float blended_x = dir_x * settings.inertia + f11 * (1.0F - settings.inertia);
                const float blended_y = dir_y * settings.inertia + f13 * (1.0F - settings.inertia);
                const float blended_len = std::sqrt(blended_x * blended_x + blended_y * blended_y);
                if (blended_len > 1e-7F) {
                    f11 = blended_x / blended_len;
                    f13 = blended_y / blended_len;
                }
            }
            dir_x = f11;
            dir_y = f13;

            const float chebyshev_comp = std::max(std::abs(f11), std::abs(f13));
            if (chebyshev_comp <= 1e-7F) {
                break;
            }
            // When jitter > 0, blend between Chebyshev step normalization and Euclidean unit step length
            const float max_comp = (1.0F - settings.jitter) * chebyshev_comp + settings.jitter * 1.0F;

            // Sediment carry capacity formula matching native WM LTE:
            // C = Slope * C_SCALE (0.75) * sediment_capacity * water
            const float carry_capacity = slope_len * kCapacityScale * settings.sediment_capacity * water;

            // Spatial rock hardness modulation: H in [0.0 = soft/erodible, 1.0 = resistant]
            float hardness = settings.rock_hardness;
            if (hardness_mask != nullptr) {
                hardness = std::clamp(hardness_mask[cell_idx] * settings.rock_hardness, 0.0F, 1.0F);
            }

            if (carry_capacity <= sediment) {
                // Deposition of excess sediment: delta_s = (sediment - Capacity) * k_dep (0.50)
                const float deposit_amount = (sediment - carry_capacity) * kDepositionRate;
                sediment -= deposit_amount;
                elevation[cell_idx] += deposit_amount;
                deposition_mask[cell_idx] += deposit_amount;
            } else {
                // Bedrock detachment matching FUN_180128000:
                // delta_s = (1.0 - H) * 0.50 * (Capacity - sediment)
                const float erode_amount = (1.0F - hardness) * 0.5F * (carry_capacity - sediment);
                sediment += erode_amount;
                elevation[cell_idx] -= erode_amount;
                wear_mask[cell_idx] += erode_amount;
            }

            // Move droplet downstream along dominant slope axis (Chebyshev normalization)
            px += f11 * (1.0F / max_comp);
            py += f13 * (1.0F / max_comp);

            // Accumulate water flow discharge matching native WM LTE
            flow_mask[cell_idx] += kFlowDischargeIncrement;
        }
    }

    return true;
}

// Shields stream power valley incision
// Formula: tau = sqrt(Area - Area_thresh) * PhysicalSlope
void apply_channeled_incision(std::span<float> elevation, std::span<float> wear_mask, std::span<const float> flow_mask,
                              const float* hardness_mask, std::uint32_t width, std::uint32_t height,
                              const Erode2Settings& settings, const terrain::WorldSpec& world) {
    if (width < 2U || height < 2U || settings.channel_depth <= 0.0F) {
        return;
    }

    float max_flow = 1e-6F;
    for (const float f : flow_mask) {
        max_flow = std::max(max_flow, f);
    }

    const float inv_max_flow = 1.0F / max_flow;
    const float drain_thresh = std::clamp(settings.drainage_area_threshold, 0.001F, 0.5F);
    const float cell_dx = width > 1U ? world.size_x_meters / static_cast<float>(width - 1U) : 1.0F;
    const float cell_dz = height > 1U ? world.size_z_meters / static_cast<float>(height - 1U) : 1.0F;
    const float elev_scale = world.max_height_meters;

    for (std::uint32_t y = 1U; y + 1U < height; ++y) {
        for (std::uint32_t x = 1U; x + 1U < width; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y) * width + x;
            const float flow_val = flow_mask[idx];
            const float norm_area = flow_val * inv_max_flow;

            if (norm_area < drain_thresh) {
                continue;
            }

            const float z_c = elevation[idx];
            const float dz_dx = (elevation[idx + 1U] - elevation[idx - 1U]) * 0.5F * (elev_scale / cell_dx);
            const float dz_dy = (elevation[idx + width] - elevation[idx - width]) * 0.5F * (elev_scale / cell_dz);
            const float physical_slope = std::sqrt(dz_dx * dz_dx + dz_dy * dz_dy);

            if (physical_slope < settings.channel_gradient) {
                continue;
            }

            float hardness = settings.rock_hardness;
            if (hardness_mask != nullptr) {
                hardness = std::clamp(hardness_mask[idx] * settings.rock_hardness, 0.0F, 1.0F);
            }

            // Stream power: tau = sqrt(Area - Area_thresh) * Slope
            const float stream_power = std::sqrt(norm_area - drain_thresh) * physical_slope;

            // Normalized channel incision depth
            float incision = stream_power * settings.channel_depth * 0.02F * (1.0F - hardness);

            // Floodplain and base riverbed datum deepening
            if (settings.river_depth > 0.0F && norm_area > drain_thresh * 1.5F) {
                const float river_carve =
                    settings.river_depth * (norm_area - drain_thresh * 1.5F) * (1.0F + settings.river_bias);
                incision += river_carve * 0.05F * (1.0F - hardness);
            }

            incision = std::min(incision, z_c * 0.5F); // Prevent terrain inversion below datum
            if (incision > 0.0F) {
                elevation[idx] -= incision;
                wear_mask[idx] += incision;
            }
        }
    }
}

} // namespace

std::string validate_erode2_settings(const Erode2Settings& settings, const terrain::WorldSpec& world) {
    if (!std::isfinite(settings.amount) || settings.amount < 0.0F || settings.amount > 2.0F) {
        return "Erode2 amount must be a finite value in [0, 2].";
    }
    if (!std::isfinite(settings.rock_hardness) || settings.rock_hardness < 0.0F || settings.rock_hardness > 1.0F) {
        return "Erode2 rock hardness must be a finite value in [0, 1].";
    }
    if (!std::isfinite(settings.sediment_capacity) || settings.sediment_capacity < 0.0F ||
        settings.sediment_capacity > 2.0F) {
        return "Erode2 sediment capacity must be a finite value in [0, 2].";
    }
    if (!std::isfinite(settings.channel_depth) || settings.channel_depth < 0.0F || settings.channel_depth > 2.0F) {
        return "Erode2 channel depth must be a finite value in [0, 2].";
    }
    if (!std::isfinite(settings.river_depth) || settings.river_depth < 0.0F || settings.river_depth > 2.0F) {
        return "Erode2 river depth must be a finite value in [0, 2].";
    }
    if (!std::isfinite(settings.river_bias) || settings.river_bias < -1.0F || settings.river_bias > 1.0F) {
        return "Erode2 river bias must be a finite value in [-1, 1].";
    }
    if (!std::isfinite(settings.channel_gradient) || settings.channel_gradient < 0.0F ||
        settings.channel_gradient > 10.0F) {
        return "Erode2 channel gradient must be a finite value in [0, 10].";
    }
    if (!std::isfinite(settings.drainage_area_threshold) || settings.drainage_area_threshold < 0.001F ||
        settings.drainage_area_threshold > 0.5F) {
        return "Erode2 drainage area threshold must be a finite value in [0.001, 0.5].";
    }
    if (!std::isfinite(settings.multiscale_bias) || settings.multiscale_bias < 0.0F ||
        settings.multiscale_bias > 1.0F) {
        return "Erode2 multiscale bias must be a finite value in [0, 1].";
    }
    if (settings.multiscale_octaves < 1U || settings.multiscale_octaves > 8U) {
        return "Erode2 multiscale octaves must be between 1 and 8.";
    }
    if (!std::isfinite(settings.inertia) || settings.inertia < 0.0F || settings.inertia > 1.0F) {
        return "Erode2 inertia must be a finite value in [0, 1].";
    }
    if (!std::isfinite(settings.jitter) || settings.jitter < 0.0F || settings.jitter > 1.0F) {
        return "Erode2 jitter must be a finite value in [0, 1].";
    }
    if (!std::isfinite(settings.sinuosity) || settings.sinuosity < 0.0F || settings.sinuosity > 1.0F) {
        return "Erode2 sinuosity must be a finite value in [0, 1].";
    }
    if (!std::isfinite(settings.erosion_scale_meters) || settings.erosion_scale_meters < 1.0F ||
        settings.erosion_scale_meters > 100000.0F) {
        return "Erode2 erosion scale meters must be a finite value in [1, 100000].";
    }
    return terrain::validate_world_spec(world);
}

Erode2Result run_erode2_simulation(const terrain::HeightField& input, const terrain::HeightField* hardness_mask,
                                   const terrain::WorldSpec& world, const Erode2Settings& settings,
                                   std::stop_token stop_token) {
    const std::string validation_error = validate_erode2_settings(settings, world);
    if (!validation_error.empty()) {
        return {Erode2Status::failed, nullptr, nullptr, nullptr, nullptr, validation_error};
    }

    const terrain::HeightFieldDimensions dims = input.dimensions();
    const std::uint32_t width = dims.width;
    const std::uint32_t height = dims.height;

    auto output = std::make_shared<terrain::HeightField>(dims, 0.0F);
    auto wear = std::make_shared<terrain::HeightField>(dims, 0.0F);
    auto deposition = std::make_shared<terrain::HeightField>(dims, 0.0F);
    auto flow = std::make_shared<terrain::HeightField>(dims, 0.0F);

    std::ranges::copy(input.samples(), output->writable_samples().begin());

    if (settings.amount <= 0.0F || width < 2U || height < 2U) {
        return {Erode2Status::succeeded, std::move(output), std::move(wear),
                std::move(deposition),   std::move(flow),   {}};
    }

    GpuBackend* const gpu = GpuComputeContext::instance().backend();
    const GpuExecutionMode mode = GpuComputeContext::instance().execution_mode();
    const bool gpu_enabled = GpuComputeContext::instance().is_gpu_enabled();
    if (gpu != nullptr && gpu_enabled && mode != GpuExecutionMode::cpu_fallback && width >= 8U && width == height) {
        Erode2Result gpu_result = run_erode2_gpu(input, hardness_mask, world, settings, stop_token);
        if (gpu_result.status == Erode2Status::succeeded || gpu_result.status == Erode2Status::cancelled) {
            return gpu_result;
        }
        if (mode == GpuExecutionMode::force_gpu) {
            return gpu_result;
        }
        // Fall through to CPU path on GPU failure in auto_detect mode
    } else if (mode == GpuExecutionMode::force_gpu) {
        return {Erode2Status::failed,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                "GPU execution was forced, but GPU compute backend is unavailable or disabled."};
    }

    const float* const hardness_ptr =
        (hardness_mask != nullptr && hardness_mask->sample_count() == output->sample_count())
            ? hardness_mask->samples().data()
            : nullptr;

    std::uint32_t octaves = 1U;
    if (settings.multiscale_enable && width >= 32U && height >= 32U) {
        const float scale_level =
            compute_scale_level(settings.erosion_scale_meters, world.size_x_meters, static_cast<int>(width));
        const auto derived = static_cast<std::uint32_t>(std::max(1, static_cast<int>(std::ceil(scale_level))));
        octaves = std::clamp(derived, 1U, settings.multiscale_octaves);
    }

    const std::vector<float> input_samples(input.samples().begin(), input.samples().end());
    const std::vector<Erode2PyramidLevel> levels = build_erode2_residual_pyramid(input_samples, width, height, octaves);
    const std::size_t num_levels = levels.size();

    // Downsampled hardness mask pyramid
    std::vector<std::vector<float>> hardness_pyramid(num_levels);
    if (hardness_ptr != nullptr) {
        hardness_pyramid[0].assign(hardness_mask->samples().begin(), hardness_mask->samples().end());
        for (std::size_t i = 1U; i < num_levels; ++i) {
            hardness_pyramid[i] = downsample_half(hardness_pyramid[i - 1U], levels[i - 1U].width, levels[i - 1U].height,
                                                  levels[i].width, levels[i].height);
        }
    }

    std::vector<float> processed_child_height;
    std::vector<float> processed_child_wear;
    std::vector<float> processed_child_deposition;
    std::vector<float> processed_child_flow;
    std::uint32_t processed_child_w = 0U;
    std::uint32_t processed_child_h = 0U;

    for (int level_idx = static_cast<int>(num_levels) - 1; level_idx >= 0; --level_idx) {
        if (stop_token.stop_requested()) {
            return {Erode2Status::cancelled, nullptr, nullptr, nullptr, nullptr, "Erode2 cancelled."};
        }

        const auto ulevel = static_cast<std::size_t>(level_idx);
        const Erode2PyramidLevel& cur_level = levels[ulevel];
        const std::size_t cell_count = static_cast<std::size_t>(cur_level.width) * cur_level.height;

        std::vector<float> cur_height;
        std::vector<float> cur_wear;
        std::vector<float> cur_deposition;
        std::vector<float> cur_flow;

        if (processed_child_height.empty()) {
            cur_height = cur_level.base;
            cur_wear.assign(cell_count, 0.0F);
            cur_deposition.assign(cell_count, 0.0F);
            cur_flow.assign(cell_count, 0.0F);
        } else {
            const float cell_ratio =
                (world.size_x_meters / static_cast<float>(processed_child_w)) / world.max_height_meters;
            Erode2ProjectedParent proj =
                project_erode2_child_to_parent(processed_child_height, processed_child_wear, processed_child_deposition,
                                               processed_child_w, processed_child_h, cur_level, cell_ratio);
            cur_height = std::move(proj.heightmap);
            cur_wear = std::move(proj.wear);
            cur_deposition = std::move(proj.deposition);
            // Upsample coarse flow to seed parent drainage paths
            cur_flow = upsample_double(processed_child_flow, processed_child_w, processed_child_h, cur_level.width,
                                       cur_level.height);
        }

        const float* const cur_hardness = hardness_ptr != nullptr ? hardness_pyramid[ulevel].data() : nullptr;

        // Level settings with multiscale bias weighting
        Erode2Settings level_settings = settings;
        if (num_levels > 1U) {
            float weight = 1.0F;
            if (settings.exponential_octaves) {
                weight = erode2_exponential_stage_response(ulevel, num_levels, settings.multiscale_bias);
            } else {
                const float t = static_cast<float>(level_idx) / static_cast<float>(num_levels - 1U);
                weight = (1.0F - settings.multiscale_bias) * (1.0F - 0.5F * t) +
                         settings.multiscale_bias * (0.5F + 0.5F * t);
            }
            level_settings.amount = settings.amount * weight;
            level_settings.channel_depth = settings.channel_depth * weight;
            level_settings.river_depth = settings.river_depth * weight;
        }

        const std::uint32_t seed_offset = static_cast<std::uint32_t>(level_idx) * 10007U;

        // Step 1: Droplet simulation on current level
        const bool droplet_ok =
            run_droplet_pass(cur_height, cur_wear, cur_deposition, cur_flow, cur_hardness, cur_level.width,
                             cur_level.height, level_settings, world, seed_offset, stop_token);
        if (!droplet_ok) {
            return {Erode2Status::cancelled, nullptr, nullptr, nullptr, nullptr, "Erode2 cancelled."};
        }

        if (stop_token.stop_requested()) {
            return {Erode2Status::cancelled, nullptr, nullptr, nullptr, nullptr, "Erode2 cancelled."};
        }

        // Step 2: Stream power incision on current level
        if (level_settings.type == Erode2Type::channeled || level_settings.type == Erode2Type::deep_incision) {
            apply_channeled_incision(cur_height, cur_wear, cur_flow, cur_hardness, cur_level.width, cur_level.height,
                                     level_settings, world);
        }

        if (level_idx == 0) {
            std::ranges::copy(cur_height, output->writable_samples().begin());
            std::ranges::copy(cur_wear, wear->writable_samples().begin());
            std::ranges::copy(cur_deposition, deposition->writable_samples().begin());
            std::ranges::copy(cur_flow, flow->writable_samples().begin());
        } else {
            processed_child_height = std::move(cur_height);
            processed_child_wear = std::move(cur_wear);
            processed_child_deposition = std::move(cur_deposition);
            processed_child_flow = std::move(cur_flow);
            processed_child_w = cur_level.width;
            processed_child_h = cur_level.height;
        }
    }

    // Step 3: Strict Output Normalization
    for (float& val : output->writable_samples()) {
        val = std::clamp(val, 0.0F, 1.0F);
    }
    for (float& val : wear->writable_samples()) {
        val = std::clamp(val, 0.0F, 1.0F);
    }
    for (float& val : deposition->writable_samples()) {
        val = std::clamp(val, 0.0F, 1.0F);
    }
    for (float& val : flow->writable_samples()) {
        val = std::clamp(val, 0.0F, 1.0F);
    }

    return {Erode2Status::succeeded, std::move(output), std::move(wear), std::move(deposition), std::move(flow), {}};
}

} // namespace zenoise::sim

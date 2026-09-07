#include <cmath>
#include <cstddef>
#include <numbers>

#define SS 5
#define boundaryCondition 0
// 0 periodic
// 1 noFlux
// 2 dirichlet
namespace Parameters {
enum class model {
  modelA,
  modelB,
  modelCALPHAD,
};
inline constexpr auto simModel{model::modelCALPHAD};
inline constexpr bool ifTFC{true};
inline constexpr bool if_noise{true};
inline constexpr bool if_Vp_ramp{false};
inline constexpr bool if_output_field_history{true};
inline constexpr bool if_start_from_step0{true};
inline constexpr bool if_1_eps{true}; // 1: eps1, 0: (eps1, eps2)
using real = double;

namespace Material {
using std::numbers::sqrt2;
inline constexpr real a1{2.0 * sqrt2 / 3.0};
inline constexpr real cinf{0.632514105}; // at% Ti cs = 0.515285917
inline constexpr int S{SS};              // S = W/W0, INT
inline constexpr real W0{0.25};          // nm, capillary length
inline constexpr real Gamma{199.0};
// K*nm, Gibbs-Thomson coefficient. Taken as reasonable
// compromise between 203.359 and 194.517.
inline constexpr real muk0{0.57};
// nm/ns/K, interface kinetic coefficient,
// unprincpled selection between 0.7 and 0.49
inline constexpr real tau0{(S * S * W0 * W0) / (Gamma * muk0)};
// ns, 5^2/196/0.5 = 0.2551 ns
inline constexpr real Tmelt{2161.59};           // K, liquidus temperature
inline constexpr real deltaT0{Tmelt - 2118.35}; // K, Tmelt - T0
inline constexpr real Dl{2.05};
// nm^2/ns, liquid diffusion coefficient, μm^2/s = 1e-3 nm^2/ns
// #define    GG     (1.0e-3)  // K/nm, temperature gradient
// #define    Vp      1.0 // m/s

// #define    eps1   0.06 // capillary anisotropy strength
// #define    eps2   (0) // capillary anisotropy strength
// #define    epk1   0.15   // kinetic anisotropy strength

// only for dilute limit model
inline constexpr real ke{0.8146631243899296}; // partition coefficient
inline real bb{0.5 * log(ke)};
inline constexpr real d0{Gamma / deltaT0};
inline real lambda{-bb * a1 * S * W0 / d0 / (1.0 - ke)};

inline constexpr real latentHeat{22003.0};  // J/mol, latent heat
inline constexpr real molar_weight{64.418}; // g/mol, molar average
inline constexpr real num_moles{93353.43470588748};
// mol/m^3, reciprocal of molar average of volumes
inline constexpr real cp{40.0};
// J/mol/K, specific heat, approximating between the solid and
// liquid heat capacities of the alloy
inline constexpr real Lcp{latentHeat / cp}; // K, // (340.5)
inline constexpr real density{1.0e-6 * num_moles * molar_weight}; // g/cm^3
inline constexpr real latentHeat2{latentHeat * num_moles}; // J/m^3, latent heat
inline constexpr real surface_tension{Gamma * 1e-9 * latentHeat2 / Tmelt};
// 0.145811 // J/m^2, solid-liquid interfacial free-energy ...

inline constexpr real RR{8.3145}; // J/mol/K, gas constant
inline constexpr real v0{
    1.17e-5}; // old(1.071e-5) // m^3/mol, average molar volume
inline constexpr real hh{surface_tension / (a1 * S * W0 * 1.0e-9)}; // J/m^3
inline constexpr real h0{RR * Tmelt / v0};                          // J/m^3
} // namespace Material

namespace Simulation {
/* For the following constants:
 *
 * Ny-2 needs to be divisible by 10.
 *
 * i_tip_target is the maintained position of the solid-liquid interface. Its
 * difference with Nx should be > 5*Dl/Vp.
 *
 * di and dj make the output results sparser the larger they are: we only out
 * put every di points along the x-axis and every dj points along the y-axis.
 *
 * for the BLOCK_SIZEs, a few combos tend to work better than others, due to the
 * warp size on the GPUs we tend to use:
 * 32x8 or 32x4 is best, 64x4 is good, but 16x16 is bad and should be avoided.*/
// #define Nx_all 768
// #define Ny_all 832
inline constexpr std::size_t Nx{0};
inline constexpr std::size_t Ny{0};

#if SS == 1
inline constexpr real dx{0.8};
inline constexpr real AA{1.0};
#endif
#if SS == 3
inline constexpr real dx{0.6};
inline constexpr real AA{6.0};
#endif
#if SS == 5
inline constexpr real dx{0.6};
inline constexpr real AA{12.0};
#endif

inline constexpr std::size_t i_tip_target{Nx - 268};
inline constexpr std::size_t di{1};
inline constexpr std::size_t dj{1};
inline constexpr std::size_t BLOCK_SIZE_X{32};
inline constexpr std::size_t BLOCK_SIZE_Y{8};

// #define    total_time    (100*1000) // ns
inline constexpr std::size_t num_fields_output{120}; // output fields
inline constexpr std::size_t num_tip_output{10};     // output tip information
inline constexpr std::size_t num_check_point_output{120}; // output check points
// #define    run_time (8) // hours
// #define    if_load  0 // 0: start from initial; 1: read check point
// #define    path_input  "../eps0.02_Vp1_gamma0_gamma0/"

using std::numbers::pi;
inline constexpr std::size_t num_orientation{1}; // number of orientations

//////////// temperature field calculation
inline constexpr real TCOLD{2000.0};
inline constexpr std::size_t ratio_dxT_dx{10};
inline constexpr real dxT{dx * ratio_dxT_dx};

inline constexpr real length_sample{(Nx - 2) * SS * Material::W0 * dx};
inline constexpr std::size_t NxT{(Nx - 2) / ratio_dxT_dx + 2};
inline constexpr std::size_t NyT{(Ny - 2) / ratio_dxT_dx + 2};

inline constexpr real DT{1.1e4}; // (1.2885e4) // nm^2/ns

inline constexpr std::size_t BLOCK_SIZE_T_X{64};
inline constexpr std::size_t BLOCK_SIZE_T_Y{4};

// if there are many random nulei in the initial condition
inline constexpr std::size_t initial_nuclei_border_i{i_tip_target};
// all grains are at the left
inline constexpr std::size_t num_nuclei_target{4000};
inline constexpr real initial_max_radius{6.0 / dx}; // radius of initial nuclei
inline constexpr int j_border{Ny / 2};
// initial j value at the border of grains 1 and 2

// #define    random_seed 0
inline constexpr real noise_amplitude{0.3};
inline constexpr real phi_interface{0.0}; // phi value at the interface

// projections of crystal axes on the lab coordinates
// 1-9: x'=(x1,x2,x3), y'=(y1,y2,y3), z'=(z1,z2,z3)
inline constexpr real num_axes{9}; // orientation(num_orientation, num_axes)
inline constexpr real alpha0{0.0}; // [0, 45], Euler angles
inline constexpr real beta0{0.0};  // [0, 45]
inline constexpr real gamma0{0.0}; // [0, 45]
inline constexpr real alpha1{0.0}; // [0, 45]
inline constexpr real beta1{0.0};  // [0, 45]
inline constexpr real gamma1{0.0}; // [0, 45]

// boundary condition
inline constexpr std::size_t imin{1};
inline constexpr std::size_t imax{Nx - 2};
inline constexpr std::size_t iTmin{1};
inline constexpr std::size_t iTmax{NxT - 2};

#if boundaryCondition == 0 // periodic
inline constexpr std::size_t jmin{Ny - 2};
inline constexpr std::size_t jmax{1};
inline constexpr std::size_t jTmin{NyT - 2};
inline constexpr std::size_t jTmax{1};
#endif

#if boundaryCondition == 1 // noFlux
inline constexpr std::size_t jmin{1};
inline constexpr std::size_t jmax{Ny - 2};
inline constexpr std::size_t jTmin{1};
inline constexpr std::size_t jTmax{NyT - 2};
#endif
// temperature oscillation
inline constexpr int OSCILLATION{0};
// ( osc_A*sin(-osc_omgea*step*dt*tau0)/undercooling )
inline constexpr real osc_A{2.0}; // K, amplitude of oscillation
inline constexpr real osc_omega{2.0 * pi / 400.0};
// 3.6941e-4 ns^-1, frequency, t0 = Dh/V^2 = 17 μs,
// pi/w/tau0/dt = 3.14/3.6941e-4 = 4.16e5 steps
inline constexpr real osc_b{20.0}; // nm, decaying distance, can be tip radius
// #define    Dh    (1.7e4) // nm^2/ns, diffusivity of T, from Pinomaa et al,
// 2020
} // namespace Simulation
} // namespace Parameters

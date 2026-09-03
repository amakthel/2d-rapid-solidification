#ifndef SIMULATOR_MAIN_HPP_
#define SIMULATOR_MAIN_HPP_
#include <numbers>
namespace Parameters {
    enum Class model {
    modelA,
    modelB,
    modelCALPHAD,
}
        namespace Simulation {
            inline constexpr auto simModel{model::modelCALPHAD};
            inline constexpr bool ifTFC{true};
            inline constexpr bool if_noise{true};
            inline constexpr bool if_Vp_ramp{false};
            inline constexpr bool if_output_field_history{true};
            inline constexpr bool if_start_from_step0{true};
inline constexpr bool if_1_eps{true}; // 1: eps1, 0: (eps1, eps2)
            // #define Nx_all 768
            // #define Ny_all 832
            inline constexpr size_t Nx{Nx_all};
inline constexpr size_t Ny{Ny_all}; // Ny-2 should be divided by ratio_dxT_dx=10
inline constexpr size_t i_tip_target{Nx-268}; // position of the solid-liquid interface;  its distance to Nx should be > ~ 5*Dl/V
inline constexpr size_t di{1}; // output result for every di points
inline constexpr size_t dj{1}; // output result for every di points
inline constexpr size_t BLOCK_SIZE_X{32}; // 32x8 32x4 best; 64x4 good; 16x16 bad
inline constexpr size_t BLOCK_SIZE_Y{8};

// #define    total_time    (100*1000) // ns
            inline constexpr size_t num_fields_output{120}; // output fields
inline constexpr size_t num_tip_output{10};   // output tip information
inline constexpr size_t num_check_point_output{120}; // output check points
// #define    run_time (8) // hours
// #define    if_load  0 // 0: start from initial; 1: read check point
// #define    path_input  "../eps0.02_Vp1_gamma0_gamma0/"

            using real = double;
            using std::numbers::sqrt2;
            using std::numbers::pi;
            inline constexpr real a1{2.0*sqrt2/3.0};
            inline constexpr size_t num_orientation{1}; // number of orientations

inline constexpr cinf{0.632514105};  // at% Ti cs = 0.515285917
#define    SS      5     // S = W/W0, INT
#define    W0      0.25    // nm, capillary length
#define    Gamma   (199.0)   // K*nm, Gibbs-Thomson coefficient. Taken as reasonable compromise between 203.359 and 194.517.
#define    muk0    (0.57)  // nm/ns/K, interface kinetic coefficient, unprincpled selection between 0.7 and 0.49
#define    tau0    ( pow2(SS*W0)/(Gamma*muk0) ) // ns, 5^2/196/0.5 = 0.2551 ns
#define    Tmelt   (2161.59) // K, liquidus temperature
#define    deltaT0 (Tmelt-2118.35) // K, Tmelt - T0
#define    Dl      (2.05)    // nm^2/ns, liquid diffusion coefficient, μm^2/s = 1e-3 nm^2/ns
// #define    GG     (1.0e-3)  // K/nm, temperature gradient
// #define    Vp      1.0 // m/s

// #define    eps1   0.06 // capillary anisotropy strength
// #define    eps2   (0) // capillary anisotropy strength
// #define    epk1   0.15   // kinetic anisotropy strength

// only for dilute limit model
#define    ke    0.8146631243899296 // partition coefficient
#define    bb     (0.5*log(ke))
#define    d0     (Gamma/deltaT0) //
#define    lambda   (-bb*a1*SS*W0/d0/(1.0-ke)) //

#define    latentHeat (22003.0) // J/mol, latent heat
#define    molar_weight (64.418) // g/mol, molar average
#define    num_moles (93353.43470588748) // mol/m^3, reciprocal of molar average of volumes
#define    cp (40.0) // J/mol/K, specific heat, approximating between the solid and liquid heat capacities of the alloy
#define    Lcp (latentHeat/cp) // K, // (340.5)
#define    density  (1.0e-6*num_moles*molar_weight) // g/cm^3
#define    latentHeat2 (latentHeat*num_moles) // J/m^3, latent heat
#define    surface_tension (Gamma*1e-9*latentHeat2/Tmelt) // 0.145811 // J/m^2, solid-liquid interfacial free-energy ...

#define    RR 8.3145 // J/mol/K, gas constant
#define    v0 (1.17e-5) // old(1.071e-5) // m^3/mol, average molar volume
#define    hh ( surface_tension/(a1*SS*W0*1.0e-9) ) // J/m^3
#define    h0 ( RR*Tmelt/v0 ) // J/m^3

//////////// temperature field calculation
#define    TCOLD (2000.0)
#define    ratio_dxT_dx 10
#define    dxT  (dx*ratio_dxT_dx)
#define    DT  (1.1e4) // (1.2885e4) // nm^2/ns

#define    BLOCK_SIZE_T_X 64
#define    BLOCK_SIZE_T_Y 4

// if there are many random nulei in the initial condition
#define    initial_nuclei_border_i i_tip_target // all grains are at the left
#define    num_nuclei_target 4000
#define    initial_max_radius (6.0/dx) // radius of initial nuclei
#define    j_border   (Ny/2) // initial j value at the border of grains 1 and 2

// #define    random_seed 0
#define    noise_amplitude 0.3
#define    phi_interface 0. // phi value at the interface

// projections of crystal axes on the lab coordinates
// 1-9: x'=(x1,x2,x3), y'=(y1,y2,y3), z'=(z1,z2,z3)
#define    num_axes  9 // orientation(num_orientation, num_axes)
#define    alpha0   (0) // [0, 45], Euler angles
#define    beta0    (0) // [0, 45]
#define    gamma0   (0) // [0, 45]
#define    alpha1   (0) // [0, 45]
#define    beta1    (0) // [0, 45]
#define    gamma1   (0) // [0, 45]

// boundary condition
#define    imin  1
#define    imax  (Nx-2)
#define    iTmin  1
#define    iTmax  (NxT-2)

#define     Periodic  1
#if (Periodic)
    #define     jmin (Ny-2)
    #define     jmax (1)
    #define     jTmin  (NyT-2)
    #define     jTmax  1
#else // no-flux
    #define     jmin 1
    #define     jmax (Ny-2)
    #define     jTmin  1
    #define     jTmax  (NyT-2)
#endif

// temperature oscillation
#define    OSCILLATION 0 // ( osc_A*sin(-osc_omgea*step*dt*tau0)/undercooling )
#define    osc_A (2.0) // K, amplitude of oscillation
#define    osc_omgea (2.0*PI/400.0)// 3.6941e-4 ns^-1, frequency, t0 = Dh/V^2 = 17 μs, pi/w/tau0/dt = 3.14/3.6941e-4 = 4.16e5 steps
#define    osc_b (20.0) // nm, decaying distance, can be tip radius
// #define    Dh    (1.7e4) // nm^2/ns, diffusivity of T, from Pinomaa et al, 2020
    }
    namespace Material {

    }
}

#endif // SIMULATOR_MAIN_HPP_

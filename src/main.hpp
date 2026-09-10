#ifndef MAIN_H_
#define MAIN_H_

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

using real = double;

struct constants {
  long int step;
  long int initial_step;

  long int total_step;
  int step_fields;      // number of steps to output fields
  int step_tip;         // number of steps to get the tip information
  int step_check_point; // number of steps to save the check point
  int step_pull_back;   // number of steps to pull back

  int if_begin_tip;   // if 1: print the header to tip.txt
  int if_begin_field; // if 1: print the header to c_*.txt, psi_*.txt, etc
  int i_offset;       // itip - i_tip_target
  int i_offset_history_initial; // accumulated i_offset at the begining of the
                                // simulation
  int i_offset_history;         // accumulated i_offset
  int i_ignore; // for pull back: If i_ignore>0, ignore writing the history
                // data; else write the data

  real dt; // dimensionless time step of phase field and concentration
  real x_tip_history; // unit: W; tip position; used to calculate tip velocity;

  real delta_initial;       // initial dimensionless temperature
  real delta;               // dimensionless temperature
  real l_offset_Vp_initial; // unit: W; lattice offset only due to pulling
                            // velocity
  real l_offset_Vp; // unit: W; lattice offset only due to pulling velocity

  ////////////////////////////// for TFC
  int i_left_end_initial; // index of the left border of the PF in the whole
                          // temperature field
  int i_left_end;

  real dtT;         // dimensionless time step of the temperature field
  int ratio_dt_dtT; // dt/dtT
  int NxT;          // physical temperature field
  int NyT;
  int NxT_all; // NxT_all > NxT; some of lattices are not used
  int NyT_all;

  real Tcold;         // K
  real Thot;          // K
  real length_sample; // nm
};
__global__ void setup_kernel(unsigned long long seed, curandState *state);
__global__ void initial(real *c, real *phi, real *psi, real *orientation,
                        constants *para);
__global__ void compute_phi(real *c, real *phi, real *psi, real *T_fine,
                            real *c_next, real *phi_next, real *psi_next,
                            real *orientation, curandState *state,
                            constants *para, long int step,
                            int i_offset_history, real l_offset_Vp);
__global__ void compute_psi(real *c, real *phi, real *psi, real *c_next,
                            real *phi_next, real *psi_next, real *orientation,
                            curandState *state, constants *para, long int step);
__global__ void compute_c(real *c, real *phi, real *psi, real *c_next,
                          real *phi_next, real *psi_next, real *orientation,
                          curandState *state, constants *para, long int step);
__global__ void compute_c_CALPHAD(real *c, real *phi, real *psi, real *T_fine,
                                  real *c_next, real *phi_next, real *psi_next,
                                  real *orientation, curandState *state,
                                  constants *para, long int step,
                                  int i_offset_history, real l_offset_Vp);
__global__ void pull_back(real *c, real *phi, real *psi, real *c_next,
                          real *phi_next, real *psi_next, int i_offset,
                          long int step);
__global__ void initial_temperature(real *T, constants *para);
__global__ void integrate_phi(real *dphidt, real *psi, real *psi_next,
                              constants *para, int i_left_end, long int step);
__global__ void compute_temperature(real *T, real *T_next, real *dphidt,
                                    constants *para, long int step,
                                    int sub_step);
__global__ void interpolate_temperature(real *T, real *T_fine, constants *para,
                                        int i_left_end, long int step);

__host__ __device__ real Vp_ramp(real t);
__host__ __device__ real x_pull_back(real t);
__host__ __device__ real get_local_psi(real *phi, int i, int j);
__host__ __device__ real get_delta(constants *para, int i, int i_offset_history,
                                   real l_offset_Vp);
__host__ __device__ void set_orientation(real *orientation, real alpha,
                                         real beta, real gamma);
__host__ __device__ real Gl(real T, real c);
__host__ __device__ real Gs(real T, real c);
__host__ __device__ real Glc(real T, real c);
__host__ __device__ real Gsc(real T, real c);

void set_initial(real *h_c, real *h_phi, real *h_psi, real *h_T, real *h_T_fine,
                 constants *h_para, real *h_orientation, real *d_c, real *d_phi,
                 real *d_psi, real *d_T, real *d_T_fine, constants *d_para,
                 real *d_orientation, curandState *d_state, size_t size_grid,
                 size_t size_grid_T, dim3 numBlocks, dim3 threadsPerBlock,
                 dim3 numBlocksT, dim3 threadsPerBlockT);
void set_fields_output(real *h_c, real *h_phi, real *h_psi, real *h_T,
                       real *h_T_fine, constants *h_para, real *h_orientation,
                       real *d_c, real *d_phi, real *d_psi, real *d_T,
                       real *d_T_fine, constants *d_para, size_t size_grid,
                       size_t size_grid_T, real start_time, long int step,
                       dim3 numBlocks, dim3 threadsPerBlock);
void set_tip_output(real *h_c, real *h_phi, real *h_psi, real *h_T,
                    real *h_T_fine, constants *h_para, real *h_orientation,
                    real *d_c, real *d_phi, real *d_psi, real *d_T,
                    real *d_T_fine, constants *d_para, size_t size_grid,
                    size_t size_grid_T, real start_time, long int step,
                    dim3 numBlocks, dim3 threadsPerBlock);
void set_check_point(real *h_c, real *h_phi, real *h_psi, real *h_T, real *d_c,
                     real *d_phi, real *d_psi, real *d_T, constants *h_para,
                     size_t size_grid, size_t size_grid_T, long int step,
                     int final_step);
void set_pull_back(real *h_c, real *h_phi, real *h_psi, real *h_T,
                   constants *h_para, real *d_c, real *d_phi, real *d_psi,
                   real *d_T, constants *d_para, size_t size_grid,
                   long int step);

void set_nuclei(real *c, real *phi, real *psi, real *orientation,
                constants *para, int seed);
void set_nuclei_melt_pool(real *c, real *phi, real *psi, real *orientation,
                          constants *para, int seed);
void set_parameters(constants *para);

void generate_random_orientation(real *orientation, int seed);
void get_itip_each_grain(real *psi, int *tip);
int get_itip_all_grain(real *psi);
int get_local_grain(real *phi, int i, int j);
real get_average(real *data, int a);

void save_parameters(constants *para, std::size_t i_tip);
void save_field(real *data, int index, const char *field);
void save_T_full(real *T, constants *para, long int step);
void save_final(real *phi);

void save_orientation(real *orientation);
void save_check_point(real *data, const char *field);
void save_check_point_temperature(real *T, constants *para);
void save_check_point_parameters(constants *para, int i_tip);
void load_check_point(real *data, const char *field);
void load_orientation(real *orientation);
void load_check_point_temperature(real *T, constants *para);
void load_check_point_parameters(constants *para);
void load_parameters(constants *para);

void ensure_directory(const char *dirname);
int set_cuda_device();
void AutoBlockSize(int *Bloc, int devNx, int devNy);

#endif // MAIN_H_


#include "constants.hpp"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#define pos(x, y) ((y) * Nx_all + (x))
#define psN(x, y, a)                                                           \
  ((a) * Nx_all * Ny_all + (y) * Nx_all + (x)) // position in the phi matrix
#define psT(x, y) ((y) * NxT_all + (x))

#define pow2(x) ((x) * (x))
#define pow3(x) (pow2(x) * (x))
#define pow4(x) (pow2(pow2(x)))
#define pow5(x) (pow2(pow2(x)) * (x))
#define pow6(x) (pow2(pow2(x)) * pow2(x))
#define pow7(x) (pow2(pow2(x)) * pow2(x) * (x))
#define pow9(x) (pow3(pow3(x)))

#define qq(p) (0.5 * AA * (1.0 - (p)) - 0.25 * (AA - 1) * pow2(1.0 - (p)))
#define dqq_dp(p) (-0.5 * (1.0 + (p) * (AA - 1))) // first dirivative

#define gg(p)                                                                  \
  (1.875 * (p) - 1.25 * pow3(p) + 0.375 * pow5(p)) // 15.0/8.0 = 1.875
#define dgg_dp(p) (1.875 * pow2(1.0 - pow2(p)))    // first dirivative
#define dgg_dpdp(p) (7.5 * (p) * (pow2(p) - 1.0))  // second dirivative
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

void save_parameters(constants *para, int i_tip);
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

int main(int argc, char **argv) {
  if (set_cuda_device() !=
      0)      // set the index of device and display the device information
    return 1; // Exit program if no GPU is found

  long int step;
  real start_time = clock() / (1.0 * CLOCKS_PER_SEC);

  //////////////////////////////////////////////////////////////////////////////////////////////////////
  /// parameters
  constants *h_para, *d_para;
  h_para = (constants *)malloc(sizeof(constants));
  cudaMalloc((void **)&d_para, sizeof(constants));
  set_parameters(h_para); // set parameters before use the elements

  //////////////////////////////////////////////////////////////////////////////////////////////////////
  /// matrix definitions
  size_t size_grid = Nx_all * Ny_all * sizeof(real);
  real *h_c, *h_phi, *h_psi,
      *h_orientation; // psi: num_orientation - 1 + Σ phi
  real *d_c, *d_phi, *d_psi, *d_orientation, *d_c_next, *d_phi_next,
      *d_psi_next;
  real *d_temp;
  curandState *d_state;

  h_c = (real *)malloc(size_grid);
  h_phi = (real *)malloc(num_orientation * size_grid);
  h_psi = (real *)malloc(size_grid);
  h_orientation = (real *)malloc(num_axes * num_orientation * sizeof(real));

  cudaMalloc((void **)&d_c, size_grid);
  cudaMalloc((void **)&d_phi, num_orientation * size_grid);
  cudaMalloc((void **)&d_psi, size_grid);
  cudaMalloc((void **)&d_c_next, size_grid);
  cudaMalloc((void **)&d_phi_next, num_orientation * size_grid);
  cudaMalloc((void **)&d_psi_next, size_grid);
  cudaMalloc((void **)&d_orientation,
             num_axes * num_orientation * sizeof(real));
  cudaMalloc((void **)&d_state, Nx_all * Ny_all * sizeof(curandState));

  dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 numBlocks(Nx_all / threadsPerBlock.x, Ny_all / threadsPerBlock.y);

  //////////////////////////////////////////////////////////////////////////////////////////////////////
  /// TFC
  size_t size_grid_T = 0;
  real *h_T = NULL,
       *h_T_fine = NULL; // not used in FTA, but appear in function definitions
  real *d_T = NULL, *d_T_fine = NULL;
  dim3 threadsPerBlockT(1, 1);
  dim3 numBlocksT(1, 1);

#if (ifTFC)
  int sub_step;
  real *h_dphidt, *d_T_next, *d_dphidt;

  size_grid_T = h_para->NxT_all * h_para->NyT_all * sizeof(real);
  h_T = (real *)malloc(size_grid_T);
  h_T_fine = (real *)malloc(size_grid);
  h_dphidt = (real *)malloc(size_grid_T);

  cudaMalloc((void **)&d_T, size_grid_T);
  cudaMalloc((void **)&d_T_next, size_grid_T);
  cudaMalloc((void **)&d_dphidt, size_grid_T);
  cudaMalloc((void **)&d_T_fine, size_grid);

  threadsPerBlockT = dim3(BLOCK_SIZE_T_X, BLOCK_SIZE_T_Y);
  numBlocksT = dim3(h_para->NxT_all / threadsPerBlockT.x,
                    h_para->NyT_all / threadsPerBlockT.y);
#endif

  //////////////////////////////////////////////////////////////////////////////////////////////////////
  /// initial condition
  set_initial(h_c, h_phi, h_psi, h_T, h_T_fine, h_para, h_orientation, d_c,
              d_phi, d_psi, d_T, d_T_fine, d_para, d_orientation, d_state,
              size_grid, size_grid_T, numBlocks, threadsPerBlock, numBlocksT,
              threadsPerBlockT);

  // real delta = get_delta(h_para, i_tip_target, h_para->step,
  // h_para->i_offset_history, h_para->l_offset_Vp); printf("delta=%g,
  // i_tip_target=%d, step=%d, i_offset_history=%d, l_offset_Vp=%g\n",
  //     delta, i_tip_target, h_para->step, h_para->i_offset_history,
  //     h_para->l_offset_Vp);

  //////////////////////////////////////////////////////////////////////////////////////////////////////
  /// iterations
  for (step = h_para->initial_step; step <= h_para->total_step; step++) {

    if (step == h_para->initial_step) {
      std::cout << "total steps for this simulation is: " << h_para->total_step
                << std::endl;
    }
    set_fields_output(h_c, h_phi, h_psi, h_T, h_T_fine, h_para, h_orientation,
                      d_c, d_phi, d_psi, d_T, d_T_fine, d_para, size_grid,
                      size_grid_T, start_time, step, numBlocks,
                      threadsPerBlock);
    set_tip_output(h_c, h_phi, h_psi, h_T, h_T_fine, h_para, h_orientation, d_c,
                   d_phi, d_psi, d_T, d_T_fine, d_para, size_grid, size_grid_T,
                   start_time, step, numBlocks,
                   threadsPerBlock); // flag
    set_check_point(h_c, h_phi, h_psi, h_T, d_c, d_phi, d_psi, d_T, h_para,
                    size_grid, size_grid_T, step, 0);

    if (step % h_para->step_pull_back == 0) {
      set_pull_back(h_c, h_phi, h_psi, h_T, h_para, d_c, d_phi, d_psi, d_T,
                    d_para, size_grid, step);
      pull_back<<<numBlocks, threadsPerBlock>>>(d_c, d_phi, d_psi, d_c_next,
                                                d_phi_next, d_psi_next,
                                                h_para->i_offset, step);
      d_temp = d_c;
      d_c = d_c_next;
      d_c_next = d_temp;
      d_temp = d_phi;
      d_phi = d_phi_next;
      d_phi_next = d_temp;
      d_temp = d_psi;
      d_psi = d_psi_next;
      d_psi_next = d_temp;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    compute_phi<<<numBlocks, threadsPerBlock>>>(
        d_c, d_phi, d_psi, d_T_fine, d_c_next, d_phi_next, d_psi_next,
        d_orientation, d_state, d_para, step, h_para->i_offset_history,
        h_para->l_offset_Vp);
    compute_psi<<<numBlocks, threadsPerBlock>>>(
        d_c, d_phi, d_psi, d_c_next, d_phi_next, d_psi_next, d_orientation,
        d_state, d_para, step);
#if (modelCALPHAD)
    compute_c_CALPHAD<<<numBlocks, threadsPerBlock>>>(
        d_c, d_phi, d_psi, d_T_fine, d_c_next, d_phi_next, d_psi_next,
        d_orientation, d_state, d_para, step, h_para->i_offset_history,
        h_para->l_offset_Vp);
#else
    compute_c<<<numBlocks, threadsPerBlock>>>(
        d_c, d_phi, d_psi, d_c_next, d_phi_next, d_psi_next, d_orientation,
        d_state, d_para, step);
#endif

#if (ifTFC)
    integrate_phi<<<numBlocksT, threadsPerBlockT>>>(
        d_dphidt, d_psi, d_psi_next, d_para, h_para->i_left_end, step);
    for (sub_step = 0; sub_step < h_para->ratio_dt_dtT; sub_step++) {
      compute_temperature<<<numBlocksT, threadsPerBlockT>>>(
          d_T, d_T_next, d_dphidt, d_para, step, sub_step);
      d_temp = d_T;
      d_T = d_T_next;
      d_T_next = d_temp;
    }
    interpolate_temperature<<<numBlocks, threadsPerBlock>>>(
        d_T, d_T_fine, d_para, h_para->i_left_end, step);
#endif

    d_temp = d_c;
    d_c = d_c_next;
    d_c_next = d_temp;
    d_temp = d_phi;
    d_phi = d_phi_next;
    d_phi_next = d_temp;
    d_temp = d_psi;
    d_psi = d_psi_next;
    d_psi_next = d_temp;

    h_para->l_offset_Vp +=
        Vp_ramp(step * tau0 * h_para->dt) * h_para->dt * tau0 / (SS * W0);
    //////////////////////////////////////////////////////////////////////////////////////////////////
    if (clock() / (1.0 * CLOCKS_PER_SEC) - start_time >
        run_time * 3600.0 - 1000.0)
      break;
  }

  set_check_point(h_c, h_phi, h_psi, h_T, d_c, d_phi, d_psi, d_T, h_para,
                  size_grid, size_grid_T, step, 1);

  free(h_c);
  free(h_phi);
  free(h_psi);
  free(h_orientation);
  free(h_para);

  cudaFree(d_c);
  cudaFree(d_phi);
  cudaFree(d_psi);
  cudaFree(d_c_next);
  cudaFree(d_phi_next);
  cudaFree(d_psi_next);
  cudaFree(d_orientation);
  cudaFree(d_para);
  cudaFree(d_state);

#if (ifTFC)
  free(h_T);
  free(h_T_fine);
  free(h_dphidt);
  cudaFree(d_T);
  cudaFree(d_T_fine);
  cudaFree(d_T_next);
  cudaFree(d_dphidt);
#endif

  return EXIT_SUCCESS;
}

void set_initial(real *h_c, real *h_phi, real *h_psi, real *h_T, real *h_T_fine,
                 constants *h_para, real *h_orientation, real *d_c, real *d_phi,
                 real *d_psi, real *d_T, real *d_T_fine, constants *d_para,
                 real *d_orientation, curandState *d_state, size_t size_grid,
                 size_t size_grid_T, dim3 numBlocks, dim3 threadsPerBlock,
                 dim3 numBlocksT, dim3 threadsPerBlockT) {
  ensure_directory("data"); // create the folder if not exsit

  // int random_seed = time(NULL)+clock();
  setup_kernel<<<numBlocks, threadsPerBlock>>>(random_seed, d_state);

#if (if_load == 0)
  cudaMemcpy(d_para, h_para, sizeof(constants),
             cudaMemcpyHostToDevice); // copy Tcold to the device
  initial<<<numBlocks, threadsPerBlock>>>(d_c, d_phi, d_psi, d_orientation,
                                          d_para);
  cudaMemcpy(h_orientation, d_orientation,
             num_axes * num_orientation * sizeof(real), cudaMemcpyDeviceToHost);
  save_orientation(h_orientation);

#if (ifTFC)
  initial_temperature<<<numBlocksT, threadsPerBlockT>>>(d_T, d_para);
  interpolate_temperature<<<numBlocks, threadsPerBlock>>>(
      d_T, d_T_fine, d_para, h_para->i_left_end, h_para->initial_step);
#endif

  // // if initial grains are randomly distributed
  // set_nuclei_melt_pool( h_c, h_phi, h_psi, h_orientation, h_para,
  // random_seed ); save_orientation(h_orientation); cudaMemcpy(d_c, h_c,
  // size_grid, cudaMemcpyHostToDevice); cudaMemcpy(d_phi, h_phi,
  // num_orientation*size_grid, cudaMemcpyHostToDevice); cudaMemcpy(d_psi,
  // h_psi, size_grid, cudaMemcpyHostToDevice); cudaMemcpy(d_orientation,
  // h_orientation, num_axes*num_orientation*sizeof(real),
  // cudaMemcpyHostToDevice);
#else
  load_parameters(h_para);
  load_orientation(h_orientation);
  load_check_point(h_c, "c");
  load_check_point(h_phi, "phi");

#if (ifTFC)
  load_check_point_temperature(h_T, h_para);
  cudaMemcpy(d_T, h_T, size_grid_T, cudaMemcpyHostToDevice);
#endif
  printf("Load data finished!\n\n");

  for (int j = 0; j < Ny; j++) {
    for (int i = 0; i < Nx; i++) {
      h_psi[pos(i, j)] = get_local_psi(h_phi, i, j);
    }
  }

  // set_orientation(h_orientation, alpha0, beta0, gamma0);
  // if (num_orientation==2)
  //     set_orientation(h_orientation+num_axes, alpha1, beta1, gamma1);

  save_orientation(h_orientation);
  cudaMemcpy(d_c, h_c, size_grid, cudaMemcpyHostToDevice);
  cudaMemcpy(d_phi, h_phi, num_orientation * size_grid, cudaMemcpyHostToDevice);
  cudaMemcpy(d_psi, h_psi, size_grid, cudaMemcpyHostToDevice);
  cudaMemcpy(d_orientation, h_orientation,
             num_axes * num_orientation * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_para, h_para, sizeof(constants),
             cudaMemcpyHostToDevice); // copy Tcold to the device
#endif

  save_parameters(h_para, i_tip_target);
}

void set_fields_output(real *h_c, real *h_phi, real *h_psi, real *h_T,
                       real *h_T_fine, constants *h_para, real *h_orientation,
                       real *d_c, real *d_phi, real *d_psi, real *d_T,
                       real *d_T_fine, constants *d_para, size_t size_grid,
                       size_t size_grid_T, real start_time, long int step,
                       dim3 numBlocks, dim3 threadsPerBlock) {
  if (step % h_para->step_fields == 0 || step == h_para->initial_step) // flag
  {
    real time_now = clock() / (1.0 * CLOCKS_PER_SEC);
    cudaMemcpy(h_c, d_c, size_grid, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_phi, d_phi, num_orientation * size_grid,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_psi, d_psi, size_grid, cudaMemcpyDeviceToHost);

    if (h_para->if_begin_field == 1) {
      printf("(1)        (2)        (3)        (4)\n");
      printf("step       t[us]      t[s]       t[h]\n");
      h_para->if_begin_field = 0;
    }

    real end_time = clock() / (1.0 * CLOCKS_PER_SEC);
    printf("%-10d ", step);
    printf("%-10.3f ", step * h_para->dt * tau0 / 1000);
    printf("%-10.0f ", end_time - start_time);
    printf("%-10.1f ", (end_time - start_time) / 3600);
    printf("\n");
    fflush(stdout);

    int index = step / h_para->step_fields;
    save_field(h_c, index, "c");
    save_field(h_phi, index, "phi0");
    // save_field(h_phi+, index, "phi1");
    // save_field(h_psi, index, "psi");
    // save_field(h_phi, index, "grain"); // int values

    // #if (ifTFC)
    //     interpolate_temperature<<<numBlocks, threadsPerBlock>>>(d_T,
    //     d_T_fine, d_para, h_para->i_left_end, step); cudaMemcpy(h_T_fine,
    //     d_T_fine, size_grid, cudaMemcpyDeviceToHost); cudaMemcpy(h_T,
    //     d_T, size_grid_T, cudaMemcpyDeviceToHost);

    //     save_T_full(h_T, h_para, index);
    //     save_field(h_T_fine, index, "Tfine");
    // #endif
  }
}

void set_check_point(real *h_c, real *h_phi, real *h_psi, real *h_T, real *d_c,
                     real *d_phi, real *d_psi, real *d_T, constants *h_para,
                     size_t size_grid, size_t size_grid_T, long int step,
                     int final_step) {
  if (step % h_para->step_check_point == 0 || final_step == 1) {
    real time_now = clock() / (1.0 * CLOCKS_PER_SEC);
    cudaMemcpy(h_c, d_c, size_grid, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_phi, d_phi, num_orientation * size_grid,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_psi, d_psi, size_grid, cudaMemcpyDeviceToHost);
    save_check_point(h_c, "c");
    save_check_point(h_phi, "phi");

    // h_para->delta is updated in the set_tip_output function
    h_para->step = step;
    int i_tip = get_itip_all_grain(h_psi);
    save_parameters(h_para, i_tip);

#if (ifTFC)
    cudaMemcpy(h_T, d_T, size_grid_T, cudaMemcpyDeviceToHost);
    save_check_point_temperature(h_T, h_para);
#endif
  }
}

void set_pull_back(real *h_c, real *h_phi, real *h_psi, real *h_T,
                   constants *h_para, real *d_c, real *d_phi, real *d_psi,
                   real *d_T, constants *d_para, size_t size_grid,
                   long int step) {
  cudaMemcpy(h_psi, d_psi, size_grid, cudaMemcpyDeviceToHost);
  h_para->i_offset = get_itip_all_grain(h_psi) - i_tip_target;
  h_para->i_offset_history += h_para->i_offset;

#if (ifTFC)
  h_para->i_left_end = h_para->i_left_end_initial + h_para->i_offset_history -
                       int(h_para->l_offset_Vp / dx);
#endif

  cudaMemcpy(d_para, h_para, sizeof(constants), cudaMemcpyHostToDevice);

#if (if_output_field_history)
  ///////////////////////// record the offset history
  char offset_name[256];
  snprintf(offset_name, sizeof(offset_name), "%s/data/offset.txt", path_input);
  FILE *file_ioffset = fopen(offset_name, "a");
  fprintf(file_ioffset, "%ld\t%d\t%d\t%d\n", step, h_para->i_offset,
          h_para->i_offset_history, h_para->step_pull_back);
  fclose(file_ioffset);

  ///////////////////////// record the history data
  cudaMemcpy(h_c, d_c, size_grid, cudaMemcpyDeviceToHost);
  char file_name[256];
  snprintf(file_name, sizeof(file_name), "%s/data/c_history_temp.txt",
           path_input);
  FILE *file_history = fopen(file_name, "a");
  h_para->i_ignore -=
      h_para->i_offset; // if i_ignore>0, ignore writing; else write the data
  int i, j;
  for (i = 0; i < h_para->i_offset; i++) {
    if ((i + h_para->i_offset_history) % di == 0) {
      if (h_para->i_ignore <= 0) {
        for (j = 0; j < Ny; j = j + dj) {
          fprintf(file_history, "%g\t", h_c[pos(i, j)]);
        }
        fprintf(file_history, "\n");
      }
      h_para->i_ignore--;
      if (h_para->i_ignore < 0)
        h_para->i_ignore = 0; // for the case that i_offset<0
    }
  }
  fclose(file_history);
#endif
}

__global__ void setup_kernel(unsigned long long seed, curandState *state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  curand_init(seed, pos(i, j), 0, &state[pos(i, j)]);
}

__host__ __device__ void set_orientation(real *orientation, real alpha,
                                         real beta, real gamma) {
  alpha = alpha * PI / 180.0;
  beta = beta * PI / 180.0;
  gamma = gamma * PI / 180.0;

  orientation[0] = cos(beta) * cos(gamma);
  orientation[1] =
      sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma);
  orientation[2] =
      cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma);

  orientation[3] = cos(beta) * sin(gamma);
  orientation[4] =
      sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma);
  orientation[5] =
      cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma);

  orientation[6] = -sin(beta);
  orientation[7] = sin(alpha) * cos(beta);
  orientation[8] = cos(alpha) * cos(beta);
}

void generate_random_orientation(real *orientation,
                                 int seed) // for polycrystalline
{
  srand(seed);
  real theta, eta; // theta: azimuthal angle, eta: polar angle
  real x1, x2, x3, y1, y2, y3, dot, rb, z1, z2, z3;
  for (int a = 0; a < num_orientation; a++) {
    theta = 2.0 * PI * ((real)rand() / (real)RAND_MAX);
    eta = acos(1.0 - 2.0 * ((real)rand() / (real)RAND_MAX));
    x1 = sin(eta) * cos(theta);
    x2 = sin(eta) * sin(theta);
    x3 = cos(eta);

    theta = 2.0 * PI * ((real)rand() / (real)RAND_MAX);
    eta = acos(1.0 - 2.0 * ((real)rand() / (real)RAND_MAX));
    y1 = sin(eta) * cos(theta);
    y2 = sin(eta) * sin(theta);
    y3 = cos(eta);

    dot = x1 * y1 + x2 * y2 + x3 * y3;
    y1 = y1 - dot * x1;
    y2 = y2 - dot * x2;
    y3 = y3 - dot * x3;
    rb = sqrt(y1 * y1 + y2 * y2 + y3 * y3);

    y1 = y1 / rb;
    y2 = y2 / rb;
    y3 = y3 / rb;

    z1 = x2 * y3 - y2 * x3;
    z2 = y1 * x3 - x1 * y3;
    z3 = x1 * y2 - y1 * x2;

    orientation[a * num_axes + 0] = x1;
    orientation[a * num_axes + 1] = x2;
    orientation[a * num_axes + 2] = x3;
    orientation[a * num_axes + 3] = y1;
    orientation[a * num_axes + 4] = y2;
    orientation[a * num_axes + 5] = y3;
    orientation[a * num_axes + 6] = z1;
    orientation[a * num_axes + 7] = z2;
    orientation[a * num_axes + 8] = z3;
  }
}

// Use this initial condition if there are random nuclei
// low: nuclei[3*i+2] = 0, high: nuclei[3*i+2] = 1
// Bicrystal: low: [0, j_border], high: (j_border, Ny_all-1]
// j_border is the position of boundary between these two grains
void set_nuclei(real *c, real *phi, real *psi, real *orientation,
                constants *para, int seed) // flag
{
  int i, j, l, a, nuclei[num_nuclei_target * 3];
  // generate_random_orientation(orientation, seed);
  set_orientation(orientation, alpha0, beta0, gamma0);
  if (num_orientation == 2)
    set_orientation(orientation + num_axes, alpha1, beta1, gamma1);

  /////////////////////////////////// set the distribution of grains
  srand(seed);
  for (i = 0; i < num_nuclei_target; i++) {
    int random = (real)rand();
    int random2 = random % (Ny * initial_nuclei_border_i);
    nuclei[3 * i + 0] = random2 % initial_nuclei_border_i; // i
    nuclei[3 * i + 1] = random2 / initial_nuclei_border_i; // j
    random = rand();
    nuclei[3 * i + 2] = random % num_orientation;

    if (nuclei[3 * i + 2] == 0 && nuclei[3 * i + 1] >= j_border) {
      nuclei[3 * i + 1] =
          int((nuclei[3 * i + 1] - j_border) * j_border / (Ny - j_border));
    }
    if (nuclei[3 * i + 2] == 1 && nuclei[3 * i + 1] < j_border) {
      nuclei[3 * i + 1] =
          int(nuclei[3 * i + 1] * (Ny - j_border) / j_border) + j_border;
    }
  }

  // if (i,j) is closest to the center of nucleus l, then (i,j) is within
  // nucleus l
  ///////////////////////////////////
  real min_dis, dis, dis_jlow, dis_jup, dis_temp;
  int index_nucleus;
  for (j = 0; j < Ny; j++) {
    for (i = 0; i < Nx; i++) {
      /////////////////////// liquid
      c[pos(i, j)] = 1.0;
      for (a = 0; a < num_orientation; a++)
        phi[psN(i, j, a)] = -1.0;

      /////////////////////// find the nearest nucleus
      min_dis = 100000.0;
      for (l = 0; l < num_nuclei_target; l++) {
        dis = sqrt(pow2(i - nuclei[3 * l + 0]) + pow2(j - nuclei[3 * l + 1]));
        dis_jlow = sqrt(pow2(i - nuclei[3 * l + 0]) +
                        pow2(j - (nuclei[3 * l + 1] -
                                  (Ny - 2)))); // nuclei on the upper mirror
        dis_jup = sqrt(pow2(i - nuclei[3 * l + 0]) +
                       pow2(j - (nuclei[3 * l + 1] +
                                 (Ny - 2)))); // nuclei on the lower mirror

        dis_temp = min_dis;

        if (min_dis > dis)
          min_dis = dis;
        if (min_dis > dis_jlow)
          min_dis = dis_jlow;
        if (min_dis > dis_jup)
          min_dis = dis_jup;

        if (min_dis < dis_temp)
          index_nucleus = l;
      }

      /////////////////////// (i,j) is within nucleus l
      if (min_dis < initial_max_radius) {
        c[pos(i, j)] = ke * abs(para->delta_initial);
        phi[psN(i, j, nuclei[3 * index_nucleus + 2])] = 1.0;
      }

      psi[pos(i, j)] = get_local_psi(phi, i, j);
    }
  }
}

#define r_pool (Nx - 200) // radius of the circular melt pool
void set_nuclei_melt_pool(real *c, real *phi, real *psi, real *orientation,
                          constants *para, int seed) {
  int i, j, l, a, nuclei[num_nuclei_target * 2 * 3];
  generate_random_orientation(orientation, seed);

  /////////////////////////////////// set the distribution of grains
  int ind = 0;
  int num_nuclei;
  real random, possibility,
      num_lattice_outside_ellipse = Nx * Ny - PI * pow2(r_pool) / 4.0;

  for (j = 0; j < Ny; j++) {
    for (i = 0; i < Nx; i++) {
      if (pow2(i - Nx) + pow2(j - Ny) > pow2(r_pool)) {
        random = (real)rand() / (real)RAND_MAX;
        possibility = num_nuclei_target / (num_lattice_outside_ellipse);
        if (random < possibility) {
          nuclei[3 * ind + 0] = i;
          nuclei[3 * ind + 1] = j;
          nuclei[3 * ind + 2] =
              (int)floor(random / (possibility / num_orientation));
          ind++;
        }
      }
    }
  }
  num_nuclei = ind;
  printf("number of nuclei is %d\n", num_nuclei);

  // if (i,j) is closest to the center of nucleus l, then (i,j) is within
  // nucleus l
  ///////////////////////////////////
  real min_dis, dis, dis_jlow, dis_jup, dis_temp;
  int index_nuclei;
  for (j = 0; j < Ny; j++) {
    for (i = 0; i < Nx; i++) {
      /////////////////////// liquid
      c[pos(i, j)] = 1.0;
      for (a = 0; a < num_orientation; a++)
        phi[psN(i, j, a)] = -1.0;

      if (pow2(i - Nx) + pow2(j - Ny) > pow2(r_pool - initial_max_radius)) {
        /////////////////////// find the nearest nuclei
        min_dis = 100000.0;
        for (l = 0; l < num_nuclei; l++) {
          dis = sqrt(pow2(i - nuclei[3 * l + 0]) + pow2(j - nuclei[3 * l + 1]));
          dis_jlow = sqrt(pow2(i - nuclei[3 * l + 0]) +
                          pow2(j - (nuclei[3 * l + 1] -
                                    (Ny - 2)))); // nucleus on the upper mirror
          dis_jup = sqrt(pow2(i - nuclei[3 * l + 0]) +
                         pow2(j - (nuclei[3 * l + 1] +
                                   (Ny - 2)))); // nucleus on the lower mirror

          dis_temp = min_dis;

          if (min_dis > dis)
            min_dis = dis;
          if (min_dis > dis_jlow)
            min_dis = dis_jlow;
          if (min_dis > dis_jup)
            min_dis = dis_jup;

          if (min_dis < dis_temp)
            index_nuclei = l;
        }

        /////////////////////// (i,j) is within nuclei l
        if (min_dis < initial_max_radius) {
          c[pos(i, j)] = ke * abs(para->delta_initial);
          phi[psN(i, j, nuclei[3 * index_nuclei + 2])] = 1.0;
        }
      }
      psi[pos(i, j)] = get_local_psi(phi, i, j);
    }
  }
}

__global__ void initial(real *c, real *phi, real *psi, real *orientation,
                        constants *para) // flag
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int ps = pos(i, j);

  // if ( pow2(i*1.0/i_tip_target) + pow2((j-Ny/2*1.0)/(Ny/2-20))<1.0 )
  // {
  //     c[ps] = cinf*0.5;
  //     phi[psN(i,j,0)] = 1.0;
  // }
  // else
  // {
  //     c[ps] = cinf;
  //     phi[psN(i,j,0)] = -1.0;
  // }

  real itip = i_tip_target + 0.5 * sin(2.0 * PI * j / Ny);

  // real pp = -tanh( (i-itip)*dx/sqrt(2.0) ); // consider going back for this
  // part ... phi[ps] = pp; real cc = exp(bb*(1.0 + gg(pp))); // 0.3184*tanh(
  // ((i-itip)*dx-0.08)/0.67 ) + 0.6816;

  phi[ps] = -tanh((i - itip) * dx / sqrt(2.0));
  real cc = 0.3184 * tanh(((i - itip) * dx - 0.08) / 0.67) + 0.6816;

  // real c0 = 0.363227;
  // real cc = 0.5*(1.0-c0)*tanh( ((i-itip)*dx-0.08)/0.67 ) + 0.5*(1.0+c0);
  c[ps] = cc * cinf;

  psi[ps] = get_local_psi(phi, i, j);

  if (i == 1 && j == 1) {
    set_orientation(orientation, alpha0, beta0, gamma0);
    if (num_orientation == 2)
      set_orientation(orientation + num_axes, alpha1, beta1, gamma1);
  }
}

__global__ void initial_temperature(real *T, constants *para) {
  int iT = threadIdx.x + blockIdx.x * blockDim.x;
  int jT = threadIdx.y + blockIdx.y * blockDim.y;
  int NxT_all = para->NxT_all;
  T[psT(iT, jT)] = para->Tcold + (iT - 1) * SS * W0 * dx * ratio_dxT_dx * GG;
}

__device__ real get_anisotropy(real *phi, real *orientation, real *as, real *ak,
                               int i, int j, int a) {
  int psa = psN(i, j, a);
  real phi00 = phi[psa];
  real phi10 = phi[psN(i + 1, j, a)];
  real phi01 = phi[psN(i, j + 1, a)];
  real phim0 = phi[psN(i - 1, j, a)];
  real phi0m = phi[psN(i, j - 1, a)];
  real phi11 = phi[psN(i + 1, j + 1, a)];
  real phim1 = phi[psN(i - 1, j + 1, a)];
  real phi1m = phi[psN(i + 1, j - 1, a)];
  real phimm = phi[psN(i - 1, j - 1, a)];

  real dphi_dx = (phi10 - phim0) / (2.0 * dx);
  real dphi_dy = (phi01 - phi0m) / (2.0 * dx);
  real abs_gradient_phi = sqrt(pow2(dphi_dx) + pow2(dphi_dy));

  real dphi_dx_dx = (phi10 - 2.0 * phi00 + phim0) / (dx * dx);
  real dphi_dy_dy = (phi01 - 2.0 * phi00 + phi0m) / (dx * dx);
  real dphi_dx_dy = (phi11 - phim1 - phi1m + phimm) / (4. * dx * dx);

  real laplacian_phi = 2.0 / 3.0 * (dphi_dx_dx + dphi_dy_dy);
  laplacian_phi += 1.0 / 3.0 * (phi11 + phim1 + phi1m + phimm - 4.0 * phi00) /
                   (2.0 * dx * dx);

  real dtheta_dx, dtheta_dy, cos_theta, sin_theta;
  if (abs_gradient_phi > 1.0e-15) {
    dtheta_dx =
        (dphi_dx * dphi_dx_dy - dphi_dy * dphi_dx_dx) / pow2(abs_gradient_phi);
    dtheta_dy =
        (dphi_dx * dphi_dy_dy - dphi_dy * dphi_dx_dy) / pow2(abs_gradient_phi);
    cos_theta = dphi_dx / abs_gradient_phi;
    sin_theta = dphi_dy / abs_gradient_phi;
  } else {
    dtheta_dx = 0.0;
    dtheta_dy = 0.0;
    cos_theta = 0.0;
    sin_theta = 0.0;
  }

  real x1 = orientation[a * num_axes + 0];
  real x2 = orientation[a * num_axes + 1];
  real y1 = orientation[a * num_axes + 3];
  real y2 = orientation[a * num_axes + 4];
  real z1 = orientation[a * num_axes + 6];
  real z2 = orientation[a * num_axes + 7];

  real nxp = x1 * cos_theta + x2 * sin_theta; // n'_x
  real nyp = y1 * cos_theta + y2 * sin_theta; // n'_y
  real nzp = z1 * cos_theta + z2 * sin_theta; // n'_z
  real dnxp_dtheta_1 = -x1 * sin_theta + x2 * cos_theta;
  real dnyp_dtheta_1 = -y1 * sin_theta + y2 * cos_theta;
  real dnzp_dtheta_1 = -z1 * sin_theta + z2 * cos_theta;
  real dnxp_dtheta_2 = -nxp;
  real dnyp_dtheta_2 = -nyp;
  real dnzp_dtheta_2 = -nzp;

  real np4 = pow4(nxp) + pow4(nyp) + pow4(nzp);
  real dnp4_dtheta_1 =
      4.0 * (pow3(nxp) * dnxp_dtheta_1 + pow3(nyp) * dnyp_dtheta_1 +
             pow3(nzp) * dnzp_dtheta_1);
  real dnp4_dtheta_2 =
      3.0 * pow2(nxp * dnxp_dtheta_1) + pow3(nxp) * dnxp_dtheta_2;
  dnp4_dtheta_2 += 3.0 * pow2(nyp * dnyp_dtheta_1) + pow3(nyp) * dnyp_dtheta_2;
  dnp4_dtheta_2 += 3.0 * pow2(nzp * dnzp_dtheta_1) + pow3(nzp) * dnzp_dtheta_2;
  dnp4_dtheta_2 *= 4;

  if constexpr (if_1_eps) {
    *ak = 1.0 - 3.0 * epk1 + 4.0 * epk1 * np4;
    *as = 1.0 - 3.0 * eps1 + 4.0 * eps1 * np4;
    real da_dtheta_1 = 4.0 * eps1 * dnp4_dtheta_1;
    real da_dtheta_2 = 4.0 * eps1 * dnp4_dtheta_2;
  } else
    *ak = 1.0 + epk1 * (np4 - 0.6);
  *as = 1.0 + eps1 * (np4 - 0.6) +
        eps2 * (3.0 * np4 + 66.0 * pow2(nxp * nyp * nzp) - 17.0 / 7.0);

  // np2 = pow2(nxp*nyp*nzp)
  real dnp2_dtheta_1 = 2.0 * nxp * nyp * nzp *
                       (dnxp_dtheta_1 * nyp * nzp + nxp * dnyp_dtheta_1 * nzp +
                        nxp * nyp * dnzp_dtheta_1);
  real dnp2_dtheta_2 = pow2(dnxp_dtheta_1 * nyp * nzp) +
                       pow2(nxp * dnyp_dtheta_1 * nzp) +
                       pow2(nxp * nyp * dnzp_dtheta_1);
  dnp2_dtheta_2 += nxp * dnxp_dtheta_2 * pow2(nyp * nzp) +
                   nyp * dnyp_dtheta_2 * pow2(nxp * nzp) +
                   nzp * dnzp_dtheta_2 * pow2(nxp * nyp);
  dnp2_dtheta_2 += 4.0 * nxp * nyp * nzp *
                   (dnxp_dtheta_1 * dnyp_dtheta_1 * nzp +
                    nxp * dnyp_dtheta_1 * dnzp_dtheta_1 +
                    dnxp_dtheta_1 * nyp * dnzp_dtheta_1);
  dnp2_dtheta_2 *= 2.0;

  real da_dtheta_1 =
      (eps1 + 3.0 * eps2) * dnp4_dtheta_1 + 66.0 * eps2 * dnp2_dtheta_1;
  real da_dtheta_2 =
      (eps1 + 3.0 * eps2) * dnp4_dtheta_2 + 66.0 * eps2 * dnp2_dtheta_2;
#endif

  real anisotropy =
      pow2(*as) * laplacian_phi +
      2.0 * (*as) * da_dtheta_1 * (dtheta_dx * dphi_dx + dtheta_dy * dphi_dy) +
      (da_dtheta_2 * (*as) + pow2(da_dtheta_1)) *
          (dtheta_dy * dphi_dx - dtheta_dx * dphi_dy);

  return anisotropy;
}

// V100 has 256^2 registers per block; a real needs 2 registers; a int needs 1
// register
__global__ void compute_phi(real *c, real *phi, real *psi, real *T_fine,
                            real *c_next, real *phi_next,
                            real *psi_next, // flag
                            real *orientation, curandState *state,
                            constants *para, long int step,
                            int i_offset_history, real l_offset_Vp) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int a, ps = pos(i, j);
  real c00 = c[ps];

  if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1) {
    for (a = 0; a < num_orientation; a++) {
      int psa = psN(i, j, a);
      real phi00 = phi[psa];

      real sum_phi = 0;
      for (int b = 0; b < num_orientation; b++)
        if (b != a)
          sum_phi += pow2(0.5 * (1.0 + phi[psN(i, j, b)]));

      real as, ak;
      real anisotropy = get_anisotropy(phi, orientation, &as, &ak, i, j, a);
      real term = phi00 - pow3(phi00) + anisotropy;

#if (ifTFC)
      real T00 = T_fine[ps];
      // real delta = (T00-Tmelt)/deltaT0;
#else
      real delta = get_delta(para, i, i_offset_history, l_offset_Vp);
      real omega = -6.0 * lambda *
                   (2 * ke + (ke - 1.0) / bb *
                                 delta); // coupling between neighboring grains
      real T00 = delta * deltaT0 + Tmelt;
#endif

#if (modelA)
      term += -lambda * dgg_dp(phi00) *
              (c00 + delta * exp(bb * (1.0 + gg(phi00)))); //
      term += -omega * 0.5 * (1.0 + phi00) *
              sum_phi; // coupling between neighboring grains
#elif (modelB)
      term +=
          -lambda * dgg_dp(phi00) * (c00 + delta * (ke - 1.0) / (2.0 * bb)); //
      term += -omega * 0.5 * (1.0 + phi00) *
              sum_phi; // coupling between neighboring grains
#elif (modelCALPHAD)
      term +=
          (Gl(T00, c00) - Gs(T00, c00)) * dgg_dp(phi00) / (2.0 * hh * v0); //
#endif

      real tau = 1.0 + 1.0e4 * pow2(0.5 * (1.0 + phi00)) * sum_phi;
      phi_next[psa] = phi00 + para->dt * ak / (pow2(as) * tau) * term;
    }

///////////////////////////////////////////////////////////////////////////////////
/// add noise
#if (if_noise)
    curandState localState;
    localState = state[ps];

    for (a = 0; a < num_orientation; a++) {
      real random = curand_uniform_real(&localState);
      real reference = sqrt(2.0) * atanh(phi[psN(i, j, a)]);
      real fluctuation = noise_amplitude * sqrt(para->dt) * (random - 0.5);
      phi_next[psN(i, j, a)] += tanh((reference + fluctuation) / sqrt(2.0)) -
                                tanh(reference / sqrt(2.0));
    }
    state[ps] = localState;
#endif
  } // i,j
}

__device__ __forceinline__ void compute_boundary_individual(real *data, int i,
                                                            int j, int a) {
  if (i == imin)
    data[psN(0, j, a)] = data[psN(imin, j, a)];
  else if (i == imax)
    data[psN(Nx - 1, j, a)] = data[psN(imax, j, a)];
  if (j == jmin)
    data[psN(i, 0, a)] = data[psN(i, jmin, a)];
  else if (j == jmax)
    data[psN(i, Ny - 1, a)] = data[psN(i, jmax, a)];
}

__global__ void compute_psi(real *c, real *phi, real *psi, real *c_next,
                            real *phi_next, real *psi_next, real *orientation,
                            curandState *state, constants *para,
                            long int step) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int a, ps = pos(i, j);

  if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1) {
    ///////////////////////////////////////////////////////////////////////////////////
    /// psi, should be <=1
    psi_next[ps] = get_local_psi(phi_next, i, j);

    real num_phi_higher_than_minus1 = 0;
    for (a = 0; a < num_orientation; a++) {
      if (phi_next[psN(i, j, a)] > -1)
        num_phi_higher_than_minus1++;
    }

    real psi_temp = num_orientation - 1;
    if (psi_next[ps] > 1.0) {
      psi_temp = num_orientation - 1;
      for (a = 0; a < num_orientation; a++) {
        phi_next[psN(i, j, a)] -= (psi_next[ps] - 1.0) *
                                  (phi_next[psN(i, j, a)] > -1) /
                                  num_phi_higher_than_minus1;
        psi_temp += phi_next[psN(i, j, a)];
      }
      psi_next[ps] = psi_temp;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    for (a = 0; a < num_orientation; a++) {
      phi_next[psN(i, j, a)] = fmax(phi_next[psN(i, j, a)], -1.0);
      phi_next[psN(i, j, a)] = fmin(phi_next[psN(i, j, a)], 1.0);
    }
  }

  compute_boundary_individual(psi_next, i, j, 0);
  for (a = 0; a < num_orientation; a++)
    compute_boundary_individual(phi_next, i, j, a);
}

__global__ void compute_c(real *c, real *phi, real *psi, real *c_next,
                          real *phi_next, real *psi_next, real *orientation,
                          curandState *state, constants *para,
                          long int step) // flag
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int a;

  if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1) {
    int ps00 = pos(i, j);
    int ps10 = pos(i + 1, j);
    int psm0 = pos(i - 1, j);
    int ps01 = pos(i, j + 1);
    int ps0m = pos(i, j - 1);
    int ps11 = pos(i + 1, j + 1);
    int psm1 = pos(i - 1, j + 1);
    int ps1m = pos(i + 1, j - 1);
    int psmm = pos(i - 1, j - 1);

    real c00 = c[ps00];
    // real c10 = c[ps10];
    // real cm0 = c[psm0];
    // real c01 = c[ps01];
    // real c0m = c[ps0m];
    // real c11 = c[ps11];
    // real cm1 = c[psm1];
    // real c1m = c[ps1m];
    // real cmm = c[psmm];

    real g00 = 0, g10 = 0, gm0 = 0, g01 = 0, g0m = 0;
    // real g11=0, gm1=0, g1m=0, gmm=0;
    for (a = 0; a < num_orientation; a++) {
      g00 += gg(phi[psN(i, j, a)]);
      g10 += gg(phi[psN(i + 1, j, a)]);
      gm0 += gg(phi[psN(i - 1, j, a)]);
      g01 += gg(phi[psN(i, j + 1, a)]);
      g0m += gg(phi[psN(i, j - 1, a)]);
      // g11 += gg(phi[psN(i+1,j+1,a)]);
      // gm1 += gg(phi[psN(i-1,j+1,a)]);
      // g1m += gg(phi[psN(i+1,j-1,a)]);
      // gmm += gg(phi[psN(i-1,j-1,a)]);
    }

    real S00 = qq(psi[ps00]) * c00;
    real S10 = qq(psi[ps10]) * c[ps10];
    real S01 = qq(psi[ps01]) * c[ps01];
    real Sm0 = qq(psi[psm0]) * c[psm0];
    real S0m = qq(psi[ps0m]) * c[ps0m];
    real S11 = qq(psi[ps11]) * c[ps11];
    real Sm1 = qq(psi[psm1]) * c[psm1];
    real S1m = qq(psi[ps1m]) * c[ps1m];
    real Smm = qq(psi[psmm]) * c[psmm];

    real Shh = 0.25 * (S11 + S10 + S01 + S00);
    real Shn = 0.25 * (S10 + S00 + S1m + S0m);
    real Snh = 0.25 * (Sm1 + S01 + Sm0 + S00);
    real Snn = 0.25 * (Sm0 + Smm + S0m + S00);

    //////////////////// <10> contribution ///////// -1/2: n, -1: m
    real F10 = 0.25 * (S10 + S00 + Shh + Shn) *
               (log(c[ps10] / c00) - bb * (g10 - g00));
    real F01 = 0.25 * (S01 + S00 + Shh + Snh) *
               (log(c[ps01] / c00) - bb * (g01 - g00));
    real Fm0 = 0.25 * (S00 + Sm0 + Snh + Snn) *
               (log(c[psm0] / c00) - bb * (gm0 - g00));
    real F0m = 0.25 * (S00 + S0m + Shn + Snn) *
               (log(c[ps0m] / c00) - bb * (g0m - g00));

    real src10 =
        (F10 + F01 + Fm0 + F0m) / (dx * dx); // the divergence evaluated by <10>

    //////////////////// <11> contribution
    // real F11 = 0.25*(S11+S00+S01+S10)*( log(c[ps11]/c00)-bb*(g11-g00) );
    // real Fm1 = 0.25*(Sm1+S00+S01+Sm0)*( log(c[psm1]/c00)-bb*(gm1-g00) );
    // real F1m = 0.25*(S00+S1m+S10+S0m)*( log(c[ps1m]/c00)-bb*(g1m-g00) );
    // real Fmm = 0.25*(S00+Smm+Sm0+S0m)*( log(c[psmm]/c00)-bb*(gmm-g00) );

    // real src11 = (F11+Fm1+F1m+Fmm)/(2.0*dx*dx);  // the divergence
    // evaluated by <11>

    ////////////////////
    // real dc = para->dt*Dl/(Gamma*muk0)*(2.0*src10 + src11)/3.0;
    real dc = para->dt * Dl / (Gamma * muk0) * src10;

    c_next[ps00] = fmax(c00 + dc, 1.0e-6);
  }
  compute_boundary_individual(c_next, i, j, 0);
}

__global__ void compute_c_CALPHAD(real *c, real *phi, real *psi, real *T_fine,
                                  real *c_next, real *phi_next,
                                  real *psi_next, // flag
                                  real *orientation, curandState *state,
                                  constants *para, long int step,
                                  int i_offset_history, real l_offset_Vp) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1) {
    int ps00 = pos(i, j);
    int ps10 = pos(i + 1, j);
    int psm0 = pos(i - 1, j);
    int ps01 = pos(i, j + 1);
    int ps0m = pos(i, j - 1);
    int ps11 = pos(i + 1, j + 1);
    int psm1 = pos(i - 1, j + 1);
    int ps1m = pos(i + 1, j - 1);
    int psmm = pos(i - 1, j - 1);

    real c00 = c[ps00];
    real c10 = c[ps10];
    real cm0 = c[psm0];
    real c01 = c[ps01];
    real c0m = c[ps0m];
    real c11 = c[ps11];
    real cm1 = c[psm1];
    real c1m = c[ps1m];
    real cmm = c[psmm];

    real psi00 = psi[ps00];
    real psi10 = psi[ps10];
    real psim0 = psi[psm0];
    real psi01 = psi[ps01];
    real psi0m = psi[ps0m];
    real psi11 = psi[ps11];
    real psim1 = psi[psm1];
    real psi1m = psi[ps1m];
    real psimm = psi[psmm];

#if (ifTFC)
    real T00 = T_fine[ps00];
    real T10 = T_fine[ps10];
    real Tm0 = T_fine[psm0];
    real T01 = T_fine[ps01];
    real T0m = T_fine[ps0m];
#else
    real delta00 = get_delta(para, i, i_offset_history, l_offset_Vp);
    real delta10 = get_delta(para, i + 1, i_offset_history, l_offset_Vp);
    real deltam0 = get_delta(para, i - 1, i_offset_history, l_offset_Vp);
    real T00 = delta00 * deltaT0 + Tmelt;
    real T10 = delta10 * deltaT0 + Tmelt;
    real Tm0 = deltam0 * deltaT0 + Tmelt;
    real T01 = T00;
    real T0m = T00;
#endif

    real g00 = gg(psi00);
    real g10 = gg(psi10);
    real g01 = gg(psi01);
    real gm0 = gg(psim0);
    real g0m = gg(psi0m);

    real S00 = qq(psi00) * c00 * (1.0 - c00);
    real S10 = qq(psi10) * c10 * (1.0 - c10);
    real S01 = qq(psi01) * c01 * (1.0 - c01);
    real Sm0 = qq(psim0) * cm0 * (1.0 - cm0);
    real S0m = qq(psi0m) * c0m * (1.0 - c0m);
    real S11 = qq(psi11) * c11 * (1.0 - c11);
    real Sm1 = qq(psim1) * cm1 * (1.0 - cm1);
    real S1m = qq(psi1m) * c1m * (1.0 - c1m);
    real Smm = qq(psimm) * cmm * (1.0 - cmm);

    real Shh = 0.25 * (S11 + S10 + S01 + S00);
    real Shn = 0.25 * (S10 + S00 + S1m + S0m);
    real Snh = 0.25 * (Sm1 + S01 + Sm0 + S00);
    real Snn = 0.25 * (Sm0 + Smm + S0m + S00);

    real f00 =
        0.5 * (1.0 + g00) * Gsc(T00, c00) + 0.5 * (1.0 - g00) * Glc(T00, c00);
    real f10 =
        0.5 * (1.0 + g10) * Gsc(T10, c10) + 0.5 * (1.0 - g10) * Glc(T10, c10);
    real f01 =
        0.5 * (1.0 + g01) * Gsc(T01, c01) + 0.5 * (1.0 - g01) * Glc(T01, c01);
    real fm0 =
        0.5 * (1.0 + gm0) * Gsc(Tm0, cm0) + 0.5 * (1.0 - gm0) * Glc(Tm0, cm0);
    real f0m =
        0.5 * (1.0 + g0m) * Gsc(T0m, c0m) + 0.5 * (1.0 - g0m) * Glc(T0m, c0m);

    //////////////////// <10> contribution ///////// -1/2: n, -1: m
    real denom = 1.0 / (h0 * v0);
    real F10 = 0.25 * (S10 + S00 + Shh + Shn) * (f10 - f00) * denom;
    real F01 = 0.25 * (S01 + S00 + Shh + Snh) * (f01 - f00) * denom;
    real Fm0 = 0.25 * (S00 + Sm0 + Snh + Snn) * (fm0 - f00) * denom;
    real F0m = 0.25 * (S00 + S0m + Shn + Snn) * (f0m - f00) * denom;

    real dc =
        para->dt * Dl / (Gamma * muk0) * (F10 + F01 + Fm0 + F0m) / (dx * dx);

    c_next[ps00] = fmax(c00 + dc, 1.0e-6);
  }
  compute_boundary_individual(c_next, i, j, 0);
}

__global__ void pull_back(real *c, real *phi, real *psi, real *c_next,
                          real *phi_next, real *psi_next, int i_offset,
                          long int step) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  int i_next = i + i_offset;
  i_next = max(i_next, 1);
  i_next = min(i_next, Nx - 2);

  if (i_next < Nx - 2)
    c_next[pos(i, j)] = c[pos(i_next, j)];
  else {
#if (modelCALPHAD)
    c_next[pos(i, j)] = cinf; // concentration
#else
    c_next[pos(i, j)] = 1.0; // scaled concentration
#endif
  }

  for (int a = 0; a < num_orientation; a++)
    phi_next[psN(i, j, a)] = phi[psN(i_next, j, a)];
  psi_next[pos(i, j)] = psi[pos(i_next, j)];
}

__global__ void integrate_phi(real *dphidt, real *psi, real *psi_next,
                              constants *para, int i_left_end,
                              long int step) // flag
{
  int iT =
      threadIdx.x + blockIdx.x * blockDim.x; // index of the temperature field
  int jT = threadIdx.y + blockIdx.y * blockDim.y;

  int NxT_all = para->NxT_all;
  int NyT = para->NyT;

  int iT1 = i_left_end / ratio_dxT_dx; // the first iT index for the left
                                       // border of the PF simulation domain
  int iTn = (i_left_end + Nx - 3) /
            ratio_dxT_dx; // the last iT index for the right border of the PF
                          // simulation domain

  real integral = 0;

  if (iT >= iT1 && iT <= iTn && jT >= 1 && jT <= NyT - 2) {
    int i0 =
        ratio_dxT_dx * iT + 1 - i_left_end; // in the T domain (iT1,jT), the
                                            // first i index of the PF domain
    int in = i0 + ratio_dxT_dx -
             1; // in the T domain (iT1,jT), the last i index of the PF domain
    if (i0 < 1)
      i0 = 1;
    if (in > Nx - 2)
      in = Nx - 2;

    int num_lattice = 0;
    for (int j = 1 + ratio_dxT_dx * (jT - 1); j <= ratio_dxT_dx * jT; j++) {
      for (int i = i0; i <= in; i++) {
        integral += psi_next[pos(i, j)] - psi[pos(i, j)];
        num_lattice++;
      }
    }
    dphidt[psT(iT, jT)] = integral * 0.5;
  } else
    dphidt[psT(iT, jT)] = 0;
}

__global__ void compute_temperature(real *T, real *T_next, real *dphidt,
                                    constants *para, long int step,
                                    int sub_step) {
  int iT =
      threadIdx.x + blockIdx.x * blockDim.x; // index of the temperature field
  int jT = threadIdx.y + blockIdx.y * blockDim.y;
  int NxT = para->NxT;
  int NyT = para->NyT;
  int NxT_all = para->NxT_all;

  int ps00 = psT(iT, jT);
  int ps10 = psT(iT + 1, jT);
  int psm0 = psT(iT - 1, jT);
  int ps01 = psT(iT, jT + 1);
  int ps0m = psT(iT, jT - 1);

  real T00 = T[ps00];
  real T10 = T[ps10];
  real Tm0 = T[psm0];
  real T01 = T[ps01];
  real T0m = T[ps0m];

  if (iT >= 1 && iT <= NxT - 2 && jT >= 1 && jT <= NyT - 2) {
    real laplacian = (T10 + Tm0 + T01 + T0m - 4.0 * T00) / pow2(dxT);
    real dTdt = Vp_ramp(step * tau0 * para->dt) * tau0 / (SS * W0) *
                    (T10 - Tm0) / (2.0 * dxT) +
                laplacian * DT / (Gamma * muk0);
    T_next[ps00] =
        T[ps00] + para->dtT * dTdt +
        Lcp * dphidt[ps00] / (para->ratio_dt_dtT * pow2(ratio_dxT_dx));
  }

  /////////////////// boundary condition
  if (iT == iTmin)
    T_next[psT(0, jT)] = para->Tcold;
  else if (iT == iTmax)
    T_next[psT(NxT - 1, jT)] = para->Thot;
  if (jT == jTmin)
    T_next[psT(iT, 0)] = T_next[psT(iT, jTmin)];
  else if (jT == jTmax)
    T_next[psT(iT, NyT - 1)] = T_next[psT(iT, jTmax)];
}

__global__ void interpolate_temperature(real *T, real *T_fine, constants *para,
                                        int i_left_end, long int step) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= 0 && i < Nx && j >= 0 && j < Ny) {
    int NxT_all = para->NxT_all;

    int iT = (i - 1 + i_left_end) /
             ratio_dxT_dx; // coarse grid; i=1 corresponds to i_left_end
    int jT =
        (j - 1 + dxT) / ratio_dxT_dx; // coarse grid; j=1 corresponds to dxT

    real x = (i - 1.0 + i_left_end) / ratio_dxT_dx - iT;
    real y = (j - 1.0 + dxT) / ratio_dxT_dx - jT;

    real f00 = T[psT(iT, jT)];
    real f10 = T[psT(iT + 1, jT)];
    real f01 = T[psT(iT, jT + 1)];
    real f11 = T[psT(iT + 1, jT + 1)];

    real a00 = f00;
    real a10 = f10 - f00;
    real a01 = f01 - f00;
    real a11 = f11 - f10 - f01 + f00;

    T_fine[pos(i, j)] = a00 + a10 * x + a01 * y + a11 * x * y;
  }
}

__host__ __device__ real get_delta(constants *para, int i, int i_offset_history,
                                   real l_offset_Vp) // only for FTA
{
  return para->delta_initial +
         GG *
             (i - i_tip_target + i_offset_history -
              para->i_offset_history_initial) *
             dx * SS * W0 / deltaT0 -
         GG * (l_offset_Vp - para->l_offset_Vp_initial) * SS * W0 / deltaT0;
}

__host__ __device__ real get_local_psi(real *phi, int i, int j) {
  real psi = num_orientation - 1;
  for (int a = 0; a < num_orientation; a++)
    psi += phi[psN(i, j, a)];
  return psi;
}

__host__ __device__ real G_Ti_BCC(real T) {
  if (T >= 1155.0 && T < 1941.0)
    return 6667.385 + 105.366379 * T - 22.3771 * T * log(T) +
           1.21707e-3 * pow2(T) - 0.84534e-6 * pow3(T) - 2002750.0 / T;
  else if (T >= 1941.0)
    return 26483.26 - 182.426471 * T + 19.0900905 * T * log(T) -
           22.00832e-3 * pow2(T) + 1.228863e-6 * pow3(T) + 1400501.0 / T;
  return NAN;
}

__host__ __device__ real G_Ti_Liq(real T) {
  if (T >= 1300.0 && T < 1941.0)
    return 369519.198 - 2554.0225 * T + 342.059267 * T * log(T) -
           163.409355e-3 * pow2(T) + 12.457117e-6 * pow3(T) - 67034516.0 / T;
  else if (T >= 1941.0 && T < 4000.0)
    return -19887.066 + 298.7367 * T - 46.29 * T * log(T);
  return NAN;
}

__host__ __device__ real G_Nb_BCC(real T) {
  if (T >= 298.15 && T < 2750.0)
    return -8519.353 + 142.045475 * T - 26.4711 * T * log(T) +
           0.203475e-3 * pow2(T) - 0.35012e-6 * pow3(T) + 93399.0 / T;
  else if (T >= 2750.0)
    return -37669.3 + 271.720843 * T - 41.77 * T * log(T) +
           1528.238e29 / pow9(T);
  return NAN;
}

__host__ __device__ real G_Nb_Liq(real T) {
  if (T >= 298.15 && T < 2750.0)
    return 21262.202 + 131.229057 * T - 26.4711 * T * log(T) +
           0.203475e-3 * pow2(T) - 0.35012e-6 * pow3(T) + 93399.0 / T -
           306.098e-25 * pow7(T);
  else if (T >= 2750.0)
    return -7499.398 + 260.756148 * T - 41.77 * T * log(T);
  return NAN;
}

#define L0_BCC(T) (12315.25)
#define L0_Liq(T) (5144.29)

__host__ __device__ real Gl(real T, real c) // unit: J/mol, [T]=K
{
  real onemc = 1.0 - c;
  real log_part;
  if (c < 1.e-8)
    log_part = RR * T * (onemc * log(onemc));
  else if (onemc < 1.e-8)
    log_part = RR * T * (c * log(c));
  else
    log_part = RR * T * (onemc * log(onemc) + c * log(c));

  real mix_part = c * onemc * L0_Liq(T);
  return log_part + onemc * G_Nb_Liq(T) + c * G_Ti_Liq(T) + mix_part;
}

__host__ __device__ real Gs(real T, real c) {
  real onemc = 1.0 - c;
  real log_part;
  if (c < 1.e-8)
    log_part = RR * T * (onemc * log(onemc));
  else if (onemc < 1.e-8)
    log_part = RR * T * (c * log(c));
  else
    log_part = RR * T * (onemc * log(onemc) + c * log(c));

  real mix_part = c * onemc * L0_BCC(T);
  return log_part + onemc * G_Nb_BCC(T) + c * G_Ti_BCC(T) + mix_part;
}

__host__ __device__ real Glc(real T, real c) {
  real termL0 = L0_Liq(T) * (1.0 - 2.0 * c);

  real onemc = 1.0 - c;
  real log_part = 0.;
  if (c > 1.e-8 && onemc > 1.e-8)
    log_part = RR * T * log(c / onemc);

  return G_Ti_Liq(T) - G_Nb_Liq(T) + termL0 + log_part;
}

__host__ __device__ real Gsc(real T, real c) {
  real termL0 = L0_BCC(T) * (1.0 - 2.0 * c);

  real onemc = 1.0 - c;
  real log_part = 0.;
  if (c > 1.e-8 && onemc > 1.e-8)
    log_part = RR * T * log(c / onemc);

  return G_Ti_BCC(T) - G_Nb_BCC(T) + termL0 + log_part;
}

int get_local_grain(real *phi, int i, int j) {
  int a = 0, grain = -1;
  for (a = 0; a < num_orientation; a++)
    if (phi[psN(i, j, a)] >= 0)
      grain = a;
  return grain;
}

void get_itip_each_grain(real *phi, int *tip) {
  for (int a = 0; a < num_orientation; a++)
    tip[a] = 0;

  for (int j = 0; j < Ny; j++) {
    for (int i = 0; i < Nx; i++) {
      for (int a = 0; a < num_orientation; a++) {
        if (i > tip[a] && phi[psN(i, j, a)] >= phi_interface)
          tip[a] = i;
      }
    }
  }
}

// to get the precise j at the tip, use get_tip_int()
int get_itip_all_grain(real *psi) {
  int i_tip = 1;
  for (int j = 0; j < Ny; j++) {
    for (int i = i_tip; i < Nx; i++) {
      if (psi[pos(i, j)] > phi_interface)
        i_tip = i;
    }
  }
  return i_tip;
}

real get_average(real *data, int a) {
  real avg = 0, temp[Ny] = {0};
  for (int j = 1; j < Ny - 1; j++) {
    for (int i = 1; i < Nx - 1; i++) {
      temp[j] += data[psN(i, j, a)];
    }
    avg += temp[j];
  }
  avg /= (Nx - 2) * (Ny - 2);
  return avg;
}

void save_field(real *data, int index, const char *field) // flag
{
  char file_name[256];
  snprintf(file_name, sizeof(file_name), "%s/data/%s_%d.txt", path_input, field,
           index);

  FILE *fp = fopen(file_name, "w");
  if (fp == NULL) {
    printf("%s cannot be saved!!\n", file_name);
    return;
  } else {
    for (int j = 0; j < Ny; j = j + dj) {
      for (int i = 0; i < Nx; i = i + di) {
        if (strcmp(field, "grain") == 0)
          fprintf(fp, "%d\t", get_local_grain(data, i, j));
        else
          fprintf(fp, "%g\t", data[pos(i, j)]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
}

void save_T_full(real *T, constants *para, int index) {
  int NxT_all = para->NxT_all;
  char file_name[256];
  sprintf(file_name, "%s/data/T_full_%d.txt", path_input, index);

  FILE *fp = fopen(file_name, "w");
  if (fp == NULL) {
    printf("%s cannot be saved!!\n", file_name);
    return;
  } else {
    for (int jT = 0; jT < para->NyT; jT++) {
      for (int iT = 0; iT < para->NxT; iT++) {
        fprintf(fp, "%g\t", T[psT(iT, jT)]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
}

void save_final(real *h_phi) {
  char filename[256];
  snprintf(filename, sizeof(filename), "%s/data/grain_final.txt", path_input);
  FILE *fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("data/grain_final.txt cannot be saved!!\n");
    return;
  } else {
    for (int j = 0; j < Ny; j = j + dj) {
      for (int i = 0; i < Nx; i = i + di) {
        fprintf(fp, "%d\t", get_local_grain(h_phi, i, j));
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
}

void save_orientation(real *orientation) {
  char filename[256];
  snprintf(filename, sizeof(filename), "%s/data/orientation.txt", path_input);
  FILE *fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("data/orientation.txt cannot be saved!!\n");
    return;
  } else {
    for (int a = 0; a < num_orientation; a++) {
      for (int i = 0; i < num_axes; i++) {
        fprintf(fp, "%g\t", orientation[a * num_axes + i]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
}

void save_check_point(real *data, const char *field) // flag
{
  char fname[256];
  snprintf(fname, sizeof(fname), "%s/data/check_point_%s.bin", path_input,
           field);
  int num_slices = 1;
  if (strcmp(field, "phi") == 0)
    num_slices = num_orientation;

  ofstream fp(fname);
  if (!fp) {
    cout << fname << " cannot be saved!!" << endl;
    return;
  } else {
    for (int a = 0; a < num_slices; a++) {
      for (int j = 0; j < Ny_all; j++) {
        for (int i = 0; i < Nx_all; i++) {
          fp.write((char *)&(data[psN(i, j, a)]), sizeof(real));
        }
      }
    }
    fp.close();
    cout << "Saved: " << fname << endl;
  }
}

void save_check_point_temperature(real *T, constants *para) {
  int NxT_all = para->NxT_all;
  char fname[256];
  snprintf(fname, sizeof(fname), "%s/data/check_point_temperature.bin",
           path_input);
  ofstream fp(fname);
  if (!fp) {
    cout << fname << " cannot be saved!!" << endl;
    return;
  } else {
    for (int jT = 0; jT < para->NyT; jT++) {
      for (int iT = 0; iT < para->NxT; iT++) {
        fp.write((char *)&(T[psT(iT, jT)]), sizeof(real));
      }
    }
  }
  fp.close();
  cout << "Saved: " << fname << endl;
}

void load_check_point(real *data, const char *field) // flag
{
  char fname[256];
  snprintf(fname, sizeof(fname), "%s/data/check_point_%s.bin", path_input,
           field);
  int num_slices = 1;
  if (strcmp(field, "phi") == 0)
    num_slices = num_orientation;

  ifstream fp(fname);
  if (!fp) {
    cout << fname << " not found!" << endl;
    return;
  } else {
    cout << "Reading " << fname << endl;
    for (int a = 0; a < num_slices; a++) {
      for (int j = 0; j < Ny_all; j++) {
        for (int i = 0; i < Nx_all; i++) {
          fp.read((char *)&(data[psN(i, j, a)]), sizeof(real));
        }
      }
    }
  }
  fp.close();
}

void load_orientation(real *orientation) {
  char file_name[256];
  snprintf(file_name, sizeof(file_name), "%s/data/orientation.txt", path_input);
  FILE *fp = fopen(file_name, "r");

  if (fp == NULL) {
    printf("%s not found!\n", file_name);
    return;
  } else {
    cout << "Reading " << file_name << endl;
    for (int a = 0; a < num_orientation; a++) {
      for (int i = 0; i < num_axes; i++) {
        fscanf(fp, "%lf\t", &orientation[a * num_axes + i]);
      }
      fscanf(fp, "\n");
    }
    fclose(fp);
  }
}

void load_check_point_temperature(real *T, constants *para) {
  int NxT_all = para->NxT_all;

  char fname[256];
  snprintf(fname, sizeof(fname), "%s/data/check_point_temperature.bin",
           path_input);
  ifstream fp(fname);

  if (!fp) {
    cout << fname << " not found!" << endl;
    return;
  } else {
    cout << "Reading " << fname << endl;
    for (int jT = 0; jT < para->NyT; jT++) {
      for (int iT = 0; iT < para->NxT; iT++) {
        fp.read((char *)&(T[psT(iT, jT)]), sizeof(real));
      }
    }
    fp.close();
  }
}

void ensure_directory(const char *dirname) {
  // Try to create the directory with standard permissions
  if (mkdir(dirname, 0755) != 0) {
    // If mkdir failed for a reason other than "already exists", print error
    if (errno != EEXIST) {
      printf("Error: failed to create '%s': ", dirname);
      perror("");
    }
  }
}

__host__ __device__ real Vp_ramp(real t) {
  if constexpr (if_Vp_ramp) {
    t = t * 0.001; // convert ns to us
    return Vp + 0.0107587 * t;
  } else {
    return Vp;
  }
}
__host__ __device__ real x_pull_back(real t) { return Vp * t; }

void set_parameters(constants *para) {
  para->initial_step = 0;
  para->step = 0;
  real DD = Dl * tau0 / pow2(SS * W0); // dimensionless diffusivity

  para->dt = 0.6 * dx * dx / (4.0 * DD);
  if (para->dt > 0.6 * dx * dx / 4.0)
    para->dt = 0.6 * dx * dx / 4.0;

  para->i_ignore = 0;
  para->i_offset = 0;
  para->i_offset_history_initial = 0;
  para->i_offset_history = para->i_offset_history_initial;
  para->x_tip_history = i_tip_target * dx;
  para->if_begin_tip = 1;
  para->if_begin_field = 1;

  real dt = para->dt;
  para->total_step = total_time / (tau0 * dt);
  para->step_fields = total_time / num_fields_output / (tau0 * dt);
  para->step_tip = total_time / num_tip_output / (tau0 * dt);
  para->step_check_point = total_time / num_check_point_output / (tau0 * dt);

  if constexpr (if_Vp_ramp) {
    para->step_pull_back =
        SS * W0 * dx /
        (1.0 * dt * tau0); // 103; // number of steps to pull back

  } else {
    para->step_pull_back = SS * W0 * dx / (Vp * dt * tau0);
  }

  para->delta_initial = -1.0;
  para->delta = para->delta_initial;
  para->l_offset_Vp_initial = 0; // (real)i offset due to pull pack
  para->l_offset_Vp = 0;         // (real)i offset due to pull pack

  // int Block[2]={1,1};
  // AutoBlockSize(Block, Nx_all, Ny_all);
  // printf("BLOCK_SIZE_X=%d, BLOCK_SIZE_Y=%d\n", Block[0], Block[1]);

#if (ifTFC)
  real DDT = DT * tau0 /
             pow2(SS * W0 * ratio_dxT_dx); // dimensionless thermal diffusivity
  real dtT = 0.9 * dx * dx / (4.0 * DDT);

  para->ratio_dt_dtT = 1.0;
  if (dtT < para->dt)
    para->ratio_dt_dtT = ceil(para->dt / dtT);

  para->dtT = para->dt / para->ratio_dt_dtT;

  // real length = 50.0;                            // um
  para->length_sample = (Nx - 2) * SS * W0 * dx; // nanometers
  // ceil(length * 1000 / (dx * SS * W0 * ratio_dxT_dx)) * (dx * SS * W0 *
  // ratio_dxT_dx);
  para->NxT = para->length_sample / (dx * SS * W0 * ratio_dxT_dx) +
              2; // "2" is for the boundary
  para->NxT_all = ceil(1.0 * para->NxT / BLOCK_SIZE_T_X) *
                  BLOCK_SIZE_T_X;            // some lattices not used
  para->NyT = ((Ny - 2) / ratio_dxT_dx) + 2; // "2" is for the boundary
  para->NyT_all = ceil(1.0 * para->NyT / BLOCK_SIZE_T_Y) *
                  BLOCK_SIZE_T_Y; // some lattices not used

  para->Tcold = TCOLD;
  para->Thot = para->Tcold + GG * para->length_sample;

  para->i_left_end_initial =
      (Tmelt - deltaT0 - para->Tcold) / (GG * dx * SS * W0) - i_tip_target;
  para->i_left_end = para->i_left_end_initial;

  // AutoBlockSize(Block, para->NxT_all, para->NyT_all);
  // printf("T: BLOCK_SIZE_X=%d, BLOCK_SIZE_Y=%d\n", Block[0], Block[1]);
#endif
}

void save_parameters(constants *para, int i_tip) {
  char file_name[256];
  snprintf(file_name, sizeof(file_name), "%s/parameters.txt", path_input);
  FILE *fp = fopen(file_name, "w");

  if (fp == NULL) {
    printf("Cannot save parameters.txt!!\n");
    return;
  }

  real dt = para->dt;
  fprintf(fp, "----------------------------------------- Dynamically "
              "changing variables\n");
  fprintf(fp, "%-13d step\n", para->step);
  fprintf(fp, "%-13d i_offset_history, accumulated i offset\n",
          para->i_offset_history);
  fprintf(fp, "%-13g l_offset_Vp, unit: [W]; offset due to pulling velocity\n",
          para->l_offset_Vp);
  fprintf(fp, "%-13g delta_initial, initial dimensionless temperature\n",
          para->delta);
  fprintf(fp, "\n");

  fprintf(fp, "-----------------------------------------\n");
  fprintf(fp, "%-13d Nx_all\n", Nx_all);
  fprintf(fp, "%-13d Ny_all\n", Ny_all);
  fprintf(fp, "%-13g dx => %g nm\n", dx, dx * W0 * SS);
  fprintf(fp, "%-13g W0 [nm], physical interface thickness\n", W0 * 1.0);
  fprintf(fp, "%-13g S, scale factor\n", SS * 1.0);
  fprintf(fp, "%-13g W=S*W0, nm, effective interface thickness\n", SS * W0);
  fprintf(fp, "%-13d i_tip_target => %g um\n", i_tip_target,
          i_tip_target * dx * SS * W0 * 1e-3);
  fprintf(fp, "\n");

  fprintf(fp, "%-13g dt => %g ns\n", dt, dt * tau0);
  fprintf(fp, "%-13g tau0 [ns]; characteristic time\n", tau0);
  fprintf(fp, "%-13d total steps      => %g us\n", para->total_step,
          total_time * 1e-3);
  fprintf(fp, "%-13d step_fields      => %g us\n", para->step_fields,
          para->step_fields * tau0 * dt * 1e-3);
  fprintf(fp, "%-13d step_check_point => %g us\n", para->step_check_point,
          para->step_check_point * tau0 * dt * 1e-3);
  fprintf(fp, "%-13d step_tip         => %g ns\n", para->step_tip,
          para->step_tip * tau0 * dt);
  fprintf(fp, "%-13d step_pull_back   => %g ns\n", para->step_pull_back,
          para->step_pull_back * tau0 * dt);
  fprintf(fp, "\n");

  fprintf(fp, "%-13g Gamma [K*nm], Gibbs-Thomson coefficient\n", Gamma);
  fprintf(fp, "%-13g muk0 [nm/ns/K], interface kinetic coefficient\n", muk0);
  fprintf(fp, "%-13g nm^2/ns => %g m^2/s, Dl, diffusivity in the liquid\n", Dl,
          Dl * 1e-9);
  fprintf(fp, "%-13g K/nm => %g K/m, G, thermal gradient\n", GG, GG * 1e9);
  fprintf(fp, "%-13g Vp [m/s], pulling velocity \n", Vp);
  fprintf(fp, "\n");

  fprintf(fp, "%-13g Tmelt [K], melting temperature \n", Tmelt);
  fprintf(fp, "%-13g cinf, nominal composition \n", cinf);
  fprintf(fp, "%-13g deltaT0 [K], |me|*cinf for dilute alloys \n", deltaT0);
  fprintf(fp,
          "%-13g d0 [nm] = Gamma/deltaT0, capillary length, only for dilute "
          "alloys\n",
          d0);
  fprintf(fp, "%-13g ke, partition coefficient, only for dilute alloys\n", ke);
  fprintf(fp, "%-13g b=ln(ke)/2, only for dilute alloys\n", bb);
  fprintf(fp, "%-13g lambda, only for dilute alloys\n", lambda);
  fprintf(fp, "\n");

  fprintf(fp, "%-13g gamma0 [degrees], orientation angle\n", gamma0 * 1.0);
  fprintf(fp, "%-13g gamma1 [degrees], orientation angle\n", gamma1 * 1.0);
  fprintf(fp,
          "%-13g eps1, anisotropy parameter of the interfacial free energy \n",
          eps1);
  fprintf(fp,
          "%-13g eps2, anisotropy parameter of the interfacial free energy \n",
          eps2 * 1.0);
  fprintf(fp, "%-13g epk1, anisotropy parameter of the kinetic coefficient \n",
          epk1);
  fprintf(fp, "%-13ld random_seed\n", random_seed);
#if (if_load)
  fprintf(fp, "%-13s path_input\n", path_input);
#else
  fprintf(fp, "%-13s path_input\n", "./");
#endif
  fprintf(fp, "\n");

  fprintf(fp, "%-13g latentHeat [J/mol], latent heat\n", latentHeat);
  fprintf(fp, "%-13g cp [J/mol/K], specific heat\n", cp * 1.0);
  fprintf(fp, "%-13g Lcp = latentHeat/cp [K]\n", Lcp);
  fprintf(fp, "%-13g molar_weight [g/mol] \n", molar_weight);
  fprintf(fp, "%-13g density [g/cm^3]\n", density);
  fprintf(fp, "%-13g num_moles [mol/m^3], number of moles per m^3\n",
          num_moles);
  fprintf(fp, "%-13g latentHeat2 [J/m^3], latent heat\n", latentHeat2);
  fprintf(fp, "%-13g surface_tension [J/m^2]\n", surface_tension);
  fprintf(fp, "%-13g hh = surface_tension/(a1*S*W0*1e-9) [J/m^3]\n", hh);
  fprintf(fp, "%-13g h0 = RR*Tmelt/v0 [J/m^3]\n", h0);
  fprintf(fp, "\n");

#if (ifTFC)
  fprintf(fp, "----------------------------------------- Variables for "
              "temperature field calculation (TFC)\n");
  fprintf(fp, "-------------------- Dynamically changing variables\n");
  fprintf(fp,
          "%-13d i_left_end_initial, The i index of the left border of PF in "
          "the T field.\n",
          para->i_left_end_initial);
  fprintf(fp,
          "%-13d i_left_end, The i index of the left border of PF in the T "
          "field.\n",
          para->i_left_end);
  fprintf(fp, "\n");

  fprintf(fp, "%-13g dxT => %g nm\n", dxT, dxT * SS * W0);
  fprintf(fp, "%-13d dxT/dx, larger dx for the temperature field\n",
          ratio_dxT_dx);
  fprintf(fp, "%-13d NxT => %g um\n", para->NxT, para->NxT * dxT / 1000);
  fprintf(fp, "%-13d NyT => %g um\n", para->NyT, para->NyT * dxT / 1000);
  fprintf(fp, "%-13d NxT_all => %g um, some lattices not used\n", para->NxT_all,
          para->NxT_all * dxT / 1000);
  fprintf(fp, "%-13d NyT_all => %g um, some lattices not used\n", para->NyT_all,
          para->NyT_all * dxT / 1000);
  fprintf(fp, "%-13g sample length [nm] => %g um\n", para->length_sample,
          para->length_sample / 1000);
  fprintf(fp, "\n");

  fprintf(fp, "%-13g dtT => %g ns\n", para->dtT, para->dtT * tau0);
  fprintf(fp, "%-13d dt/dtT, smaller dt for the temperature field\n",
          para->ratio_dt_dtT);
  fprintf(fp, "\n");

  fprintf(fp, "%-13g D_T [nm^2/ns], thermal diffusivity\n", DT * 1.0);
  fprintf(fp, "%-13g Tcold [K]\n", para->Tcold);
  fprintf(fp, "%-13g Thot [K]\n", para->Thot);
#endif

  fclose(fp);
}

void load_parameters(constants *para) {
  char file_name[256];
  snprintf(file_name, sizeof(file_name), "%s/parameters.txt", path_input);
  FILE *fp = fopen(file_name, "r");
  if (fp == NULL) {
    printf("%s not found!\n", file_name);
    return;
  }

  printf("Reading %s\n", file_name);
  char dummy[512];    // Buffer to consume comments/headers
  real temp;          // Temp variable for reals not in 'para'
  int temp_int;       // Temp variable for ints not in 'para'
  long temp_long;     // Temp variable for longs not in 'para'
  char temp_str[256]; // Temp variable for strings

  // 1. Skip the first header line "----------------... Dynamically changing
  // variables:"
  fgets(dummy, sizeof(dummy), fp);

  // 2. Read Dynamically Changing Variables
  fscanf(fp, "%ld", &para->initial_step);
  fgets(dummy, sizeof(dummy), fp);
  fscanf(fp, "%d", &para->i_offset_history_initial);
  fgets(dummy, sizeof(dummy), fp);
  fscanf(fp, "%lf", &para->l_offset_Vp_initial);
  fgets(dummy, sizeof(dummy), fp);
  fscanf(fp, "%lf", &para->delta_initial);
  fgets(dummy, sizeof(dummy), fp);
  para->step = para->initial_step;
  para->i_offset_history = para->i_offset_history_initial;
  para->l_offset_Vp = para->l_offset_Vp_initial;
  para->delta = para->delta_initial;

  if (if_start_from_step0) {
    para->initial_step = 0;
  }
  // 3. Skip the separator section (Blank line + Dashed line)
  fgets(dummy, sizeof(dummy), fp); // Reads the blank line
  fgets(dummy, sizeof(dummy), fp); // Reads the "----------------..." line

  // 4. Read Grid/Physics Constants (Local vars in save function -> Read to
  // temp)
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // Nx_all
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // Ny_all
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // dx
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // W0
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // S
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // W
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // i_tip_target

  // (Blank line is automatically skipped by fscanf whitespace handling)

  // 5. Read Time Variables
  fscanf(fp, "%lf", &para->dt);
  fgets(dummy, sizeof(dummy), fp); // dt (maps to para field)
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // tau0
  fscanf(fp, "%ld", &temp_long);
  fgets(dummy, sizeof(dummy), fp); // total_step
  fscanf(fp, "%d", &temp_int);
  fgets(dummy, sizeof(dummy), fp); // step_fields
  fscanf(fp, "%d", &temp_int);
  fgets(dummy, sizeof(dummy), fp); // step_check_point
  fscanf(fp, "%d", &temp_int);
  fgets(dummy, sizeof(dummy), fp); // step_tip
  fscanf(fp, "%d", &temp_int);
  fgets(dummy, sizeof(dummy), fp); // step_pull_back

  // 6. Read Material Properties 1
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // Gamma
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // muk0
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // Dl
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // GG
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // Vp

  // 7. Read Material Properties 2
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // Tmelt
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // cinf
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // deltaT0
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // d0
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // ke
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // bb
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // lambda

  // 8. Read Anisotropy / Random / Path
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // gamma0
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // gamma1
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // eps1
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // eps2
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // epk1
  fscanf(fp, "%ld", &temp_long);
  fgets(dummy, sizeof(dummy), fp); // random_seed
  fscanf(fp, "%s", temp_str);
  fgets(dummy, sizeof(dummy), fp); // path_input

  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // latentHeat
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // cp
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // Lcp
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // molar_weight
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // density
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // num_moles
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // latentHeat2
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // surface_tension
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // hh
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // h0

// 9. Temperature Field Calculation (TFC) Block
#if (ifTFC)
  fgets(dummy, sizeof(dummy), fp); // Blank
  fgets(dummy, sizeof(dummy), fp); // ----------------... Variables for TFC
  fgets(dummy, sizeof(dummy),
        fp); // -------------------- Dynamically changing variables

  fscanf(fp, "%d", &para->i_left_end_initial);
  fgets(dummy, sizeof(dummy), fp);
  fscanf(fp, "%d", &para->i_left_end);
  fgets(dummy, sizeof(dummy), fp);

  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // dxT
  fscanf(fp, "%d", &temp_int);
  fgets(dummy, sizeof(dummy), fp); // ratio_dxT_dx
  fscanf(fp, "%d", &para->NxT);
  fgets(dummy, sizeof(dummy), fp);
  fscanf(fp, "%d", &para->NyT);
  fgets(dummy, sizeof(dummy), fp);
  fscanf(fp, "%d", &para->NxT_all);
  fgets(dummy, sizeof(dummy), fp);
  fscanf(fp, "%d", &para->NyT_all);
  fgets(dummy, sizeof(dummy), fp);
  fscanf(fp, "%lf", &para->length_sample);
  fgets(dummy, sizeof(dummy), fp);

  // Time TFC
  fscanf(fp, "%lf", &para->dtT);
  fgets(dummy, sizeof(dummy), fp);
  fscanf(fp, "%d", &para->ratio_dt_dtT);
  fgets(dummy, sizeof(dummy), fp);

  // Thermal Props
  fscanf(fp, "%lf", &temp);
  fgets(dummy, sizeof(dummy), fp); // DT
  fscanf(fp, "%lf", &para->Tcold);
  fgets(dummy, sizeof(dummy), fp);
  fscanf(fp, "%lf", &para->Thot);
  fgets(dummy, sizeof(dummy), fp);
#endif

  fclose(fp);
}

#define BSIZEMAX 32  // Maximum dimension for one side (e.g., 32x32 = 1024)
#define BLOCKMAX 512 // Maximum total threads per block (safe for most GPUs)

// Function to determine optimized Block Size
// Input: devNx, devNy (Grid Dimensions)
// Output: Bloc[0] -> BLOCK_SIZE_X, Bloc[1] -> BLOCK_SIZE_Y
void AutoBlockSize(int *Bloc, int devNx, int devNy) {
  int best_metric = 100000; // Initialize with a large value

  // Default fallback if no clean divisor is found
  Bloc[0] = 32;
  Bloc[1] = 8; // 32*8 = 256 threads

  // Iterate through possible block dimensions
  for (int bx = 1; bx <= BSIZEMAX; bx++) {
    // Check if bx divides the grid width evenly
    // (Use devNx+2 if your grid includes ghost nodes)
    if ((devNx % bx) == 0) {
      for (int by = 1; by <= BSIZEMAX; by++) {
        // Check thread count limit
        if (bx * by > BLOCKMAX)
          break;

        // Check if by divides the grid height evenly
        if ((devNy % by) == 0) {
          // Heuristic: We want the block to be as square as possible
          // because square tiles minimize the halo-to-computation
          // ratio (best surface-area-to-volume ratio for 2D).

          int diff = abs(bx - by); // "Squareness" penalty (0 is best)
          int size = bx * by;      // Total threads

          // Scoring Metric (Lower is better):
          // 1. (BLOCKMAX - size): Penalize small blocks (waste
          // occupancy)
          // 2. (diff * 5): Penalize rectangular blocks (want square)
          int current_metric = (BLOCKMAX - size) + (diff * 5);

          // Prefer x-dimension slightly larger for coalescing if diff
          // is same
          if (bx < by)
            current_metric += 1;

          if (current_metric < best_metric) {
            best_metric = current_metric;
            Bloc[0] = bx;
            Bloc[1] = by;
          }
        }
      }
    }
  }

  // Fallback logic: If we didn't find a perfect divisor (common with odd
  // sizes), reset to a standard efficient block size. 32 is standard for X to
  // match warp size (coalescing).
  if (best_metric == 100000) {
    Bloc[0] = 32;
    Bloc[1] = (BLOCKMAX / 32);
    // Ensure we don't exceed devNy if devNy is very small
    if (Bloc[1] > devNy)
      Bloc[1] = devNy;
  }
}

#define LENMAX 256
void GetMemUsage(int *Array, int Num) {
  char buffer[LENMAX];
  std::string StrUse = "";
  FILE *pipe = popen("nvidia-smi -q --display=MEMORY | grep Used ", "r");
  while (!feof(pipe)) {
    if (fgets(buffer, LENMAX, pipe) != NULL) {
      StrUse += buffer;
    }
  }
  pclose(pipe);
  for (int dev = 0; dev < Num; dev++) {
    std::istringstream iss(StrUse.substr(
        StrUse.find(":") + 1, StrUse.find("MB") - StrUse.find(":") - 1));
    iss >> Array[dev];
    StrUse = StrUse.substr(StrUse.find("\n") + 1,
                           StrUse.length() - StrUse.find("\n") - 1);
  }
}
int GetFreeDevice(int Num) {
  int FreeDev = -1;
  int MemFree = 15;
  int *Memory_Use = new int[Num];

  // Check utilization of Devices
  GetMemUsage(Memory_Use, Num);
  // See if one is free
  int dev = 0;
  do {
    if (Memory_Use[dev] < MemFree) {
      // Found one...
      FreeDev = dev;
      // Check if it is really free...
      system("sleep 1s");
      GetMemUsage(Memory_Use, Num);
      if (Memory_Use[dev] > MemFree) {
        FreeDev = -1;
      }
      // twice...
      system("sleep 1s");
      GetMemUsage(Memory_Use, Num);
      if (Memory_Use[dev] > MemFree) {
        FreeDev = -1;
      }
    }
    dev++;
  } while (FreeDev == -1 && dev < Num);

  delete[] Memory_Use;

  if (FreeDev == -1) {
    printf("=======================================\n");
    system("nvidia-smi -q --display=MEMORY |grep U");
    printf("=======================================\n");
    printf("NO AVAILABLE GPU: SIMULATION ABORTED...\n");
    printf("=======================================\n\n");
  }
  return FreeDev;
}

void DisplayDeviceProperties(int index) {
  cudaDeviceProp deviceProp;
  memset(&deviceProp, 0, sizeof(deviceProp));

  if (cudaSuccess == cudaGetDeviceProperties(&deviceProp, index)) {
    printf("==============================================================="
           "=================");
    printf("\nDevice Name \t %s ", deviceProp.name);
    printf("\nDevice Index\t %d ", index);
    printf("\n-------------------------------------------------------------"
           "-------------------");
    printf("\nTotal Global Memory                  \t %g GB",
           deviceProp.totalGlobalMem / pow3(1024.0));
    printf("\nShared memory available per block    \t %ld KB",
           (long int)(deviceProp.sharedMemPerBlock / 1024));
    printf("\nNumber of registers per block \t\t %d", deviceProp.regsPerBlock);
    printf("\nWarp size in threads             \t %d", deviceProp.warpSize);
    printf("\nMemory Pitch                     \t %g GB",
           deviceProp.memPitch / pow3(1024.0));
    printf("\nMaximum threads per block        \t %d",
           deviceProp.maxThreadsPerBlock);
    printf("\nMaximum Thread Dimension (block) \t %d * %d * %d",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("\nMaximum Thread Dimension (grid)  \t %d * %d * %d",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("\nTotal constant memory            \t %g KB",
           deviceProp.totalConstMem / 1024.0);
    printf("\nCUDA ver                         \t %d.%d", deviceProp.major,
           deviceProp.minor);
    int rate;
    if (cudaSuccess ==
        cudaDeviceGetAttribute(&rate, cudaDevAttrClockRate, index)) {
      printf("\nClock rate                       \t %g GHz",
             rate / pow2(1000.0));
    } else {
      printf("\nClock rate                       \t Not found");
    }
    // printf( "\nClock rate                       \t %g GHz",
    // deviceProp.clockRate/pow2(1000.0) );
    printf("\nTexture Alignment                \t %ld bytes",
           (long int)(deviceProp.textureAlignment));
    printf("\nDevice Overlap                   \t %s",
           deviceProp.asyncEngineCount ? "Allowed" : "Not Allowed");
    printf("\nNumber of Multi processors       \t %d",
           deviceProp.multiProcessorCount);
    printf("\n-------------------------------------------------------------"
           "-------------------");
    printf("\nNumber of registers per thread \t\t %d",
           deviceProp.regsPerBlock / (BLOCK_SIZE_X * BLOCK_SIZE_Y));
    printf("\nNumber of real varibles per thread \t %d",
           deviceProp.regsPerBlock / (BLOCK_SIZE_X * BLOCK_SIZE_Y) / 2);
    printf("\nNumber of int varibles per thread \t %d",
           deviceProp.regsPerBlock / (BLOCK_SIZE_X * BLOCK_SIZE_Y));
    printf("\n============================================================="
           "===================");
    printf("\n\n");
  } else {
    printf("\nCould not get properties for device %d.....\n", index);
  }
}

int set_cuda_device() {
  int CudaDevice = 0;
  int Ndevices = 0;

  cudaGetDeviceCount(&Ndevices); // Get the number of CUDA-capable devices

  if (Ndevices < 1) {
    printf("Error: No CUDA devices found.\n");
    return -1;
  }

  // If more than one device exists, check which one is free
  if (Ndevices > 1) {
    CudaDevice = GetFreeDevice(Ndevices);
    if (CudaDevice < 0) // CudaDevice = -1: no device is free
      return -1;
  }
  // If Ndevices == 1, we default to CudaDevice = 0

  cudaSetDevice(CudaDevice);
  DisplayDeviceProperties(CudaDevice);

  return 0; // Success
}

/////////////////////////////////////////////
///////////////////////////////////////////// calculate tips
/////////////////////////////////////////////
#define num_y_data 5 // num of points along y to do 3D surface fit
#define num_x_data 5 // to find the precise x at the interface
#define num_data (num_y_data)

typedef struct {
  int i{};
  int j{};
} ij;

typedef struct {
  int i{};
  int j{};
  real x{};
  real y{};
  real R{};
} TIP;

static inline int deal_with_boundary(int x, int N0, int Nn) {
  int Nmin = min(N0, Nn);
  int Nmax = max(N0, Nn);
#if (Periodic)
  if (x <= Nmin - 1)
    x += (Nmax - Nmin + 1);
  else if (x >= Nmax + 1)
    x -= (Nmax - Nmin + 1);
#elif (NoFlux)
  if (x < Nmin)
    x = 2 * Nmin - x;
  else if (x > Nmax)
    x = 2 * Nmax - x;
#endif

  return x;
}

// Helper: Solves A * c = b for a 5x5 system using Gaussian Elimination with
// partial pivoting
void solve_5x5(real A[5][5], real b[5], real c[5]) {
  int n = 5;
  int p[5]; // Permutation vector

  // Initialize permutation
  for (int i = 0; i < n; i++)
    p[i] = i;

  // Forward elimination
  for (int k = 0; k < n - 1; k++) {
    // Find pivot
    real max_val = fabs(A[p[k]][k]);
    int max_idx = k;
    for (int i = k + 1; i < n; i++) {
      if (fabs(A[p[i]][k]) > max_val) {
        max_val = fabs(A[p[i]][k]);
        max_idx = i;
      }
    }

    // Swap rows in permutation
    int temp = p[k];
    p[k] = p[max_idx];
    p[max_idx] = temp;

    // Eliminate
    for (int i = k + 1; i < n; i++) {
      real factor = A[p[i]][k] / A[p[k]][k];
      for (int j = k; j < n; j++) {
        A[p[i]][j] -= factor * A[p[k]][j];
      }
      b[p[i]] -= factor * b[p[k]];
    }
  }

  // Back substitution
  for (int i = n - 1; i >= 0; i--) {
    real sum = 0.0;
    for (int j = i + 1; j < n; j++) {
      sum += A[p[i]][j] * c[j];
    }
    c[i] = (b[p[i]] - sum) / A[p[i]][i];
  }
}

// Main Function: Fits x = f(y) and returns x at y = phi_interface
real get_precise_x(real *x, real *y, int n) {
  // We want to fit: x = c0 + c1*y + c2*y^2 + c3*y^3 + c4*y^4
  // System size is (Order+1) x (Order+1) = 5x5

  if (n < 5) {
    printf("Error: Need at least 5 points for 4th order fit.\n");
    return 0.0;
  }

  real A[5][5] = {0}; // Normal matrix
  real b[5] = {0};    // RHS vector
  real c[5] = {0};    // Coefficients result

  for (int i = 0; i < n; i++) {
    real xi = x[i];
    real yi = y[i];

    // Precompute powers of y up to y^8 (since matrix A needs y^(4+4))
    real yp[9];
    yp[0] = 1.0;
    for (int k = 1; k <= 8; k++) {
      yp[k] = yp[k - 1] * yi;
    }

    // Fill Matrix A (Symmetric) and Vector b
    // A_jk = Sum(y^(j+k))
    // b_j  = Sum(x * y^j)
    for (int j = 0; j < 5; j++) {
      b[j] += xi * yp[j];
      for (int k = 0; k < 5; k++) {
        A[j][k] += yp[j + k];
      }
    }
  }

  // Solve for coefficients
  solve_5x5(A, b, c);

  // Evaluate polynomial at phi_interface
  real phi = phi_interface;
  real precise_x = c[0] + c[1] * phi + c[2] * phi * phi +
                   c[3] * phi * phi * phi + c[4] * phi * phi * phi * phi;

  return precise_x;
}
// Solve A x = c for a 3x3 system
static int solve_3x3(real A[3][3], real cc[3], real x[3]) {
  for (int k = 0; k < 3; ++k) {
    int pivot = k;
    real maxval = fabs(A[k][k]);

    for (int i = k + 1; i < 3; ++i) {
      if (fabs(A[i][k]) > maxval) {
        maxval = fabs(A[i][k]);
        pivot = i;
      }
    }

    if (maxval < 1e-14)
      return -1; // singular

    // swap rows
    if (pivot != k) {
      for (int j = 0; j < 3; ++j) {
        real tmp = A[k][j];
        A[k][j] = A[pivot][j];
        A[pivot][j] = tmp;
      }
      real tmpb = cc[k];
      cc[k] = cc[pivot];
      cc[pivot] = tmpb;
    }

    real piv = A[k][k];
    for (int i = k + 1; i < 3; ++i) {
      real factor = A[i][k] / piv;
      A[i][k] = 0.0;
      for (int j = k + 1; j < 3; ++j)
        A[i][j] -= factor * A[k][j];
      cc[i] -= factor * cc[k];
    }
  }

  // back substitute
  for (int i = 2; i >= 0; --i) {
    real sum = cc[i];
    for (int j = i + 1; j < 3; ++j)
      sum -= A[i][j] * x[j];
    x[i] = sum / A[i][i];
  }

  return 0;
}

/*
 * Fit a parabola:
 *     x(y) = x0 + (y - y0)^2 / (2R)
 *
 * Steps:
 *   1) Fit quadratic x ≈ a0 + a1*y + a2*y^2
 *   2) Convert (a0,a1,a2) → (x0,y0,R)
 *
 * Inputs:
 *   x[], y[]  — arrays of length n
 *   n         — must be ≥ 3
 *
 * Outputs:
 *   *x0, *y0, *R
 *
 * Returns:
 *    0 — success
 *   -1 — not enough points
 *   -2 — singular matrix
 *   -3 — a2 ≈ 0 (no curvature)
 */
int fit_parabola(const real *x, const real *y, int n, real *x0, real *y0,
                 real *R) {
  if (n < 3)
    return -1;

  real M[3][3] = {{0}};
  real cc[3] = {0};
  real aa[3]; // output: a0, a1, a2

  for (int i = 0; i < n; ++i) {
    real yi = y[i];
    real xi = x[i];

    real f0 = 1.0;
    real f1 = yi;
    real f2 = yi * yi;

    // normal equations accumulation
    M[0][0] += f0 * f0;
    M[0][1] += f0 * f1;
    M[0][2] += f0 * f2;
    M[1][0] += f1 * f0;
    M[1][1] += f1 * f1;
    M[1][2] += f1 * f2;
    M[2][0] += f2 * f0;
    M[2][1] += f2 * f1;
    M[2][2] += f2 * f2;

    cc[0] += f0 * xi;
    cc[1] += f1 * xi;
    cc[2] += f2 * xi;
  }

  if (solve_3x3(M, cc, aa) != 0)
    return -2;

  real aa0 = aa[0];
  real aa1 = aa[1];
  real aa2 = aa[2];

  if (fabs(aa2) < 1e-14)
    return -3;

  // geometric radius
  real R_loc = 1.0 / (2.0 * aa2);

  // vertex position
  real y0_loc = -aa1 / (2.0 * aa2);

  real x0_loc = aa0 - (y0_loc * y0_loc) / (2.0 * R_loc);

  *x0 = x0_loc;
  *y0 = y0_loc;
  *R = R_loc;

  return 0;
}

int get_tip_int_at_j(real *psi, int i_initial, int j) {
  int itip = 0;
  for (int i = i_initial; i < Nx - 1; i++) {
    // temp = psi[pos(i, j)] - phi_interface;
    // if ( temp*(psi[pos(i+1,j)]-phi_interface)<0 ||
    //      abs(temp)<1.0e-6 )
    if (psi[pos(i, j)] > phi_interface) {
      itip = i;
    }
  }
  return itip;
}

void get_tip_int(real *phi, ij *tip) {
  real itip = 0.0;
  for (int j = 0; j < Ny; j++) {
    int i0 = get_tip_int_at_j(phi, int(itip), j);
    real iphi0 = (phi[pos(i0 + 1, j)] - phi[pos(i0, j)]) *
                     (phi_interface - phi[pos(i0, j)]) +
                 i0;
    if (iphi0 > itip) {
      itip = iphi0;
      tip->i = int(itip);
      tip->j = j;
    }
  }
}

void get_precise_tip(real *psi, TIP *tip, long int step) {
  ij itip;
  get_tip_int(psi, &itip);
  tip->i = itip.i;
  tip->j = itip.j;

  real i_data[num_x_data], psi_data[num_x_data]; // fit data along i
  real x_data[num_data], y_data[num_data];       // for parabola fit

  int mi = (num_x_data - 1) / 2, mj = (num_y_data - 1) / 2, ind = 0, i0;
  for (int j = tip->j - mj; j <= tip->j - mj + num_y_data - 1; j++) {
    int j0 = deal_with_boundary(j, jmin, jmax);
    i0 = get_tip_int_at_j(psi, tip->i - num_x_data, j0);

    // find the precise x at j
    for (int i = i0 - mi; i <= i0 - mi + num_x_data - 1; i++) {
      i_data[i - (i0 - mi)] = i;
      psi_data[i - (i0 - mi)] = psi[pos(i, j0)];
    }
    x_data[ind] = get_precise_x(i_data, psi_data, num_x_data);
    y_data[ind] = j0;

    ind++;
  }

  real x0, y0, R;
  int status = fit_parabola(x_data, y_data, num_data, &x0, &y0, &R);
  if (status != 0)
    printf("fit_parabola failed with code %d\n", status);

  tip->x = x0;
  tip->y = y0;
  tip->R = fabs(R) * dx * W0 * SS;
  if (tip->R > 1e5)
    tip->R = 1e5;
  if (tip->R < 1.0)
    tip->R = 1.0;

  // if (step==13541666 || step==13542708)
  // {
  //     char file_name[256];
  //     snprintf(file_name, sizeof(file_name), "xy_%d.txt", step);

  //     FILE *fp = fopen(file_name, "w");
  //     for (int i=0; i<num_data; i++)
  //     {
  //         fprintf(fp, "%g\t%g\n", x_data[i], y_data[i]);
  //     }
  //     fclose(fp);

  //     snprintf(file_name, sizeof(file_name), "tip_int_%d.txt", step);
  //     fp = fopen(file_name, "w");
  //     for (int j=765; j<801; j++)
  //     {
  //         int i0 = get_tip_int_at_j(psi, 0, j);
  //         fprintf(fp, "%d\t%d\n", i0, j);
  //     }
  //     fclose(fp);
  // }
}

void save_tip(long int step, constants *h_para, real delta, real T_tip, TIP tip,
              real V_tip, real *h_c, real *h_phi) // flag
{

  char file_name[256];
  snprintf(file_name, sizeof(file_name), "%s/tip.txt", path_input);
  FILE *fp = fopen(file_name, "a");
  if (h_para->if_begin_tip == 1) {
    fprintf(fp, "(1)        (2)        (3)        (4)        (5)        (6)    "
                "    (7)        (8)        (9)        (10)       (11)\n");
    fprintf(fp,
            "step       t[us]      Δtip       Ttip[K]    Vtip[m/s]  Vp[m/s]    "
            "tip.R[nm]  tip.i      tip.j      tip.x      tip.y\n");
    h_para->if_begin_tip = 0;
    V_tip = NAN;
  }
  fprintf(fp, "%-10d ", step);                            // (1)
  fprintf(fp, "%-10g ", step * h_para->dt * tau0 * 1e-3); // (2), us

  fprintf(fp, "%-10g ", delta); // (3)
  fprintf(fp, "%-10g ", T_tip); // (4), K
  fprintf(fp, "%-10g ", V_tip); // (5), m/s, instantaneous velocity of the
                                // solid-liquid interface
  fprintf(fp, "%-10g ",
          Vp_ramp(step * tau0 * h_para->dt)); // (6), m/s, pulling velocity

  fprintf(fp, "%-10g ", tip.R); // (7), tip radius
  fprintf(fp, "%-10d ", tip.i); // (8), position (integar) of the liquid front
                                // in the x direction
  fprintf(fp, "%-10d ", tip.j); // (9), position (integar) of the liquid front
                                // in the x direction
  fprintf(fp, "%-10g ", tip.x); // (10), position (integar) of the liquid
                                // front in the x direction
  fprintf(fp, "%-10g ", tip.y); // (11), position (integar) of the liquid
                                // front in the x direction

  fprintf(fp, "\n");
  fclose(fp);
}

void set_tip_output(real *h_c, real *h_phi, real *h_psi, real *h_T,
                    real *h_T_fine, constants *h_para, real *h_orientation,
                    real *d_c, real *d_phi, real *d_psi, real *d_T,
                    real *d_T_fine, constants *d_para, size_t size_grid,
                    size_t size_grid_T, real start_time, long int step,
                    dim3 numBlocks, dim3 threadsPerBlock) {
  if (step % h_para->step_tip == 0 || step == h_para->initial_step) // flag
  {
    real time_now = clock() / (1.0 * CLOCKS_PER_SEC);
    real dt = h_para->dt;
    // cudaMemcpy(h_c, d_c, size_grid, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_phi, d_phi, num_orientation*size_grid,
    // cudaMemcpyDeviceToHost);
    cudaMemcpy(h_psi, d_psi, size_grid, cudaMemcpyDeviceToHost);

    TIP tip;
    get_precise_tip(h_psi, &tip, step);

    real x_tip_now = (h_para->i_offset_history + tip.x) * dx;
    real V_tip = (x_tip_now - h_para->x_tip_history) * SS * W0 /
                 (h_para->step_tip * tau0 * dt); // m/s
    h_para->x_tip_history = x_tip_now;
    if (V_tip < 0.0)
      V_tip = 0.01;
    if (V_tip > 10.0)
      V_tip = 10.0;

#if (ifTFC)
    interpolate_temperature<<<numBlocks, threadsPerBlock>>>(
        d_T, d_T_fine, d_para, h_para->i_left_end, step);
    cudaMemcpy(h_T_fine, d_T_fine, size_grid, cudaMemcpyDeviceToHost);

    real T_tip = h_T_fine[pos(tip.i, 0)];
    real delta = (T_tip - Tmelt) / deltaT0;
#else
    real delta =
        get_delta(h_para, tip.i, h_para->i_offset_history, h_para->l_offset_Vp);
    real T_tip = delta * deltaT0 + Tmelt;
#endif
    h_para->delta = delta; // for save_parameters()

    // printf("step=%d, delta=%g, i_offset_history=%d, l_offset_Vp=%g,
    // i_left_end=%d, i_left_end_initial=%d\n",
    //     step, delta, h_para->i_offset_history, h_para->l_offset_Vp,
    //     h_para->i_left_end, h_para->i_left_end_initial);

    save_tip(step, h_para, delta, T_tip, tip, V_tip, h_c, h_phi);
  }
}

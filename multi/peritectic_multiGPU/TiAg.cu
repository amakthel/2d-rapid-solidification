#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <unistd.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cuda.h>

using namespace std;

#define 	DT 0.015
#define		DXin 2.0
#define		DYin 2.0
#define 	DZin 2.0
#define 	Nx 512
#define		Ny 512
#define 	Nz 506 // Nz = (devNz-2)*Ngpu + 2
#define 	Ngpu 4
#define 	devNz 128 // devNz = (Nz-2)/Ngpu + 2 
#define 	di 4
#define 	dj 4
#define 	dk 4
#define		total_step (atoi(argv[1]))
#define 	out_step (atoi(argv[2]))
#define 	BLOCK_SIZE_X 16
#define 	BLOCK_SIZE_Y 8
#define 	BLOCK_SIZE_Z 4

#define 	a1 0
#define 	a2 4.43
#define 	a3 6.09
#define 	b  12.0

#define 	lambda 30.0
#define 	alpha 2.0
#define 	Da 0
#define 	Db 0
#define		Cp 		0.9395
#define 	Cpa 	0.1641
#define 	Cpb 	0.5
#define 	ma 		(-4007.8)
#define 	mb 		(-2618.7)
#define		T 		(60.0*0.000321787)

#define 	A1 	0.391701
#define 	A2 	0.216598
#define 	B11 (0.0685881)
#define 	B21 (0.037927)
#define 	B12 (-0.466867)
#define 	B22 (-0.0662661) 
#define		B1 	(B11+B12*T)
#define		B2 	(B21+B22*T)

#define		ca	( (B1+0.5*B2)/(A1+0.5*A2)-A1-A2 )
#define		cb	( (B1-0.5*B2)/(A1-0.5*A2)-A1+A2 )
#define		cla ( (B1+0.5*B2)/(A1+0.5*A2)+A1 )
#define		clb ( (B1-0.5*B2)/(A1-0.5*A2)+A1 )

#define 	pos(x,y,z)	(Nx*Ny*(z)+Nx*(y)+(x))
#define 	pow2(x) ((x)*(x))
#define 	pow3(x) ((x)*(x)*(x))
#define 	pow4(x) ((x)*(x)*(x)*(x))

#define 	g(pi,pj,pk)	( pow2(pi)*0.25*( 15.0*(1-(pi))*( 1+(pi)-pow2((pk)-(pj)) ) + (pi)*(9.0*pow2(pi)-5.0) ) )
#define		gp1(pi,pj,pk) ( 3.75*(pi)*( -4.0*pow2(pi) + 3.0*pow3(pi) + 2.0 - (pi) + (3.0*(pi)-2.0)*pow2((pj)-(pk)) ) ) // paritial gi / paritial pi
#define 	gp2(pi,pj,pk) ( 7.5*(pi-1.0)*pow2(pi)*(pj-pk) ) // partial gi / partial pj

void enableP2P();
void disableP2P();
void isUnifiedAddressing();

void initial(double *phi1, double *phi2, double *c, double *mu, int *other );
__global__ void computeMu( double *phi1, double *phi2, double *c, double *mu );
__global__ void boundaryMuX( double *mu );
__global__ void boundaryMuY( double *mu );
__global__ void boundaryMuZ( double *mu );
__global__ void diffuse( double *phi1, double *phi2, double *c, double *mu, double *phi1_next, double *phi2_next, double *c_next, int dev, int step);
__global__ void boundaryX( double *phi1_next, double *phi2_next, double *c_next );
__global__ void boundaryY( double *phi1_next, double *phi2_next, double *c_next );
__global__ void boundaryZ( double *phi1_next, double *phi2_next, double *c_next );
float ReverseFloat( const float inFloat );
void c2vtk(double *c, int step);
void data2File2D(double *data, int step, int flag, char *CrossSection, int xx);

void save_break_point(double *data, int flag);
void save_break_point2(int *data);
void load_break_point(double *data, int flag);
void load_break_point2(int *data);

int main(int argc, char **argv)
{
	// Check available number of gpus
	int Ngpu_available;
	cudaGetDeviceCount(&Ngpu_available);
	if (Ngpu > Ngpu_available)
	{
		printf("Invalid number of GPUs specified: %i is greater "
			"than the total number of GPUs in this platform (%i)\n", Ngpu, Ngpu_available);
		exit(1);
	}
	printf("Run with %i devices\n", Ngpu);

	//Check the compatibility of UVA and P2P
	isUnifiedAddressing();
	enableP2P();

	int step, i, j, k, *other;
	double *h_phi1, *h_phi2, *h_c, *h_mu;
	double *d_phi1[Ngpu], *d_phi2[Ngpu], *d_c[Ngpu], *d_mu[Ngpu], *d_phi1_next[Ngpu], *d_phi2_next[Ngpu], *d_c_next[Ngpu];
	double *halo_buff[Ngpu];
	double *phi1_buffer, *phi2_buffer, *c_buffer;
	double start_time = clock()/(1.0*CLOCKS_PER_SEC), end_time;

	size_t size_h = Nx*Ny*Nz * sizeof(double);
	size_t size_d = Nx*Ny*devNz * sizeof(double);
	size_t size_d_1 = Nx*Ny*(devNz - 1) * sizeof(double);
	size_t size_d_2 = Nx*Ny*(devNz - 2) * sizeof(double);
	size_t halo_size = Nx*Ny * sizeof(double);

	h_phi1 = (double*)malloc(size_h);
	h_phi2 = (double*)malloc(size_h);
	h_c = (double*)malloc(size_h);
	h_mu = (double*)malloc(size_h);
	other = (int*)malloc(sizeof(int));

	for ( i = 0; i < Ngpu; i++)
	{
		cudaSetDevice(i);

		cudaMalloc((void**)&d_phi1[i], size_d);
		cudaMalloc((void**)&d_phi2[i], size_d);
		cudaMalloc((void**)&d_c[i], size_d);
		cudaMalloc((void**)&d_mu[i], size_d);
		cudaMalloc((void**)&d_phi1_next[i], size_d);
		cudaMalloc((void**)&d_phi2_next[i], size_d);
		cudaMalloc((void**)&d_c_next[i], size_d);
		cudaMalloc((void**)&halo_buff[i], halo_size);

		cudaMemset(d_phi1[i], 0, size_d);
		cudaMemset(d_phi2[i], 0, size_d);
		cudaMemset(d_c[i], 0, size_d);
		cudaMemset(d_mu[i], 0, size_d);
		cudaMemset(d_phi1_next[i], 0, size_d);
		cudaMemset(d_phi2_next[i], 0, size_d);
		cudaMemset(d_c_next[i], 0, size_d);
		cudaMemset(halo_buff[i], 0, halo_size);
	}

	dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 numBlocks(Nx/threadsPerBlock.x, Ny/threadsPerBlock.y, devNz/threadsPerBlock.z);

	dim3 threadsPerBlockX(BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 numBlocksX(Ny / BLOCK_SIZE_Y, devNz / BLOCK_SIZE_Z);

	dim3 threadsPerBlockY(BLOCK_SIZE_X, BLOCK_SIZE_Z);
	dim3 numBlocksY(Nx / BLOCK_SIZE_X, devNz / BLOCK_SIZE_Z);

	dim3 threadsPerBlockZ(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 numBlocksZ(Nx / BLOCK_SIZE_X, Ny / BLOCK_SIZE_Y);

	int h_addr_shift[Ngpu]; // cudaMemcpyHostToDevice, on host
	int h_addr_shift2[Ngpu]; // cudaMemcpyDeviceToHost, on host
	int d_addr_shift2[Ngpu]; // cudaMemcpyDeviceToHost, on device
	for ( i = 0; i < Ngpu; i++)
	{
		h_addr_shift[i] = i*(devNz - 2)*Nx*Ny;
		if (i == 0)
		{
			h_addr_shift2[i] = 0;
			d_addr_shift2[i] = 0;
		}
		else
		{
			h_addr_shift2[i] = (i*(devNz - 2) + 1)*Nx*Ny;
			d_addr_shift2[i] = Nx*Ny;
		}
	}

	int repeat = atof(argv[3]);
	if (repeat==0)
	{
		initial(h_phi1, h_phi2, h_c, h_mu, other); //initial h_c
	}
	else
	{
		load_break_point(h_c, 1);
		load_break_point(h_phi1, 2);
		load_break_point(h_phi2, 3);
		load_break_point(h_mu, 4);
		load_break_point2(other);

		printf("Load data finished!\n");
	}

	cudaStream_t stream[Ngpu];
	for ( i = 0; i < Ngpu; i++)
	{
		cudaSetDevice(i);
		cudaStreamCreate(&stream[i]); // Creat cuda streams
		cudaMemcpy(d_phi1[i], h_phi1 + h_addr_shift[i], size_d, cudaMemcpyHostToDevice);
		cudaMemcpy(d_phi2[i], h_phi2 + h_addr_shift[i], size_d, cudaMemcpyHostToDevice);
		cudaMemcpy(d_c[i], h_c + h_addr_shift[i], size_d, cudaMemcpyHostToDevice);
		cudaMemcpy(d_mu[i], h_mu + h_addr_shift[i], size_d, cudaMemcpyHostToDevice);
	}

	FILE * file_pointer = fopen("out2.txt","w");
	for (step=other[0]; step<=total_step; step++)
	{
		if ( step%out_step==0 )
		{
			for ( i = 0; i < Ngpu; i++)
				cudaStreamSynchronize(stream[i]);
			for ( i = 0; i < Ngpu; i++)
			{
				cudaSetDevice(i);
				size_t size_d_temp;
				if (i == 0 || i == Ngpu-1)
				{
					size_d_temp = size_d_1;
				}
				else
				{ 
					size_d_temp = size_d_2;
				}
				cudaMemcpyAsync(h_c + h_addr_shift2[i], d_c[i] + d_addr_shift2[i], size_d_temp, cudaMemcpyDeviceToHost, stream[i]);
				// cudaMemcpyAsync(h_phi1 + h_addr_shift2[i], d_phi1[i] + d_addr_shift2[i], size_d_temp, cudaMemcpyDeviceToHost, stream[i]);
				// cudaMemcpyAsync(h_phi2 + h_addr_shift2[i], d_phi2[i] + d_addr_shift2[i], size_d_temp, cudaMemcpyDeviceToHost, stream[i]);
				// cudaMemcpyAsync(h_mu + h_addr_shift2[i], d_mu[i] + d_addr_shift2[i], size_d_temp, cudaMemcpyDeviceToHost, stream[i]);
			}

			double avg_phi1 = 0, avg_phi2 = 0, avg_c = 0;
			for (k=1; k<Nz-1; k++)
			{
				for (j=1; j<Ny-1; j++)
				{
					for (i=1; i<Nx-1; i++)
					{
						avg_phi1 += 0;//h_phi1[pos(i,j,k)]*1.0/(Nx-2)/(Ny-2)/(Nz-2);
						avg_phi2 += 0;//h_phi2[pos(i,j,k)]/(Nx-2)/(Ny-2)/(Nz-2);
						avg_c += h_c[pos(i,j,k)]/(Nx-2)/(Ny-2)/(Nz-2);
					}
				}
			}

			/////////////////////// output ///////////////////////////// tag
			end_time=clock()/(1.0*CLOCKS_PER_SEC);
			printf("step = %d\tt = %gs\tphi1 = %g\tphi2 = %g\tc = %.15f\n",
				step, end_time - start_time,
				avg_phi1, avg_phi2,
				avg_c);

			fprintf(file_pointer,"%d\t%g\t%g\t%.15f\n", step, avg_phi1, avg_phi2, avg_c);
			fflush(file_pointer);
			
			c2vtk(h_c, step);
			char cross[] = "xz";
			data2File2D(h_c, step, 1, cross, Ny/2);
			// data2File2D(h_phi1, step, 2, cross, Ny/2);
			// data2File2D(h_phi2, step, 3, cross, Ny/2);

			for (j=1; j<Ny-1; j++)
			{
				for (i=1; i<Nx-1; i++)
				{
					if ( abs(h_c[pos(i,j,20)])>0.1 )
					{
						fprintf(file_pointer, "Stops at step = %d\n", step);
						goto JUMP;
					}
				}
			}
		}

		////////////////////////////////////////////// compute μ /////////////////////////////////////////////////
		for ( i = 0; i < Ngpu; i++)
			cudaStreamSynchronize(stream[i]);
		for ( i = 0; i < Ngpu; i++)
		{
			cudaSetDevice(i);
			computeMu <<< numBlocks, threadsPerBlock, 0, stream[i] >>> (d_phi1[i], d_phi2[i], d_c[i], d_mu[i]);
			boundaryMuX <<< numBlocksX, threadsPerBlockX, 0, stream[i] >>> (d_mu[i]);
			boundaryMuY <<< numBlocksY, threadsPerBlockY, 0, stream[i] >>> (d_mu[i]);
			boundaryMuZ <<< numBlocksZ, threadsPerBlockZ, 0, stream[i] >>> (d_mu[i]);
		}
		// Exchange halo
		for ( i = 0; i < Ngpu; i++)
			cudaStreamSynchronize(stream[i]);
		for ( i = 0; i < Ngpu-1; i++)
		{
			cudaMemcpyAsync(halo_buff[i], d_mu[i] + (devNz - 1)*Nx*Ny, halo_size, cudaMemcpyDefault, stream[i]);
			cudaMemcpyAsync(d_mu[i] + (devNz - 1)*Nx*Ny, d_mu[i+1], halo_size, cudaMemcpyDefault, stream[i]);
			cudaMemcpyAsync(d_mu[i+1], halo_buff[i], halo_size, cudaMemcpyDefault, stream[i]);
		}
		
		////////////////////////////////////////////// compute φ and c ////////////////////////////////////////////
		for ( i = 0; i < Ngpu; i++)
			cudaStreamSynchronize(stream[i]);
		for ( i = 0; i < Ngpu; i++)
		{
			cudaSetDevice(i);
			diffuse <<< numBlocks, threadsPerBlock, 0, stream[i] >>> (d_phi1[i], d_phi2[i], d_c[i], d_mu[i], d_phi1_next[i], d_phi2_next[i], d_c_next[i],
					i, step);
			boundaryX <<< numBlocksX, threadsPerBlockX, 0, stream[i] >>> (d_phi1_next[i], d_phi2_next[i], d_c_next[i]);
			boundaryY <<< numBlocksY, threadsPerBlockY, 0, stream[i] >>> (d_phi1_next[i], d_phi2_next[i], d_c_next[i]);
			boundaryZ <<< numBlocksZ, threadsPerBlockZ, 0, stream[i] >>> (d_phi1_next[i], d_phi2_next[i], d_c_next[i]);
		}

		// Exchange halo
		for ( i = 0; i < Ngpu; i++)
			cudaStreamSynchronize(stream[i]);
		for ( i = 0; i < Ngpu-1; i++)
		{
			cudaMemcpyAsync(halo_buff[i], d_phi1_next[i] + (devNz - 1)*Nx*Ny, halo_size, cudaMemcpyDefault, stream[i]);
			cudaMemcpyAsync(d_phi1_next[i] + (devNz - 1)*Nx*Ny, d_phi1_next[i+1], halo_size, cudaMemcpyDefault, stream[i]);
			cudaMemcpyAsync(d_phi1_next[i+1], halo_buff[i], halo_size, cudaMemcpyDefault, stream[i]);

			cudaMemcpyAsync(halo_buff[i], d_phi2_next[i] + (devNz - 1)*Nx*Ny, halo_size, cudaMemcpyDefault, stream[i]);
			cudaMemcpyAsync(d_phi2_next[i] + (devNz - 1)*Nx*Ny, d_phi2_next[i+1], halo_size, cudaMemcpyDefault, stream[i]);
			cudaMemcpyAsync(d_phi2_next[i+1], halo_buff[i], halo_size, cudaMemcpyDefault, stream[i]);

			cudaMemcpyAsync(halo_buff[i], d_c_next[i] + (devNz - 1)*Nx*Ny, halo_size, cudaMemcpyDefault, stream[i]);
			cudaMemcpyAsync(d_c_next[i] + (devNz - 1)*Nx*Ny, d_c_next[i+1], halo_size, cudaMemcpyDefault, stream[i]);
			cudaMemcpyAsync(d_c_next[i+1], halo_buff[i], halo_size, cudaMemcpyDefault, stream[i]);
		}

		////////////////////////////////////////////// exchange pointers /////////////////////////////////////////////
		for ( i = 0; i < Ngpu; i++)
			cudaStreamSynchronize(stream[i]);
		for ( i = 0; i < Ngpu; i++)
		{
			cudaSetDevice(i);
			phi1_buffer = d_phi1[i]; 	d_phi1[i] = d_phi1_next[i]; 	d_phi1_next[i] = phi1_buffer;
			phi2_buffer = d_phi2[i]; 	d_phi2[i] = d_phi2_next[i]; 	d_phi2_next[i] = phi2_buffer;
			c_buffer = d_c_next[i];		d_c_next[i] = d_c[i];			d_c[i] = c_buffer;
		}
	}

	JUMP: fclose(file_pointer);

	//////////////////////////// copy variables to cpu then save them
	other[0] = step;
	for ( i = 0; i < Ngpu; i++)
	{
		cudaSetDevice(i);
		size_t size_d_temp;
		if (i == 0 || i == Ngpu-1)
		{
			size_d_temp = size_d_1;
		}
		else
		{ 
			size_d_temp = size_d_2;
		}
		cudaMemcpyAsync(h_c + h_addr_shift2[i], d_c[i] + d_addr_shift2[i], size_d_temp, cudaMemcpyDeviceToHost, stream[i]);
		cudaMemcpyAsync(h_phi1 + h_addr_shift2[i], d_phi1[i] + d_addr_shift2[i], size_d_temp, cudaMemcpyDeviceToHost, stream[i]);
		cudaMemcpyAsync(h_phi2 + h_addr_shift2[i], d_phi2[i] + d_addr_shift2[i], size_d_temp, cudaMemcpyDeviceToHost, stream[i]);
		cudaMemcpyAsync(h_mu + h_addr_shift2[i], d_mu[i] + d_addr_shift2[i], size_d_temp, cudaMemcpyDeviceToHost, stream[i]);
	}
	for ( i = 0; i < Ngpu; i++)
		cudaStreamSynchronize(stream[i]);

	save_break_point(h_c, 1);
	save_break_point(h_phi1, 2);
	save_break_point(h_phi2, 3);
	save_break_point(h_mu, 4);
	save_break_point2(other);

	//////////////////////////////// free variables
	void disableP2P();
	for ( i = 0; i < Ngpu; i++)
	{
		cudaSetDevice(i);
		cudaStreamDestroy(stream[i]);
		cudaFree(d_phi1[i]);
		cudaFree(d_phi2[i]);
		cudaFree(d_c[i]);
		cudaFree(d_mu[i]);
		cudaFree(d_phi1_next[i]);
		cudaFree(d_phi2_next[i]);
		cudaFree(d_c_next[i]);
		cudaFree(halo_buff[i]);
	}

	free(h_phi1);
	free(h_phi2);
	free(h_c);
	free(h_mu);
	free(other);

	return EXIT_SUCCESS;
}

void initial( double *phi1, double *phi2, double *c, double *mu, int *other)
{
	double R = 30.0;
	double theta = 1.33;
	double radius = R/sin(theta);
	double roff = R/tan(theta);

	int zcut = Nz-100;
	for (int k=0; k<Nz; k++)
	{
		for (int j=0; j<Ny; j++)
		{
			for (int i=0; i<Nx; i++)
			{
				int ps = pos(i,j,k);
				if ( k>=zcut && ( pow2(i-128)+pow2(j-128)+pow2(k-(zcut-roff))<=pow2(radius)
						|| pow2(i-300)+pow2(j-240)+pow2(k-(zcut-roff))<=pow2(radius) ) )
				{
					phi1[ps] = 0;
					phi2[ps] = 1.0;
					c[ps] = ca;
				}
				else if ( k<zcut )
				{
					phi1[ps] = 0;
					phi2[ps] = 0;
					c[ps] = cb;
				}
				else 
				{
					phi1[ps] = 1.0;
					phi2[ps] = 0;
					c[ps] = clb;
				}
				double p1 = phi1[ps], p2 = phi2[ps], p3 = 1.0-p1-p2;
				double r1 = A1, r2 = -A1-A2, r3 = -A1+A2;
				mu[ps] = c[ps] - r1*g(p1,p2,p3) - r2*g(p2,p3,p1) - r3*g(p3,p1,p2);
			}
		}
	}

	other[0] = 0;
}


__global__ void computeMu( double *phi1, double *phi2, double *c, double *mu)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int ps = pos(i,j,k);

	double p1 = phi1[ps], p2 = phi2[ps], p3 = 1.0-p1-p2;
	double r1 = A1, r2 = -A1-A2, r3 = -A1+A2;
	mu[ps] = c[ps] - r1*g(p1,p2,p3) - r2*g(p2,p3,p1) - r3*g(p3,p1,p2);
}

__global__ void boundaryMuX(double *mu)
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;

	mu[pos(0, j, k)] = mu[pos(Nx-2, j, k)];
	mu[pos(Nx-1, j, k)] = mu[pos(1, j, k)];
}

__global__ void boundaryMuY(double *mu)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;

	mu[pos(i, 0, k)] = mu[pos(i, Ny-2, k)];
	mu[pos(i, Ny-1, k)] = mu[pos(i, 1, k)];
}

__global__ void boundaryMuZ(double *mu)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	mu[pos(i, j, 0)] = mu[pos(i, j, 1)];
	mu[pos(i, j, devNz-1)] = mu[pos(i, j, devNz-2)];
}


__global__ void diffuse( double *phi1, double *phi2, double *c, double *mu, double *phi1_next, double *phi2_next, double *c_next, int dev, int step)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int ps = pos(i,j,k);

	if (i*j*k>0 && i<Nx-1 && j<Ny-1 && k<devNz-1)
	{
		////////////////////////////////////////////// compute phi ///////////////////////////////////////
		double p1 = phi1[ps], p2 = phi2[ps], p3 = 1.0-p1-p2, muu = mu[ps];
		double r1 = A1, r2 = -A1-A2, r3 = -A1+A2;
		double s1 = B1, s2 = -B1-B2, s3 = -B1+B2;
		double fcphi1 = lambda*( gp1(p1,p2,p3)*(s1-r1*muu) + gp2(p2,p1,p3)*(s2-r2*muu) + gp2(p3,p1,p2)*(s3-r3*muu) );
		double fcphi2 = lambda*( gp2(p1,p2,p3)*(s1-r1*muu) + gp1(p2,p3,p1)*(s2-r2*muu) + gp2(p3,p2,p1)*(s3-r3*muu) );
		double fcphi3 = lambda*( gp2(p1,p3,p2)*(s1-r1*muu) + gp2(p2,p3,p1)*(s2-r2*muu) + gp1(p3,p1,p2)*(s3-r3*muu) );

		double laplacian_p1 = (phi1[pos(i+1,j,k)]+phi1[pos(i-1,j,k)]-2.0*p1)*DXin*DXin + (phi1[pos(i,j+1,k)]+phi1[pos(i,j-1,k)]-2.0*p1)*DYin*DYin + (phi1[pos(i,j,k+1)]+phi1[pos(i,j,k-1)]-2.0*p1)*DZin*DZin;
		double laplacian_p2 = (phi2[pos(i+1,j,k)]+phi2[pos(i-1,j,k)]-2.0*p2)*DXin*DXin + (phi2[pos(i,j+1,k)]+phi2[pos(i,j-1,k)]-2.0*p2)*DYin*DYin + (phi2[pos(i,j,k+1)]+phi2[pos(i,j,k-1)]-2.0*p2)*DZin*DZin;
		double laplacian_p3 = - laplacian_p1 - laplacian_p2;

		double fphi1 = 2.0*p1*(1.0-p1)*(1.0-2.0*p1) + a1*pow2(p2*p3)*(3.0+2*b*p1) + 2.0*a2*p1*pow2(p3)*(3*p1*p3+3.0*p2+b*pow2(p2))
						+ 2.0*a3*p1*pow2(p2)*(3.0*p1*p2+3.0*p3+b*pow2(p3)) + fcphi1;
		double fphi2 = 2.0*p2*(1.0-p2)*(1.0-2.0*p2) + 2.0*a1*p2*pow2(p3)*(3*p2*p3+3.0*p1+b*pow2(p1)) + a2*pow2(p1*p3)*(3.0+2*b*p2) 
						+ 2.0*a3*p2*pow2(p1)*(3.0*p1*p2+3.0*p3+b*pow2(p3)) + fcphi2;
		double fphi3 = 2.0*p3*(1.0-p3)*(1.0-2.0*p3) + 2.0*a1*p3*pow2(p2)*(3.0*p2*p3+3.0*p1+b*pow2(p1)) + 2.0*a2*p3*pow2(p1)*(3*p1*p3+3.0*p2+b*pow2(p2))
						 + a3*pow2(p1*p2)*(3.0+2*b*p3) + fcphi3;

		double muphi1 = fphi1 - laplacian_p1;
		double muphi2 = fphi2 - laplacian_p2;
		double muphi3 = fphi3 - laplacian_p3;

		double phi1t = - (2.0*muphi1 - muphi2 - muphi3)*0.333333333333333333333333333;
		double phi2t = - (2.0*muphi2 - muphi1 - muphi3)*0.333333333333333333333333333;
		// double phi3t = - (2.0*muphi3 - muphi1 - muphi2)*0.333333333333333333333333333;

		phi1_next[ps] = p1 + DT*phi1t;
		phi2_next[ps] = p2 + DT*phi2t;

		/////////////////////////////////// compute c ///////////////////////////////////////
		double Hxp = ( 0.5*(p1+phi1[pos(i+1,j,k)]) + Da*0.5*(p2+phi2[pos(i+1,j,k)]) + Db*0.5*(p3+1.0-phi1[pos(i+1,j,k)]-phi2[pos(i+1,j,k)]) ) * (mu[pos(i+1,j,k)]-muu)*DXin;
		double Hxn = ( 0.5*(p1+phi1[pos(i-1,j,k)]) + Da*0.5*(p2+phi2[pos(i-1,j,k)]) + Db*0.5*(p3+1.0-phi1[pos(i-1,j,k)]-phi2[pos(i-1,j,k)]) ) * (muu-mu[pos(i-1,j,k)])*DXin;
		double Hyp = ( 0.5*(p1+phi1[pos(i,j+1,k)]) + Da*0.5*(p2+phi2[pos(i,j+1,k)]) + Db*0.5*(p3+1.0-phi1[pos(i,j+1,k)]-phi2[pos(i,j+1,k)]) ) * (mu[pos(i,j+1,k)]-muu)*DYin;
		double Hyn = ( 0.5*(p1+phi1[pos(i,j-1,k)]) + Da*0.5*(p2+phi2[pos(i,j-1,k)]) + Db*0.5*(p3+1.0-phi1[pos(i,j-1,k)]-phi2[pos(i,j-1,k)]) ) * (muu-mu[pos(i,j-1,k)])*DYin;
		double Hzp = ( 0.5*(p1+phi1[pos(i,j,k+1)]) + Da*0.5*(p2+phi2[pos(i,j,k+1)]) + Db*0.5*(p3+1.0-phi1[pos(i,j,k+1)]-phi2[pos(i,j,k+1)]) ) * (mu[pos(i,j,k+1)]-muu)*DZin;
		double Hzn = ( 0.5*(p1+phi1[pos(i,j,k-1)]) + Da*0.5*(p2+phi2[pos(i,j,k-1)]) + Db*0.5*(p3+1.0-phi1[pos(i,j,k-1)]-phi2[pos(i,j,k-1)]) ) * (muu-mu[pos(i,j,k-1)])*DZin;

		c_next[ps] = c[ps] + DT*alpha*( (Hxp-Hxn)*DXin + (Hyp-Hyn)*DYin + (Hzp-Hzn)*DZin );
	}
}



__global__ void boundaryX(double *phi1_next, double *phi2_next, double *c_next)
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;

	phi1_next[pos(0, j, k)] = phi1_next[pos(Nx-2, j, k)];
	phi2_next[pos(0, j, k)] = phi2_next[pos(Nx-2, j, k)];
	c_next[pos(0, j, k)] = c_next[pos(Nx-2, j, k)];

	phi1_next[pos(Nx-1, j, k)] = phi1_next[pos(1, j, k)];
	phi2_next[pos(Nx-1, j, k)] = phi2_next[pos(1, j, k)];
	c_next[pos(Nx-1, j, k)] = c_next[pos(1, j, k)];
}

__global__ void boundaryY(double *phi1_next, double *phi2_next, double *c_next )
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;

	phi1_next[pos(i, 0, k)] = phi1_next[pos(i, Ny-2, k)];
	phi2_next[pos(i, 0, k)] = phi2_next[pos(i, Ny-2, k)];
	c_next[pos(i, 0, k)] = c_next[pos(i, Ny-2, k)];

	phi1_next[pos(i, Ny-1, k)] = phi1_next[pos(i, 1, k)];
	phi2_next[pos(i, Ny-1, k)] = phi2_next[pos(i, 1, k)];
	c_next[pos(i, Ny-1, k)] = c_next[pos(i, 1, k)];
}

__global__ void boundaryZ(double *phi1_next, double *phi2_next, double *c_next )
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	phi1_next[pos(i, j, 0)] = phi1_next[pos(i, j, 1)];
	phi2_next[pos(i, j, 0)] = phi2_next[pos(i, j, 1)];
	c_next[pos(i, j, 0)] = c_next[pos(i, j, 1)];

	phi1_next[pos(i, j, devNz-1)] = phi1_next[pos(i, j, devNz-2)];
	phi2_next[pos(i, j, devNz-1)] = phi2_next[pos(i, j, devNz-2)];
	c_next[pos(i, j, devNz-1)] = c_next[pos(i, j, devNz-2)];
}


void enableP2P() // Check the compatibility of P2P communication. If supported, enable P2P.
{
	int peer_access_available = 0;

	for (int i = 0; i < Ngpu-1; i++)
	{
		cudaSetDevice(i);
		peer_access_available = 0;
		cudaDeviceCanAccessPeer(&peer_access_available, i, i + 1);
		printf("> Peer acess from GPU%d to GPU%d: %s \n", i, i + 1, (peer_access_available ? "enabled" : "not support"));
		cudaDeviceEnablePeerAccess(i + 1, 0);

		cudaSetDevice(i + 1);
		peer_access_available = 0;
		cudaDeviceCanAccessPeer(&peer_access_available, i + 1, i);
		printf("> Peer acess from GPU%d to GPU%d: %s \n", i + 1, i, (peer_access_available ? "enabled" : "not support"));
		cudaDeviceEnablePeerAccess(i, 0);
	}

	fflush(stdout);
}

void disableP2P() // Since P2P occupies memory on device, need to disable P2P at the end.
{ 
	for (int i = 0; i < Ngpu-1; i++)
	{
		cudaSetDevice(i);
		cudaDeviceDisablePeerAccess(i + 1);

		cudaSetDevice(i + 1);
		cudaDeviceDisablePeerAccess(i);
	}
}

void isUnifiedAddressing()  //Check the compatibility of UVA. 
{
	cudaDeviceProp prop[Ngpu];

	for (int i = 0; i < Ngpu; i++)
	{
		cudaGetDeviceProperties(&prop[i], i);
		printf("> GPU%d: %s %s unified addressing\n", i, prop[i].name,
			(prop[i].unifiedAddressing ? "support" : "not support"));
	}

	fflush(stdout);
}


void data2File2D(double *data, int step, int flag, char *CrossSection, int xx)
{
	char FileName[50];
	switch (flag)
	{
		case 1: sprintf(FileName, "%-s%s%d%s%d%s", "c_2D_", CrossSection, xx, "_step", step, ".txt"); break;
		case 2: sprintf(FileName, "%-s%s%d%s%d%s", "phi1_2D_", CrossSection, xx, "_step", step, ".txt"); break;
		case 3: sprintf(FileName, "%-s%s%d%s%d%s", "phi2_2D_", CrossSection, xx, "_step", step, ".txt"); break;
	}
	
	FILE * file_pointer=fopen(FileName,"w");

	int Nj = 0, Ni = 0, dnj = 1, dni = 1;
	if ( strcmp(CrossSection,"xz")==0 )
	{
		Nj = Nz;
		Ni = Nx;
	}
	else if ( strcmp(CrossSection,"yz")==0 )
	{
		Nj = Nz;
		Ni = Ny;
	}

	for (int j=0; j<Nj; j=j+dnj)
	{
		for (int i=0; i<Ni; i=i+dni)
		{
			if ( strcmp(CrossSection,"xz")==0 )
				fprintf(file_pointer, "%.4f\t", data[pos(i,xx,j)]);
			if ( strcmp(CrossSection,"yz")==0 )
				fprintf(file_pointer, "%.4f\t", data[pos(xx,i,j)]);
		}
		fprintf(file_pointer, "\n");
	}

	fclose(file_pointer);
}

void save_break_point(double *data, int flag)
{
	string fname;
	switch (flag)
	{
		case 1: fname = "c_break_point.vtk"; break;
		case 2: fname = "phi1_break_point.vtk"; break;
		case 3: fname = "phi2_break_point.vtk"; break;
		case 4: fname = "mu_break_point.vtk"; break;
	}
	ofstream breakPoint(fname.data());

	breakPoint<<"# vtk DataFile Version 3.0"<<endl;
	breakPoint<<"concentration"<<endl;
	breakPoint<<"binary"<<endl;
	breakPoint<<"DATASET STRUCTURED_POINTS"<<endl;
	breakPoint<<"DIMENSIONS "<<Nx<<" "<<Ny<<" "<<Nz<<endl;
	breakPoint<<"ASPECT_RATIO 1 1 1"<<endl;
	breakPoint<<"ORIGIN 0 0 0"<<endl;
	breakPoint<<"POINT_DATA "<<Nx*Ny*Nz<<endl;
	breakPoint<<"SCALARS c double 1"<<endl;
	breakPoint<<"LOOKUP_TABLE default"<<endl;

	for(int k=0; k<Nz; k++)
	{
		for(int j=0; j<Ny; j++)
		{
			for(int i=0; i<Nx; i++)
			{
				breakPoint.write((char *)&(data[pos(i,j,k)]), sizeof(double));
			}
		}
	}

	breakPoint.close();
}

void save_break_point2(int *other)
{
	FILE * file_pointer = fopen("other_break_point.txt", "w");
	fprintf(file_pointer, "%d", other[0]);
	fclose(file_pointer);
}


void load_break_point(double *data, int flag)
{
	string fname, info;

	switch (flag)
	{
		case 1: fname = "c_break_point.vtk"; break;
		case 2: fname = "phi1_break_point.vtk"; break;
		case 3: fname = "phi2_break_point.vtk"; break;
		case 4: fname = "mu_break_point.vtk"; break;
	}

	ifstream breakPoint(fname.data());
	if(!breakPoint)
	{
		cout<<"No input file: "<<fname.data()<<", flag="<<flag<<endl; //<<", breakPoint="<<breakPoint<<
		exit(-1);
	}

	for (int i=0; i<10; i++)
	{
		getline(breakPoint, info);
	}

	for (int k=0; k<Nz; k++)
	{
		for (int j=0; j<Ny; j++)
		{
			for (int i=0; i<Nx; i++)
			{
				breakPoint.read((char *)&(data[pos(i,j,k)]), sizeof(double));
			}
		}
	}
	breakPoint.close();
}

void load_break_point2(int *other)
{
	FILE * file_pointer = fopen("other_break_point.txt", "r");
	if(file_pointer==NULL)
	{
		printf("File not found! other\n");
	}
	else
	{
		fscanf(file_pointer, "%d\t", &other[0]);
	}

	fclose(file_pointer);
}

void c2vtk(double *c, int step)
{
	int sizeX = (Nx-1)/di+1, sizeY = (Ny-1)/dj+1, sizeZ = (Nz-1)/dk+1;

	string fname = "c_step" + std::to_string(step) + ".vtk";
	ofstream cvtk(fname.data());

	cvtk<<"# vtk DataFile Version 3.0"<<endl;
	cvtk<<"concentration"<<endl;
	cvtk<<"binary"<<endl;
	cvtk<<"DATASET STRUCTURED_POINTS"<<endl;
	cvtk<<"DIMENSIONS "<<sizeX<<" "<<sizeY<<" "<<sizeZ<<endl;
	cvtk<<"ASPECT_RATIO 1 1 1"<<endl;
	cvtk<<"ORIGIN 0 0 0"<<endl;
	cvtk<<"POINT_DATA "<<(sizeX)*(sizeY)*(sizeZ)<<endl;
	cvtk<<"SCALARS c float 1"<<endl;
	cvtk<<"LOOKUP_TABLE default"<<endl;

	double average;
	float tmp;
	for(int k=0; k<Nz; k=k+dk)
	{
		for(int j=0; j<Ny; j=j+dj)
		{
			for(int i=0; i<Nx; i=i+di)
			{
				average = 0;
				for (int kz=k-dk/2; kz<k+dk/2; kz++)
				{
					for (int jy=j-dj/2; jy<j+dj/2; jy++)
					{
						for (int ix=i-di/2; ix<i+di/2; ix++)
						{
							average += c[pos(ix,jy,kz)];
						}
					}
				}
				tmp = ReverseFloat(average/(di*dj*dk));
				cvtk.write((char *)&tmp, sizeof(float));
			}
		}
	}

	cvtk.close();

	///////////// close the boundary
	fname = "c0_step" + std::to_string(step) + ".vtk";
	ofstream cvtk0(fname.data());

	cvtk0<<"# vtk DataFile Version 3.0"<<endl;
	cvtk0<<"concentration"<<endl;
	cvtk0<<"binary"<<endl;
	cvtk0<<"DATASET STRUCTURED_POINTS"<<endl;
	cvtk0<<"DIMENSIONS "<<sizeX<<" "<<sizeY<<" "<<sizeZ<<endl;
	cvtk0<<"ASPECT_RATIO 1 1 1"<<endl;
	cvtk0<<"ORIGIN 0 0 0"<<endl;
	cvtk0<<"POINT_DATA "<<(sizeX)*(sizeY)*(sizeZ)<<endl;
	cvtk0<<"SCALARS c float 1"<<endl;
	cvtk0<<"LOOKUP_TABLE default"<<endl;

	for(int k=0; k<Nz; k=k+dk)
	{
		for(int j=0; j<Ny; j=j+dj)
		{
			for(int i=0; i<Nx; i=i+di)
			{
				if(i*j*k==0 || i+di>=Nx-1 || j+dj>=Ny-1 || k+dk>=Nz-1)
				{
					tmp = ReverseFloat(0.5);
					cvtk0.write((char *)&tmp, sizeof(float));
				}
				else
				{
					average = 0;
					for (int kz=k-dk/2; kz<k+dk/2; kz++)
					{
						for (int jy=j-dj/2; jy<j+dj/2; jy++)
						{
							for (int ix=i-di/2; ix<i+di/2; ix++)
							{
								average += c[pos(ix,jy,kz)];
							}
						}
					}
					tmp = ReverseFloat(average/(di*dj*dk));
					cvtk0.write((char *)&tmp, sizeof(float));
				}
				
			}
		}
	}

	cvtk0.close();
}


float ReverseFloat( const float inFloat )
{
	float retVal;
	char *FloatToConvert = ( char* ) & inFloat;
	char *returnFloat = ( char* ) & retVal;

	// swap the bytes into a temporary buffer
	returnFloat[0] = FloatToConvert[3];
	returnFloat[1] = FloatToConvert[2];
	returnFloat[2] = FloatToConvert[1];
	returnFloat[3] = FloatToConvert[0];

	return retVal;
}
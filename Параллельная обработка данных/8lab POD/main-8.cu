#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include "mpi.h"

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

#define _i(i, j, k) ((k + 1) * (n_x + 2) * (n_y + 2) + (j + 1) * (n_x + 2) + i + 1)
#define _ib(i, j, k) (k * nb_x * nb_y + j * nb_x + i)


__global__ void kernalCompute(double *data, double *next, int n_x, int n_y, int n_z, double hx, double hy, double hz) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	int size = n_x * n_y * n_z;

	int i, j, k;

	for(int l = idx; l < size; l += offset)	{
		i = l % n_x;
		j = (l / n_x ) % n_y;
		k = l / (n_x * n_y);
		next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
											(data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
											(data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
												(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
	}
}



__global__ void kernalComputeDiff(double *data, double *next, int n_x, int n_y, int n_z) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	int size = (n_x + 2) * (n_y + 2) * (n_z + 2);
	int i, j, k;

	for (int l = idx; l < size; l += offset)	{
		i = l % (n_x + 2);
		j = (l / (n_x + 2)) % (n_y + 2);
		k = l / ((n_x + 2) * (n_y + 2));
		next[l] = fabs(next[l] - data[l]);
		if (i == 0 || j == 0 || k == 0 || i == (n_x + 1) || j == (n_y + 1) || k == (n_z + 1)) {
			next[l] = 0;
		}
	}
}


__global__ void kernalIJNt_to_gbuff(double *data, double *buff, int n_x, int n_y, int n_z, int layer) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	int size = n_x * n_y;

	int i, j;

	for(int l = idx; l < size; l += offset)	{
		i = l % n_x;
		j = l / n_x;
		buff[i * n_y + j] = data[_i(i, j, layer)];
	}
}

__global__ void kernalIJNt_from_gbuff(double *data, double *buff, int n_x, int n_y, int n_z, int layer) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	int size = n_x * n_y;

	int i, j;

	for(int l = idx; l < size; l += offset)	{
		i = l % n_x;
		j = l / n_x;
		data[_i(i, j, layer)] = buff[i * n_y + j];
	}
}

__global__ void kernalINK_to_gbuff(double *data, double *buff, int n_x, int n_y, int n_z, int layer) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	int size = n_x * n_z;

	int i, k;

	for(int l = idx; l < size; l += offset)	{
		i = l % n_x;
		k = l / n_x;
		buff[i * n_z + k] = data[_i(i, layer, k)];
	}
}

__global__ void kernalINK_from_gbuff(double *data, double *buff, int n_x, int n_y, int n_z, int layer) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	int size = n_x * n_z;

	int i, k;

	for(int l = idx; l < size; l += offset)	{
		i = l % n_x;
		k = l / n_x;
		data[_i(i, layer, k)] = buff[i * n_z + k];
	}
}

__global__ void kernalNJK_to_gbuff(double *data, double *buff, int n_x, int n_y, int n_z, int layer) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	int size = n_y * n_z;

	int j, k;

	for(int l = idx; l < size; l += offset)	{
		j = l % n_y;
		k = l / n_y;
		buff[j * n_z + k] = data[_i(layer, j, k)];
	}
}

__global__ void kernalNJK_from_gbuff(double *data, double *buff, int n_x, int n_y, int n_z, int layer) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	int size = n_y * n_z;

	int j, k;

	for(int l = idx; l < size; l += offset)	{
		j = l % n_y;
		k = l / n_y;
		data[_i(layer, j, k)] = buff[j * n_z + k];
	}
}


int main(int argc, char *argv[]) {
	int ib, jb, kb, nb_x, nb_y, nb_z, n_x, n_y, n_z;
	int i, j, k;
	int id, numproc, proc_name_len, gpu_count;
	char proc_name[MPI_MAX_PROCESSOR_NAME];
	char file_path[256] = "mpi.out";

	double lx, ly, lz, hx, hy, hz, bc_down, bc_up, bc_left, bc_right, bc_back, bc_front;
	double *data, *temp, *buffs, *buffr;
	double u_0, max_dif_l, max_dif, eps;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Get_processor_name(proc_name, &proc_name_len);

	CSC(cudaGetDeviceCount(&gpu_count));
	CSC(cudaSetDevice(id % gpu_count));

	MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {
		scanf("%d%d%d",&nb_x,&nb_y,&nb_z );
		scanf("%d%d%d",&n_x,&n_y,&n_z );
		scanf("%s", file_path);
		scanf("%lf",&eps);
		scanf("%lf%lf%lf",&lx,&ly,&lz);
		scanf("%lf%lf%lf%lf%lf%lf",&bc_front,&bc_back, &bc_left, &bc_right, &bc_down, &bc_up);
		scanf("%lf",&u_0);
	}


	MPI_Bcast(&n_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n_z, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nb_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nb_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nb_z, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);



	ib = id % nb_x;
	jb = (id / nb_x) % nb_y;
	kb = id / (nb_x * nb_y);

	hx = lx / (double)(n_x * nb_x);
	hy = ly / (double)(n_y * nb_y);
	hz = lz / (double)(n_z * nb_z);

	data = (double *)malloc(sizeof(double) * (n_x + 2) * (n_y + 2) * (n_z + 2));

	double *dev_data, *dev_next, *dev_buff;
	CSC(cudaMalloc(&dev_data, sizeof(double) * (n_x + 2) * (n_y + 2) * (n_z + 2)));
	CSC(cudaMalloc(&dev_next, sizeof(double) * (n_x + 2) * (n_y + 2) * (n_z + 2)));

	thrust::device_ptr<double> data_tr;

	int n_max1 = std::max(n_x, n_y);
	int n_max2 = std::max(n_z, n_x + n_y - n_max1);

	buffs = (double *)malloc(sizeof(double) * (n_max1 + 2) * (n_max2 + 2));
	buffr = (double *)malloc(sizeof(double) * (n_max1 + 2) * (n_max2 + 2));

	CSC(cudaMalloc(&dev_buff, sizeof(double) * (n_max1 + 2) * (n_max2 + 2)));


	if (id == 0) {
		fprintf(stderr, "%f %f %f %f %f %f %f %f %f %f %f %f %f\n", lx, ly, lz, hx, hy, hz, bc_front, bc_back, bc_left, bc_right, bc_down, bc_up, u_0);
		fprintf(stderr, "%f %d %d %d %d %d %d\n", eps, nb_x, nb_y, nb_z, n_x, n_y, n_z);
		fflush(stderr);
	}

	for(i = 0; i < n_x; i++)					// Инициализация блока
		for(j = 0; j < n_y; j++)
			for(k = 0; k < n_z; k++)
				data[_i(i, j, k)] = u_0;

	for(i = -1; i <= n_x; i++) {
		for(j = -1; j <= n_y; j++) {
			data[_i(i, j, -1)] = bc_front;
			data[_i(i, j, n_z)] = bc_back;
		}
	}
	for(i = -1; i <= n_x; i++) {
		for(k = -1; k <= n_z; k++) {
			data[_i(i, -1, k)] = bc_down;
			data[_i(i, n_y, k)] = bc_up;
		}
	}
	for(j = -1; j <= n_y; j++) {
		for(k = -1; k <= n_z; k++) {
			data[_i(-1, j, k)] = bc_left;
			data[_i(n_x, j, k)] = bc_right;
		}
	}

	CSC(cudaMemcpy(dev_data, data, sizeof(double) * (n_x + 2) * (n_y + 2) * (n_z + 2), cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(dev_next, data, sizeof(double) * (n_x + 2) * (n_y + 2) * (n_z + 2), cudaMemcpyHostToDevice));


	int it = 0;
	while(1) {
		MPI_Barrier(MPI_COMM_WORLD);
		it++;
		for(int dir = 0; dir < 2; dir++) {
			if ((ib + dir) & 1) {
				if (ib > 0) {

					kernalNJK_to_gbuff<<<64, 64>>>(dev_data, dev_buff, n_x, n_y, n_z, 0);
					CSC(cudaMemcpy(buffs, dev_buff, sizeof(double) * n_x * n_y, cudaMemcpyDeviceToHost));

					MPI_Sendrecv(buffs, n_y * n_z, MPI_DOUBLE, _ib(ib - 1, jb, kb), id,
								buffr, n_y * n_z, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);

					CSC(cudaMemcpy(dev_buff, buffr, sizeof(double) * n_x * n_y, cudaMemcpyHostToDevice));
					kernalNJK_from_gbuff<<<64, 64>>>(dev_data, dev_buff, n_x, n_y, n_z, -1);

				}
			} else {
				if (ib + 1 < nb_x) {

					kernalNJK_to_gbuff<<<64, 64>>>(dev_data, dev_buff, n_x, n_y, n_z, n_x - 1);
					CSC(cudaMemcpy(buffs, dev_buff, sizeof(double) * n_x * n_y, cudaMemcpyDeviceToHost));

					MPI_Sendrecv(buffs, n_y * n_z, MPI_DOUBLE, _ib(ib + 1, jb, kb), id,
								buffr, n_y * n_z, MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);

					CSC(cudaMemcpy(dev_buff, buffr, sizeof(double) * n_x * n_y, cudaMemcpyHostToDevice));
					kernalNJK_from_gbuff<<<64, 64>>>(dev_data, dev_buff, n_x, n_y, n_z, n_x);

				}
			}
		}
		// Y-axies (j)
		for(int dir = 0; dir < 2; dir++) {
			if ((jb + dir)& 1) {
				if (jb > 0) {
					kernalINK_to_gbuff<<<64, 64>>>(dev_data, dev_buff, n_x, n_y, n_z, 0);
					CSC(cudaMemcpy(buffs, dev_buff, sizeof(double) * n_x * n_y, cudaMemcpyDeviceToHost));

					MPI_Sendrecv(buffs, n_x * n_z, MPI_DOUBLE, _ib(ib, jb - 1, kb), id,
								buffr, n_x * n_z, MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);

					CSC(cudaMemcpy(dev_buff, buffr, sizeof(double) * n_x * n_y, cudaMemcpyHostToDevice));
					kernalINK_from_gbuff<<<64, 64>>>(dev_data, dev_buff, n_x, n_y, n_z, -1);
				}
			} else {
				if (jb + 1 < nb_y) {
					kernalINK_to_gbuff<<<64, 64>>>(dev_data, dev_buff, n_x, n_y, n_z, n_y - 1);
					CSC(cudaMemcpy(buffs, dev_buff, sizeof(double) * n_x * n_y, cudaMemcpyDeviceToHost));

					MPI_Sendrecv(buffs, n_x * n_z, MPI_DOUBLE, _ib(ib, jb + 1, kb), id,
								buffr, n_x * n_z, MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);

					CSC(cudaMemcpy(dev_buff, buffr, sizeof(double) * n_x * n_y, cudaMemcpyHostToDevice));
					kernalINK_from_gbuff<<<64, 64>>>(dev_data, dev_buff, n_x, n_y, n_z, n_y);
				}
			}
		}

		// Z-axies (k)
		for(int dir = 0; dir < 2; dir++) {
			if ((kb + dir)& 1) {
				if (kb > 0) {
					kernalIJNt_to_gbuff<<<64, 64>>>(dev_data, dev_buff, n_x, n_y, n_z, 0);
					CSC(cudaMemcpy(buffs, dev_buff, sizeof(double) * n_x * n_y, cudaMemcpyDeviceToHost));

					MPI_Sendrecv(buffs, n_x * n_y, MPI_DOUBLE, _ib(ib, jb, kb - 1), id,
								buffr, n_x * n_y, MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);

					CSC(cudaMemcpy(dev_buff, buffr, sizeof(double) * n_x * n_y, cudaMemcpyHostToDevice));
					kernalIJNt_from_gbuff<<<64, 64>>>(dev_data, dev_buff, n_x, n_y, n_z, -1);
				}
			} else {
				if (kb + 1 < nb_z) {
					kernalIJNt_to_gbuff<<<64, 64>>>(dev_data, dev_buff, n_x, n_y, n_z, n_z - 1);
					CSC(cudaMemcpy(buffs, dev_buff, sizeof(double) * n_x * n_y, cudaMemcpyDeviceToHost));

					MPI_Sendrecv(buffs, n_x * n_y, MPI_DOUBLE, _ib(ib, jb, kb + 1), id,
								buffr, n_x * n_y, MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);

					CSC(cudaMemcpy(dev_buff, buffr, sizeof(double) * n_x * n_y, cudaMemcpyHostToDevice));
					kernalIJNt_from_gbuff<<<64, 64>>>(dev_data, dev_buff, n_x, n_y, n_z, n_z);
				}
			}
		}


		MPI_Barrier(MPI_COMM_WORLD);

		kernalCompute<<<64, 64>>>(dev_data, dev_next, n_x, n_y, n_z, hx, hy, hz);
		CSC(cudaGetLastError());

		temp = dev_next;
		dev_next = dev_data;
		dev_data = temp;

		max_dif_l = 0;

		kernalComputeDiff<<<64, 64>>>(dev_data, dev_next, n_x, n_y, n_z);
		CSC(cudaGetLastError());

		data_tr = thrust::device_pointer_cast(dev_next);

		max_dif_l = *(thrust::max_element(data_tr, data_tr + (n_x + 2) * (n_y + 2) * (n_z + 2)));

		CSC(cudaMemcpy(dev_next, dev_data, sizeof(double) * (n_x + 2) * (n_y + 2) * (n_z + 2), cudaMemcpyDeviceToDevice));

		MPI_Allreduce(&max_dif_l, &max_dif, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		if (max_dif < eps) {
			fprintf(stderr, "e %f, max_dif_l %f, max_dif%f, it%d\n", eps, max_dif_l, max_dif, it);
			fflush(stderr);
			break;
		}
	}

	CSC(cudaMemcpy(data, dev_data, sizeof(double) * (n_x + 2) * (n_y + 2) * (n_z + 2), cudaMemcpyDeviceToHost));

	CSC(cudaFree(dev_data));
	CSC(cudaFree(dev_next));
	CSC(cudaFree(dev_buff));

	int print_size = 15;
	char* str = (char *)malloc(sizeof(char) * (n_x * n_y * n_z * print_size + 1));
	memset(str, ' ', sizeof(char) * (n_x * n_y * n_z * print_size));

	for(k = 0; k < n_z; k++) {
		for(j = 0; j < n_y; j++) {
			for(i = 0; i < n_x; i++)
				sprintf(str + (i + j * n_x + k * n_x * n_y) * print_size, "%.7e ", data[_i(i, j, k)]);
			if (ib + 1 == nb_x)
				str[((j + 1) * n_x + k * n_x * n_y) * print_size - 1] = '\n';
		}
	}

	for (i = 0; i < n_x * n_y * n_z * print_size; i++)
		if (str[i] == '\0')
			str[i] = ' ';
	str[n_x * n_y * n_z * print_size] = '\0';

	MPI_Datatype gridT, blockT;
	MPI_Type_vector(n_y, n_x * print_size, n_x * print_size * nb_x, MPI_CHAR, &blockT);
	MPI_Type_commit(&blockT);

	MPI_Type_create_hvector(n_z, 1, n_x * nb_x * n_y * nb_y * print_size, blockT, &gridT);
	MPI_Type_commit(&gridT);

	MPI_File fp;
	MPI_File_delete(file_path, MPI_INFO_NULL);
	MPI_File_open(MPI_COMM_WORLD, file_path, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);

	MPI_File_set_view(fp, sizeof(char) * (n_x * ib + n_x * nb_x * n_y * jb + n_x * nb_x * n_y * nb_y * n_z * kb) * print_size, MPI_CHAR, gridT, "native", MPI_INFO_NULL);
	MPI_File_write_all(fp, str, n_x * n_y * n_z * print_size, MPI_CHAR, &status);

	MPI_File_close(&fp);

	free(data);
	free(str);
	free(buffr);
	free(buffs);

	MPI_Finalize();
	return 0;
}

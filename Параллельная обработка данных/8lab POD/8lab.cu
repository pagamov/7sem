#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

using namespace std;

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)

#define _i(i, j, k) ((k + 1) * (nx + 2) * (ny + 2) + (j + 1) * (nx + 2) + i + 1)
#define _iD(i, j, k) ((k + 1) * (dim[0] + 2) * (dim[1] + 2) + (j + 1) * (dim[0] + 2) + i + 1)
#define _ib(i, j, k) (k * box[0] * box[1] + j * box[0] + i)

__global__ void kernel_setup(double * arr, int nx, int ny, int nz, double u_0) {
    for (int k = blockDim.z * blockIdx.z + threadIdx.z; k < nz; k += blockDim.z * gridDim.z)
    	for (int j = blockDim.y * blockIdx.y + threadIdx.y; j < ny; j += blockDim.x * gridDim.x)
    		for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nx; i += blockDim.x * gridDim.x)
    			arr[_i(i, j, k)] = u_0;
}

__global__ void kernel_LR_setup(double * arr, int nx, int ny, int nz, double bc, int x_ind) {
    for (int k = blockDim.y * blockIdx.y + threadIdx.y; k < nz; k += blockDim.y * gridDim.y)
    	for (int j = blockDim.x * blockIdx.x + threadIdx.x; j < ny; j += blockDim.x * gridDim.x)
    		arr[_i(x_ind, j, k)] = bc;
}

__global__ void kernel_FB_setup(double * arr, int nx, int ny, int nz, double bc, int y_ind) {
    for (int k = blockDim.y * blockIdx.y + threadIdx.y; k < nz; k += blockDim.y * gridDim.y)
    	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nx; i += blockDim.x * gridDim.x)
    		arr[_i(i, y_ind, k)] = bc;
}

__global__ void kernel_DU_setup(double * arr, int nx, int ny, double bc, int z_ind) {
    for (int j = blockDim.y * blockIdx.y + threadIdx.y; j < ny; j += blockDim.y * gridDim.y)
    	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nx; i += blockDim.x * gridDim.x)
    		arr[_i(i, j, z_ind)] = bc;
}

__global__ void kernel_send_LR(double * buf, double* arr, int nx, int ny, int nz, int x_ind) {
    for (int k = blockDim.y * blockIdx.y + threadIdx.y; k < nz; k += blockDim.y * gridDim.y)
    	for (int j = blockDim.x * blockIdx.x + threadIdx.x; j < ny; j += blockDim.x * gridDim.x)
    		buf[k * ny + j] = arr[_i(x_ind, j, k)];
}

__global__ void kernel_recive_LR(double * buf, double * arr, int nx, int ny, int nz, int x_ind) {
    for (int k = blockDim.y * blockIdx.y + threadIdx.y; k < nz; k += blockDim.y * gridDim.y)
    	for (int j = blockDim.x * blockIdx.x + threadIdx.x; j < ny; j += blockDim.x * gridDim.x)
    		arr[_i(x_ind, j, k)] = buf[k * ny + j];
}

__global__ void kernel_send_FB(double * buf, double * arr, int nx, int ny, int nz, int y_ind) {
    for (int k = blockDim.y * blockIdx.y + threadIdx.y; k < nz; k += blockDim.y * gridDim.y)
    	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nx; i += blockDim.x * gridDim.x)
    		buf[k * nx + i] = arr[_i(i, y_ind, k)];
}

__global__ void kernel_recive_FB(double * buf, double * arr, int nx, int ny, int nz, int y_ind) {
    for (int k = blockDim.y * blockIdx.y + threadIdx.y; k < nz; k += blockDim.y * gridDim.y)
    	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nx; i += blockDim.x * gridDim.x)
    		arr[_i(i, y_ind, k)] = buf[k * nx + i];
}

__global__ void kernel_send_DU(double * buf, double * arr, int nx, int ny, int z_ind) {
    for (int j = blockDim.y * blockIdx.y + threadIdx.y; j < ny; j += blockDim.y * gridDim.y)
    	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nx; i += blockDim.x * gridDim.x)
    		buf[j * nx + i] = arr[_i(i, j, z_ind)];
}

__global__ void kernel_recive_DU(double * buf, double * arr, int nx, int ny, int z_ind) {
    for (int j = blockDim.y * blockIdx.y + threadIdx.y; j < ny; j += blockDim.y * gridDim.y)
    	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nx; i += blockDim.x * gridDim.x)
    		arr[_i(i, j, z_ind)] = buf[j * nx + i];
}

__global__ void kernel_compute(double * arr, double * next, int nx, int ny, int nz, double hx, double hy, double hz) {
    for (int k = blockDim.z * blockIdx.z + threadIdx.z; k < nz; k += blockDim.z * gridDim.z) {
    	for (int j = blockDim.y * blockIdx.y + threadIdx.y; j < ny; j += blockDim.y * gridDim.y) {
    		for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nx; i += blockDim.x * gridDim.x) {
    			next[_i(i, j, k)] = 0.5 * ((arr[_i(i + 1, j, k)] + arr[_i(i - 1, j, k)]) / (hx * hx) +
						(arr[_i(i, j + 1, k)] + arr[_i(i, j - 1, k)]) / (hy * hy) +
						(arr[_i(i, j, k + 1)] + arr[_i(i, j, k - 1)]) / (hz * hz)) /
						(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
    		}
    	}
    }
}

__global__ void kernel_diff(double * arr, double * next, int nx, int ny, int nz) {
    for (int k = blockDim.z * blockIdx.z + threadIdx.z - 1; k <= nz; k += blockDim.z * gridDim.z)
    	for (int j = blockDim.y * blockIdx.y + threadIdx.y - 1; j <= ny; j += blockDim.y * gridDim.y)
    		for (int i = blockDim.x * blockIdx.x + threadIdx.x - 1; i <= nx; i += blockDim.x * gridDim.x)
    			arr[_i(i, j, k)] = ((i != -1) && (j != -1) && (k != -1) && (i != nx) && (j != ny) && (k != nz)) * fabs(next[_i(i, j, k)] - arr[_i(i, j, k)]);
}


int main(int argc, char* argv[]) {
    int id, numproc;
	int ib, jb, kb;
    int box[3];
    int dim[3];
    char filename[100];
    double l[3];
	double hx, hy, hz;
    double u[6], u_0;
	double eps, diff;
	double *temp, *buff;
    bool f = true;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	int deviceCount;
	CSC(cudaGetDeviceCount(&deviceCount));
	CSC(cudaSetDevice(id % deviceCount));

	if (id == 0) {
        cin >> box[0] >> box[1] >> box[2];
        cin >> dim[0] >> dim[1] >> dim[2];
		scanf("%s", filename);
		cin >> eps;
        cin >> l[0] >> l[1] >> l[2];
        cin >> u[4] >> u[5] >> u[0] >> u[1] >> u[2] >> u[3] >> u_0;
	}

    MPI_Bcast(dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(box, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(l, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(filename, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

    ib = id % (box[0] * box[1]) % box[0];
    jb = id % (box[0] * box[1]) / box[0];
	kb = id / (box[0] * box[1]);

	hx = l[0] / (dim[0] * box[0]);
	hy = l[1] / (dim[1] * box[1]);
	hz = l[2] / (dim[2] * box[2]);

	double * dev_data;
	double * dev_next;
	CSC(cudaMalloc(&dev_data, sizeof(double) * (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2)));
	CSC(cudaMalloc(&dev_next, sizeof(double) * (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2)));

	buff = (double *)malloc(sizeof(double) * max(dim[0], dim[1]) * max(dim[1], dim[2]));
	int buffer_size;
	MPI_Pack_size(max(dim[0], dim[1]) * max(dim[1], dim[2]), MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
	buffer_size = 6 * (buffer_size + MPI_BSEND_OVERHEAD);
	double * buffer = (double*)malloc(buffer_size);
	MPI_Buffer_attach(buffer, buffer_size);
    double * allgbuff = (double *)malloc(sizeof(double) * box[0] * box[1] * box[2]);
	double * dev_buff;
	CSC(cudaMalloc(&dev_buff, sizeof(double) * max(dim[0], dim[1]) * max(dim[1], dim[2])));

	kernel_setup<<<dim3(8, 8, 8), dim3(32, 4, 4)>>>(dev_data, dim[0], dim[1], dim[2], u_0);
	CSC(cudaGetLastError());

	while (f) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (ib + 1 < box[0]) {
			kernel_send_LR<<<dim3(32,32), dim3(32,32)>>>(dev_buff, dev_data, dim[0], dim[1], dim[2], dim[0]-1);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * dim[1] * dim[2], cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, dim[1] * dim[2], MPI_DOUBLE, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
		}

		if (jb + 1 < box[1]) {
			kernel_send_FB<<<dim3(32,32), dim3(32,32)>>>(dev_buff, dev_data, dim[0], dim[1], dim[2], dim[1]-1);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * dim[0] * dim[2], cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, dim[0] * dim[2], MPI_DOUBLE, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
		}

		if (kb + 1 < box[2]) {
			kernel_send_DU<<<dim3(32,32), dim3(32,32)>>>(dev_buff, dev_data, dim[0], dim[1], dim[2]-1);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * dim[0] * dim[1], cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, dim[0] * dim[1], MPI_DOUBLE, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
		}

		if (ib > 0) {
			kernel_send_LR<<<dim3(32,32), dim3(32,32)>>>(dev_buff, dev_data, dim[0], dim[1], dim[2], 0);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * dim[1] * dim[2], cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, dim[1] * dim[2], MPI_DOUBLE, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
		}

		if (jb > 0) {
			kernel_send_FB<<<dim3(32,32), dim3(32,32)>>>(dev_buff, dev_data, dim[0], dim[1], dim[2], 0);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * dim[0] * dim[2], cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, dim[0] * dim[2], MPI_DOUBLE, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
		}

		if (kb > 0) {
			kernel_send_DU<<<dim3(32,32), dim3(32,32)>>>(dev_buff, dev_data, dim[0], dim[1], 0);
			CSC(cudaGetLastError());
			CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * dim[0] * dim[1], cudaMemcpyDeviceToHost));
			MPI_Bsend(buff, dim[0] * dim[1], MPI_DOUBLE, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
		}
        
        MPI_Barrier(MPI_COMM_WORLD);

		if (ib > 0) {
			MPI_Recv(buff, dim[1] * dim[2], MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * dim[1] * dim[2], cudaMemcpyHostToDevice));
			kernel_recive_LR<<<dim3(32,32), dim3(32,32)>>>(dev_buff, dev_data, dim[0], dim[1], dim[2], -1);
			CSC(cudaGetLastError());
		} else {
			kernel_LR_setup<<<dim3(32,32), dim3(32,32)>>>(dev_data, dim[0], dim[1], dim[2], u[0], -1);
			CSC(cudaGetLastError());
		}

		if (jb > 0) {
			MPI_Recv(buff, dim[0] * dim[2], MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * dim[0] * dim[2], cudaMemcpyHostToDevice));
			kernel_recive_FB<<<dim3(32,32), dim3(32,32)>>>(dev_buff, dev_data, dim[0], dim[1], dim[2], -1);
			CSC(cudaGetLastError());
		} else {
			kernel_FB_setup<<<dim3(32,32), dim3(32,32)>>>(dev_data, dim[0], dim[1], dim[2], u[2], -1);
			CSC(cudaGetLastError());
		}

		if (kb > 0) {
			MPI_Recv(buff, dim[0] * dim[1], MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * dim[0] * dim[1], cudaMemcpyHostToDevice));
			kernel_recive_DU<<<dim3(32,32), dim3(32,32)>>>(dev_buff, dev_data, dim[0], dim[1], -1);
			CSC(cudaGetLastError());
		} else {
			kernel_DU_setup<<<dim3(32,32), dim3(32,32)>>>(dev_data, dim[0], dim[1], u[4], -1);
			CSC(cudaGetLastError());
		}

		if (ib + 1 < box[0]) {
			MPI_Recv(buff, dim[1] * dim[2], MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * dim[1] * dim[2], cudaMemcpyHostToDevice));
			kernel_recive_LR<<<dim3(32,32), dim3(32,32)>>>(dev_buff, dev_data, dim[0], dim[1], dim[2], dim[0]);
			CSC(cudaGetLastError());
		} else {
			kernel_LR_setup<<<dim3(32,32), dim3(32,32)>>>(dev_data, dim[0], dim[1], dim[2], u[1], dim[0]);
			CSC(cudaGetLastError());
		}

		if (jb + 1 < box[1]) {
			MPI_Recv(buff, dim[0] * dim[2], MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * dim[0] * dim[2], cudaMemcpyHostToDevice));
			kernel_recive_FB<<<dim3(32,32), dim3(32,32)>>>(dev_buff, dev_data, dim[0], dim[1], dim[2], dim[1]);
			CSC(cudaGetLastError());
		} else {
			kernel_FB_setup<<<dim3(32,32), dim3(32,32)>>>(dev_data, dim[0], dim[1], dim[2], u[3], dim[1]);
			CSC(cudaGetLastError());
		}

		if (kb + 1 < box[2]) {
			MPI_Recv(buff, dim[0] * dim[1], MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * dim[0] * dim[1], cudaMemcpyHostToDevice));
			kernel_recive_DU<<<dim3(32,32), dim3(32,32)>>>(dev_buff, dev_data, dim[0], dim[1], dim[2]);
			CSC(cudaGetLastError());
		} else {
			kernel_DU_setup<<<dim3(32,32), dim3(32,32)>>>(dev_data, dim[0], dim[1], u[5], dim[2]);
			CSC(cudaGetLastError());
		}

		MPI_Barrier(MPI_COMM_WORLD);
		diff = 0.0;

		kernel_compute<<<dim3(8, 8, 8), dim3(32, 4, 4)>>>(dev_data, dev_next, dim[0], dim[1], dim[2], hx, hy, hz);
		CSC(cudaGetLastError());

		kernel_diff<<<dim3(8, 8, 8), dim3(32, 4, 4)>>>(dev_data, dev_next, dim[0], dim[1], dim[2]);
		CSC(cudaGetLastError());

		thrust::device_ptr<double> diffs = thrust::device_pointer_cast(dev_data);
        thrust::device_ptr<double> max_eps = thrust::max_element(diffs, diffs + (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2));
        diff = *max_eps;

        MPI_Allgather(&diff, 1, MPI_DOUBLE, allgbuff, 1, MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
        f = false;
        for (int i = 0; i < box[0] * box[1] * box[2]; i++)
            if (allgbuff[i] > eps)
                f = true;

		temp = dev_next;
		dev_next = dev_data;
		dev_data = temp;
	}

	double * data = (double *)malloc(sizeof(double) * (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2));
	cudaMemcpy(data, dev_data, sizeof(double) * (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2), cudaMemcpyDeviceToHost);

	char * bf = (char *)malloc(sizeof(char) * dim[0] * dim[1] * dim[2] * 14);
	memset(bf, ' ', sizeof(char) * dim[0] * dim[1] * dim[2] * 14);

	for (int k = 0; k < dim[2]; k++)
		for (int j = 0; j < dim[1]; j++)
			for (int i = 0; i < dim[0]; i++)
				sprintf(bf + (k * dim[0] * dim[1] + j * dim[0] + i) * 14, "%.6e", data[_iD(i, j, k)]);

	for (int i = 0; i < dim[0] * dim[1] * dim[2] * 14; i++)
		if (bf[i] == '\0')
			bf[i] = ' ';

	MPI_File fp;
	MPI_Datatype filetype;
	int sizes[] = {dim[2] * box[2], dim[1] * box[1], dim[0] * box[0] * 14};
	int subsizes[] = {dim[2], dim[1], dim[0] * 14};
	int starts[] = {id / (box[0] * box[1]) * dim[2], id % (box[0] * box[1]) / box[0] * dim[1], id % (box[0] * box[1]) % box[0] * dim[0] * 14};
	MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_CHAR, &filetype);
	MPI_Type_commit(&filetype);

	MPI_File_delete(filename, MPI_INFO_NULL);
	MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
	MPI_File_set_view(fp, 0, MPI_CHAR, filetype, "native", MPI_INFO_NULL);
	MPI_File_write_all(fp, bf, dim[0] * dim[1] * dim[2] * 14, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&fp);

	MPI_Type_free(&filetype);
	MPI_Finalize();
    CSC(cudaFree(dev_data));
	CSC(cudaFree(dev_next));
	CSC(cudaFree(dev_buff));
    // free(allgbuff);
	free(data);
	free(buff);
	free(bf);
	return 0;
}

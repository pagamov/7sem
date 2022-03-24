#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

using namespace std;

#define _i(i, j, k) (((k) + 1) * (dim[0] + 2) * (dim[1] + 2) + ((j) + 1) * (dim[0] + 2) + (i) + 1)
#define _ib(i, j, k) ((k) * box[0] * box[1] + (j) * box[0] + (i))

int main(int argc, char* argv[]) {
	int id, numproc;
	int ib, jb, kb;
	int box[3];
	int dim[3];
	string filename;
	double l[3];
	double hx, hy, hz;
	double u[6];
	bool f = true;

	double u_0;
	double eps, diff;
	double * data, * temp, * next, * buff;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {
		cin >> box[0] >> box[1] >> box[2];
		cin >> dim[0] >> dim[1] >> dim[2];
		cin >> filename;
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

	kb = id / (box[0] * box[1]);
	jb = id % (box[0] * box[1]) / box[0];
	ib = id % (box[0] * box[1]) % box[0];

	hx = l[0] / (dim[0] * box[0]);
	hy = l[1] / (dim[1] * box[1]);
	hz = l[2] / (dim[2] * box[2]);

	double h2x = 1.0 / (hx * hx), h2y = 1.0 / (hy * hy), h2z = 1.0 / (hz * hz);

	data = (double *)malloc(sizeof(double) * (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2));
	next = (double *)malloc(sizeof(double) * (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2));

	buff = (double *)malloc(sizeof(double) * max(dim[0], dim[1]) * max(dim[1], dim[2]));
	int buffer_size;
	MPI_Pack_size(max(dim[0], dim[1]) * max(dim[1], dim[2]), MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
	buffer_size = 6 * (buffer_size + MPI_BSEND_OVERHEAD);
	double* buffer = (double *)malloc(buffer_size);
	MPI_Buffer_attach(buffer, buffer_size);

	double * allgbuff = (double *)malloc(sizeof(double) * box[0] * box[1] * box[2]);

	for (int i = 0; i < dim[0]; i++)
		for (int j = 0; j < dim[1]; j++)
			for (int k = 0; k < dim[2]; k++)
				data[_i(i, j, k)] = u_0;

	for (int j = 0; j < dim[1]; j++) {
		for (int k = 0; k < dim[2]; k++) {
			data[_i(-1, j, k)] = u[0];
			next[_i(-1, j, k)] = u[0];
		}
	}

	for (int i = 0; i < dim[0]; i++) {
		for (int k = 0; k < dim[2]; k++) {
			data[_i(i, -1, k)] = u[2];
			next[_i(i, -1, k)] = u[2];
		}
	}

	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			data[_i(i, j, -1)] = u[4];
			next[_i(i, j, -1)] = u[4];
		}
	}

	for (int j = 0; j < dim[1]; j++) {
		for (int k = 0; k < dim[2]; k++) {
			data[_i(dim[0], j, k)] = u[1];
			next[_i(dim[0], j, k)] = u[1];
		}
	}

	for (int i = 0; i < dim[0]; i++) {
		for (int k = 0; k < dim[2]; k++) {
			data[_i(i, dim[1], k)] = u[3];
			next[_i(i, dim[1], k)] = u[3];
		}
	}

	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			data[_i(i, j, dim[2])] = u[5];
			next[_i(i, j, dim[2])] = u[5];
		}
	}

	while (f) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (ib + 1 < box[0]) {
			for (int j = 0; j < dim[1]; j++)
				for (int k = 0; k < dim[2]; k++)
					buff[j * dim[2] + k] = data[_i(dim[0] - 1, j, k)];
			MPI_Bsend(buff, dim[1] * dim[2], MPI_DOUBLE, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
		}

		if (jb + 1 < box[1]) {
			for (int i = 0; i < dim[0]; i++)
				for (int k = 0; k < dim[2]; k++)
					buff[i * dim[2] + k] = data[_i(i, dim[1] - 1, k)];
			MPI_Bsend(buff, dim[0] * dim[2], MPI_DOUBLE, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
		}

		if (kb + 1 < box[2]) {
			for (int i = 0; i < dim[0]; i++)
				for (int j = 0; j < dim[1]; j++)
					buff[i * dim[1] + j] = data[_i(i, j, dim[2] - 1)];
			MPI_Bsend(buff, dim[0] * dim[1], MPI_DOUBLE, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
		}

		if (ib > 0) {
			for (int j = 0; j < dim[1]; j++)
				for (int k = 0; k < dim[2]; k++)
					buff[j * dim[2] + k] = data[_i(0, j, k)];
			MPI_Bsend(buff, dim[1] * dim[2], MPI_DOUBLE, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
		}

		if (jb > 0) {
			for (int i = 0; i < dim[0]; i++)
				for (int k = 0; k < dim[2]; k++)
					buff[i * dim[2] + k] = data[_i(i, 0, k)];
			MPI_Bsend(buff, dim[0] * dim[2], MPI_DOUBLE, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
		}

		if (kb > 0) {
			for (int i = 0; i < dim[0]; i++)
				for (int j = 0; j < dim[1]; j++)
					buff[i * dim[1] + j] = data[_i(i, j, 0)];
			MPI_Bsend(buff, dim[0] * dim[1], MPI_DOUBLE, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
		}

		if (ib > 0) {
			MPI_Recv(buff, dim[1] * dim[2], MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
			for (int j = 0; j < dim[1]; j++)
				for (int k = 0; k < dim[2]; k++)
					data[_i(-1, j, k)] = buff[j * dim[2] + k];
		}

		if (jb > 0) {
			MPI_Recv(buff, dim[0] * dim[2], MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
			for (int i = 0; i < dim[0]; i++)
				for (int k = 0; k < dim[2]; k++)
					data[_i(i, -1, k)] = buff[i * dim[2] + k];
		}

		if (kb > 0) {
			MPI_Recv(buff, dim[0] * dim[1], MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
			for (int i = 0; i < dim[0]; i++)
				for (int j = 0; j < dim[1]; j++)
					data[_i(i, j, -1)] = buff[i * dim[1] + j];
		}

		if (ib + 1 < box[0]) {
			MPI_Recv(buff, dim[1] * dim[2], MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
			for (int j = 0; j < dim[1]; j++)
				for (int k = 0; k < dim[2]; k++)
					data[_i(dim[0], j, k)] = buff[j * dim[2] + k];
		}

		if (jb + 1 < box[1]) {
			MPI_Recv(buff, dim[0] * dim[2], MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
			for (int i = 0; i < dim[0]; i++)
				for (int k = 0; k < dim[2]; k++)
					data[_i(i, dim[1], k)] = buff[i * dim[2] + k];
		}

		if (kb + 1 < box[2]) {
			MPI_Recv(buff, dim[0] * dim[1], MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
			for (int i = 0; i < dim[0]; i++)
				for (int j = 0; j < dim[1]; j++)
					data[_i(i, j, dim[2])] = buff[i * dim[1] + j];
		}

		MPI_Barrier(MPI_COMM_WORLD);
		diff = 0.0;
		for (int i = 0; i < dim[0]; i++) {
			for (int j = 0; j < dim[1]; j++) {
				for (int k = 0; k < dim[2]; k++) {
					next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) * h2x +
											   (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) * h2y +
										       (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) * h2z) /
											   (h2x + h2y + h2z);

					diff = fmax(diff, fabs(next[_i(i, j, k)] - data[_i(i, j, k)]));
				}
			}
		}

		MPI_Allgather(&diff, 1, MPI_DOUBLE, allgbuff, 1, MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
        f = false;
        for (int i = 0; i < box[0] * box[1] * box[2]; i++)
            if (allgbuff[i] > eps)
                f = true;
		temp = next;
		next = data;
		data = temp;
	}

	if (id != 0) {
		for (int k = 0; k < dim[2]; k++) {
			for (int j = 0; j < dim[1]; j++) {
				for (int i = 0; i < dim[0]; i++)
					buff[i] = data[_i(i, j, k)];
				MPI_Send(buff, dim[0], MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
			}
		}
	} else {
		FILE * f = fopen(filename.c_str(), "w");
		for (int kb = 0; kb < box[2]; kb++) {
			for (int k = 0; k < dim[2]; k++) {
				for (int jb = 0; jb < box[1]; jb++) {
					for (int j = 0; j < dim[1]; j++) {
						for (int ib = 0; ib < box[0]; ib++) {
							if (_ib(ib, jb, kb) == 0)
								for (int i = 0; i < dim[0]; i++)
									buff[i] = data[_i(i, j, k)];
							else
								MPI_Recv(buff, dim[0], MPI_DOUBLE, _ib(ib, jb, kb), _ib(ib, jb, kb), MPI_COMM_WORLD, &status);
							for (int i = 0; i < dim[0]; i++)
								fprintf(f, "%.7e ", buff[i]);
						}
					}
				}
			}
		}
		fclose(f);
	}
	MPI_Finalize();
	// free(allgbuff);
	free(data);
	free(next);
	free(buff);
	return 0;
}

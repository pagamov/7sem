#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <mpi.h>
#include <omp.h>


#define _i(i, j, k) (((k) + 1) * (nx + 2) * (ny + 2) + ((j) + 1) * (nx + 2) + (i) + 1)
#define _ib(i, j, k) ((k) * nb_x * nb_y + (j) * nb_x + (i))


int main(int argc, char* argv[]) {
	int ib, jb, kb, nb_x, nb_y, nb_z, nx, ny, nz;
	int id, numproc;
	double lx, ly, lz, hx, hy, hz, down, up, left, right, front, back, u_0;
	double eps, diff;
	double *data, *next, *tmp;
	char fname[100];
	bool f = true;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	if (id == 0) {
		std::cin >> nb_x >> nb_y >> nb_z;
		std::cin >> nx >> ny >> nz;
		scanf("%s", fname);
		std::cin >> eps >> lx >> ly >> lz;
		std::cin >> down >> up >> left >> right >> front >> back >> u_0;
	}

	MPI_Bcast(&nb_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nb_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nb_z, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(fname, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

	kb = id / (nb_x * nb_y);
	jb = id % (nb_x * nb_y) / nb_x;
	ib = id % (nb_x * nb_y) % nb_x;

	hx = lx / (nx * nb_x);
	hy = ly / (ny * nb_y);
	hz = lz / (nz * nb_z);

	int sizes[] = {nz + 2, ny + 2, nx + 2};
	MPI_Datatype left_send;
	int subs_lr[] = {nz, ny, 1};
	int s_lfd[] = {1, 1, 1};
	MPI_Type_create_subarray(3, sizes, subs_lr, s_lfd, MPI_ORDER_C, MPI_DOUBLE, &left_send);
	MPI_Type_commit(&left_send);

	MPI_Datatype front_send;
	int subs_fb[] = {nz, 1, nx};
	MPI_Type_create_subarray(3, sizes, subs_fb, s_lfd, MPI_ORDER_C, MPI_DOUBLE, &front_send);
	MPI_Type_commit(&front_send);

	MPI_Datatype down_send;
	int subs_du[] = {1, ny, nx};
	MPI_Type_create_subarray(3, sizes, subs_du, s_lfd, MPI_ORDER_C, MPI_DOUBLE, &down_send);
	MPI_Type_commit(&down_send);

	MPI_Datatype right_send;
	int s_nx[] = {1, 1, nx};
	MPI_Type_create_subarray(3, sizes, subs_lr, s_nx, MPI_ORDER_C, MPI_DOUBLE, &right_send);
	MPI_Type_commit(&right_send);

	MPI_Datatype back_send;
	int s_ny[] = {1, ny, 1};
	MPI_Type_create_subarray(3, sizes, subs_fb, s_ny, MPI_ORDER_C, MPI_DOUBLE, &back_send);
	MPI_Type_commit(&back_send);

	MPI_Datatype up_send;
	int s_nz[] = {nz, 1, 1};
	MPI_Type_create_subarray(3, sizes, subs_du, s_nz, MPI_ORDER_C, MPI_DOUBLE, &up_send);
	MPI_Type_commit(&up_send);


	MPI_Datatype left_recv;
	int s_x0[] = {1, 1, 0};
	MPI_Type_create_subarray(3, sizes, subs_lr, s_x0, MPI_ORDER_C, MPI_DOUBLE, &left_recv);
	MPI_Type_commit(&left_recv);

	MPI_Datatype front_recv;
	int s_y0[] = {1, 0, 1};
	MPI_Type_create_subarray(3, sizes, subs_fb, s_y0, MPI_ORDER_C, MPI_DOUBLE, &front_recv);
	MPI_Type_commit(&front_recv);

	MPI_Datatype down_recv;
	int s_z0[] = {0, 1, 1};
	MPI_Type_create_subarray(3, sizes, subs_du, s_z0, MPI_ORDER_C, MPI_DOUBLE, &down_recv);
	MPI_Type_commit(&down_recv);

	MPI_Datatype right_recv;
	int s_r[] = {1, 1, nx + 1};
	MPI_Type_create_subarray(3, sizes, subs_lr, s_r, MPI_ORDER_C, MPI_DOUBLE, &right_recv);
	MPI_Type_commit(&right_recv);

	MPI_Datatype back_recv;
	int s_y[] = {1, ny + 1, 1};
	MPI_Type_create_subarray(3, sizes, subs_fb, s_y, MPI_ORDER_C, MPI_DOUBLE, &back_recv);
	MPI_Type_commit(&back_recv);

	MPI_Datatype up_recv;
	int s_z[] = {nz + 1, 1,  1};
	MPI_Type_create_subarray(3, sizes, subs_du, s_z, MPI_ORDER_C, MPI_DOUBLE, &up_recv);
	MPI_Type_commit(&up_recv);

	data = (double*)malloc(sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
	next = (double*)malloc(sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));

	int buffer_size;
	MPI_Pack_size(std::max(nx, ny) * std::max(ny, nz), MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
	buffer_size = 6 * (buffer_size + MPI_BSEND_OVERHEAD);
	double * buffer = (double *)malloc(buffer_size);
	MPI_Buffer_attach(buffer, buffer_size);

	double * allgbuff = (double *)malloc(sizeof(double) * nb_x * nb_y * nb_z);

	for (int i = 0; i < nx; i++)
		for (int j = 0; j < ny; j++)
			for (int k = 0; k < nz; k++)
				data[_i(i, j, k)] = u_0;

	for (int j = 0; j < ny; j++)
		for (int k = 0; k < nz; k++) {
			data[_i(-1, j, k)] = left;	next[_i(-1, j, k)] = left;
		}

	for (int i = 0; i < nx; i++)
		for (int k = 0; k < nz; k++) {
			data[_i(i, -1, k)] = front;	next[_i(i, -1, k)] = front;
		}

	for (int i = 0; i < nx; i++)
		for (int j = 0; j < ny; j++) {
			data[_i(i, j, -1)] = down;	next[_i(i, j, -1)] = down;
		}

	for (int j = 0; j < ny; j++)
		for (int k = 0; k < nz; k++) {
			data[_i(nx, j, k)] = right;	next[_i(nx, j, k)] = right;
		}

	for (int i = 0; i < nx; i++)
		for (int k = 0; k < nz; k++) {
			data[_i(i, ny, k)] = back;	next[_i(i, ny, k)] = back;
		}

	for (int i = 0; i < nx; i++)
		for (int j = 0; j < ny; j++) {
			data[_i(i, j, nz)] = up;	next[_i(i, j, nz)] = up;
		}

	while (f) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (ib + 1 < nb_x)
			MPI_Bsend(data, 1, right_send, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
		if (jb + 1 < nb_y)
			MPI_Bsend(data, 1, back_send, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
		if (kb + 1 < nb_z)
			MPI_Bsend(data, 1, up_send, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
		if (ib > 0)
			MPI_Bsend(data, 1, left_send, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
		if (jb > 0)
			MPI_Bsend(data, 1, front_send, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
		if (kb > 0)
			MPI_Bsend(data, 1, down_send, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		if (ib > 0)
			MPI_Recv(data, 1, left_recv, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
		if (jb > 0)
			MPI_Recv(data, 1, front_recv, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
		if (kb > 0)
			MPI_Recv(data, 1, down_recv, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
		if (ib + 1 < nb_x)
			MPI_Recv(data, 1, right_recv, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
		if (jb + 1 < nb_y)
			MPI_Recv(data, 1, back_recv, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
		if (kb + 1 < nb_z)
			MPI_Recv(data, 1, up_recv, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);

		MPI_Barrier(MPI_COMM_WORLD);
		diff = 0.0;
		#pragma omp parallel shared(data, next) reduction(max:diff)
		{
			int idx = omp_get_thread_num();
			int shift = omp_get_num_threads();
			for (int t = idx; t < nx * ny * nz; t += shift) {
				int i = t % nx;
				int j = t % (nx * ny) / nx;
				int k = t / (nx * ny);
				next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
										   (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
										   (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
										   (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));

				diff = fmax(diff, fabs(next[_i(i, j, k)] - data[_i(i, j, k)]));
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allgather(&diff, 1, MPI_DOUBLE, allgbuff, 1, MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
        f = false;
        for (int i = 0; i < nb_x * nb_y * nb_z; i++)
            if (allgbuff[i] > eps)
                f = true;

		tmp = next;
		next = data;
		data = tmp;
	}

	char * bf = (char*)malloc(sizeof(char) * nx * ny * nz * 14);
	memset(bf, ' ', sizeof(char) * nx * ny * nz * 14);

	for (int k = 0; k < nz; k++)
		for (int j = 0; j < ny; j++)
			for (int i = 0; i < nx; i++)
				sprintf(bf + (k * nx * ny + j * nx + i) * 14, "%.6e", data[_i(i, j, k)]);

	for (int i = 0; i < nx * ny * nz * 14; i++)
		if (bf[i] == '\0')
			bf[i] = ' ';

	MPI_File fp;
	MPI_Datatype filetype;
	int sizes_gr[] = { nz * nb_z, ny * nb_y, nx * nb_x * 14 };
	int subs[] = { nz, ny, nx * 14 };
	int starts[] = { id / (nb_x * nb_y) * nz, id % (nb_x * nb_y) / nb_x * ny,  id % (nb_x * nb_y) % nb_x * nx * 14 };
	MPI_Type_create_subarray(3, sizes_gr, subs, starts, MPI_ORDER_C, MPI_CHAR, &filetype);
	MPI_Type_commit(&filetype);

	MPI_File_delete(fname, MPI_INFO_NULL);
	MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
	MPI_File_set_view(fp, 0, MPI_CHAR, filetype, "native", MPI_INFO_NULL);
	MPI_File_write_all(fp, bf, nx * ny * nz * 14, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&fp);

	MPI_Type_free(&filetype);
	MPI_Type_free(&left_send);
	MPI_Type_free(&left_recv);
	MPI_Type_free(&right_send);
	MPI_Type_free(&right_recv);
	MPI_Type_free(&front_send);
	MPI_Type_free(&front_recv);
	MPI_Type_free(&back_send);
	MPI_Type_free(&back_recv);
	MPI_Type_free(&down_send);
	MPI_Type_free(&down_recv);
	MPI_Type_free(&up_send);
	MPI_Type_free(&up_recv);
	MPI_Finalize();
	free(data);
	free(next);
	free(bf);
	return 0;
}

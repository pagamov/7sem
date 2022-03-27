#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <mpi.h>
#include <omp.h>

using namespace std;

#define _i(i, j, k) (((k) + 1) * (box[0] + 2) * (box[1] + 2) + ((j) + 1) * (box[0] + 2) + (i) + 1)
#define _ib(i, j, k) ((k) * dim[0] * dim[1] + (j) * dim[0] + (i))

int main(int argc, char* argv[]) {
	int ib, jb, kb;
	int id, numproc;
	double hx, hy, hz, down, up, left, right, front, back, u_0;
	double eps, diff;
	double * data, * next, * temp;
	char filename[100];
	bool f = true;
	int dim[3];
	int box[3];
	double l[3];

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	if (id == 0) {
		cin >> dim[0] >> dim[1] >> dim[2];
		cin >> box[0] >> box[1] >> box[2];
		scanf("%s", filename);
		cin >> eps;
		cin >> l[0] >> l[1] >> l[2];
		cin >> down >> up >> left >> right >> front >> back >> u_0;
	}

	MPI_Bcast(dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(box, 3, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(l, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(&down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(&u_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(filename, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

	kb = id / (dim[0] * dim[1]);
	jb = id % (dim[0] * dim[1]) / dim[0];
	ib = id % (dim[0] * dim[1]) % dim[0];

	hx = l[0] / (box[0] * dim[0]);
	hy = l[1] / (box[1] * dim[1]);
	hz = l[2] / (box[2] * dim[2]);

	int sizes[] = {box[2] + 2, box[1] + 2, box[0] + 2};
	MPI_Datatype left_send;
	int subsizes_lr[] = {box[2], box[1], 1};
	int starts_lfd[] = {1, 1, 1};
	MPI_Type_create_subarray(3, sizes, subsizes_lr, starts_lfd, MPI_ORDER_C, MPI_DOUBLE, &left_send);
	MPI_Type_commit(&left_send);

	MPI_Datatype front_send;
	int subsizes_fb[] = {box[2], 1, box[0]};
	MPI_Type_create_subarray(3, sizes, subsizes_fb, starts_lfd, MPI_ORDER_C, MPI_DOUBLE, &front_send);
	MPI_Type_commit(&front_send);

	MPI_Datatype down_send;
	int subsizes_du[] = {1, box[1], box[0]};
	MPI_Type_create_subarray(3, sizes, subsizes_du, starts_lfd, MPI_ORDER_C, MPI_DOUBLE, &down_send);
	MPI_Type_commit(&down_send);

	MPI_Datatype right_send;
	int starts_box[0][] = {1, 1, box[0]};
	MPI_Type_create_subarray(3, sizes, subsizes_lr, starts_box[0], MPI_ORDER_C, MPI_DOUBLE, &right_send);
	MPI_Type_commit(&right_send);

	MPI_Datatype back_send;
	int starts_box[1][] = {1, box[1], 1};
	MPI_Type_create_subarray(3, sizes, subsizes_fb, starts_box[1], MPI_ORDER_C, MPI_DOUBLE, &back_send);
	MPI_Type_commit(&back_send);

	MPI_Datatype up_send;
	int starts_box[2][] = {box[2], 1, 1};
	MPI_Type_create_subarray(3, sizes, subsizes_du, starts_box[2], MPI_ORDER_C, MPI_DOUBLE, &up_send);
	MPI_Type_commit(&up_send);


	MPI_Datatype left_recv;
	int starts_x0[] = {1, 1, 0};
	MPI_Type_create_subarray(3, sizes, subsizes_lr, starts_x0, MPI_ORDER_C, MPI_DOUBLE, &left_recv);
	MPI_Type_commit(&left_recv);

	MPI_Datatype front_recv;
	int starts_y0[] = {1, 0, 1};
	MPI_Type_create_subarray(3, sizes, subsizes_fb, starts_y0, MPI_ORDER_C, MPI_DOUBLE, &front_recv);
	MPI_Type_commit(&front_recv);

	MPI_Datatype down_recv;
	int starts_z0[] = {0, 1, 1};
	MPI_Type_create_subarray(3, sizes, subsizes_du, starts_z0, MPI_ORDER_C, MPI_DOUBLE, &down_recv);
	MPI_Type_commit(&down_recv);

	MPI_Datatype right_recv;
	int starts_r[] = {1, 1, box[0] + 1};
	MPI_Type_create_subarray(3, sizes, subsizes_lr, starts_r, MPI_ORDER_C, MPI_DOUBLE, &right_recv);
	MPI_Type_commit(&right_recv);

	MPI_Datatype back_recv;
	int starts_y[] = {1, box[1] + 1, 1};
	MPI_Type_create_subarray(3, sizes, subsizes_fb, starts_y, MPI_ORDER_C, MPI_DOUBLE, &back_recv);
	MPI_Type_commit(&back_recv);

	MPI_Datatype up_recv;
	int starts_z[] = {box[2] + 1, 1,  1};
	MPI_Type_create_subarray(3, sizes, subsizes_du, starts_z, MPI_ORDER_C, MPI_DOUBLE, &up_recv);
	MPI_Type_commit(&up_recv);


	data = (double *)malloc(sizeof(double) * (box[0] + 2) * (box[1] + 2) * (box[2] + 2));
	next = (double *)malloc(sizeof(double) * (box[0] + 2) * (box[1] + 2) * (box[2] + 2));

	int buffer_size;
	MPI_Pack_size(max(box[0], box[1]) * max(box[1], box[2]), MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
	buffer_size = 6 * (buffer_size + MPI_BSEND_OVERHEAD);
	double * buffer = (double *)malloc(buffer_size);
	MPI_Buffer_attach(buffer, buffer_size);

	double * allgbuff = (double *)malloc(sizeof(double) * box[0] * box[1] * box[2]);

	for (int i = 0; i < box[0]; i++)
		for (int j = 0; j < box[1]; j++)
			for (int k = 0; k < box[2]; k++)
				data[_i(i, j, k)] = u_0;

	for (int j = 0; j < box[1]; j++)
		for (int k = 0; k < box[2]; k++) {
			data[_i(-1, j, k)] = left;
			next[_i(-1, j, k)] = left;
		}

	for (int i = 0; i < box[0]; i++)
		for (int k = 0; k < box[2]; k++) {
			data[_i(i, -1, k)] = front;
			next[_i(i, -1, k)] = front;
		}

	for (int i = 0; i < box[0]; i++)
		for (int j = 0; j < box[1]; j++) {
			data[_i(i, j, -1)] = down;
			next[_i(i, j, -1)] = down;
		}

	for (int j = 0; j < box[1]; j++)
		for (int k = 0; k < box[2]; k++) {
			data[_i(box[0], j, k)] = right;
			next[_i(box[0], j, k)] = right;
		}

	for (int i = 0; i < box[0]; i++)
		for (int k = 0; k < box[2]; k++) {
			data[_i(i, box[1], k)] = back;
			next[_i(i, box[1], k)] = back;
		}

	for (int i = 0; i < box[0]; i++)
		for (int j = 0; j < box[1]; j++) {
			data[_i(i, j, box[2])] = up;
			next[_i(i, j, box[2])] = up;
		}

	while (f) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (ib + 1 < dim[0])
			MPI_Bsend(data, 1, right_send, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
		if (jb + 1 < dim[1])
			MPI_Bsend(data, 1, back_send, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
		if (kb + 1 < dim[2])
			MPI_Bsend(data, 1, up_send, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
		if (ib > 0)
			MPI_Bsend(data, 1, left_send, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
		if (jb > 0)
			MPI_Bsend(data, 1, front_send, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
		if (kb > 0)
			MPI_Bsend(data, 1, down_send, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);

		if (ib > 0)
			MPI_Recv(data, 1, left_recv, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
		if (jb > 0)
			MPI_Recv(data, 1, front_recv, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
		if (kb > 0)
			MPI_Recv(data, 1, down_recv, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
		if (ib + 1 < dim[0])
			MPI_Recv(data, 1, right_recv, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
		if (jb + 1 < dim[1])
			MPI_Recv(data, 1, back_recv, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
		if (kb + 1 < dim[2])
			MPI_Recv(data, 1, up_recv, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);

		MPI_Barrier(MPI_COMM_WORLD);
		diff = 0.0;
		#pragma omp parallel shared(data, next) reduction(max:diff)
		{
			int idx = omp_get_thread_num();
			int shift = omp_get_num_threads();

			for (int t = idx; t < box[0] * box[1] * box[2]; t += shift) {
				int i = t % box[0];
				int j = t % (box[0] * box[1]) / box[0];
				int k = t / (box[0] * box[1]);
				next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
					(data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
					(data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
					(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));

				diff = fmax(diff, fabs(next[_i(i, j, k)] - data[_i(i, j, k)]));
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

	char * bf = (char*)malloc(sizeof(char) * box[0] * box[1] * box[2] * 14);
	memset(bf, ' ', sizeof(char) * box[0] * box[1] * box[2] * 14);

	for (int k = 0; k < box[2]; k++)
		for (int j = 0; j < box[1]; j++)
			for (int i = 0; i < box[0]; i++)
				sprintf(bf + (k * box[0] * box[1] + j * box[0] + i) * 14, "%.6e", data[_i(i, j, k)]);

	for (int i = 0; i < box[0] * box[1] * box[2] * 14; i++)
		if (bf[i] == '\0')
			bf[i] = ' ';

	MPI_File fp;
	MPI_Datatype filetype;
	int sizes_gr[] = { box[2] * dim[2], box[1] * dim[1], box[0] * dim[0] * 14 };
	int subsizes[] = { box[2], box[1], box[0] * 14 };
	int starts[] = { id / (dim[0] * dim[1]) * box[2], id % (dim[0] * dim[1]) / dim[0] * box[1],  id % (dim[0] * dim[1]) % dim[0] * box[0] * 14 };
	MPI_Type_create_subarray(3, sizes_gr, subsizes, starts, MPI_ORDER_C, MPI_CHAR, &filetype);
	MPI_Type_commit(&filetype);

	MPI_File_delete(filename, MPI_INFO_NULL);
	MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONl[1], MPI_INFO_NULL, &fp);
	MPI_File_set_view(fp, 0, MPI_CHAR, filetype, "native", MPI_INFO_NULL);
	MPI_File_write_all(fp, bf, box[0] * box[1] * box[2] * 14, MPI_CHAR, MPI_STATUS_IGNORE);
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
	free(allgbuff);
	free(data);
	free(next);
	free(bf);
	return 0;
}

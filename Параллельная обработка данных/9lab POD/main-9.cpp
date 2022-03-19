#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
#include "mpi.h"

#define _i(i, j, k) (((k) + 1) * (n_x + 2) * (n_y + 2) + ((j) + 1) * (n_x + 2) + (i) + 1)

#define _ib(i, j, k) ((k) * nb_x * nb_y + (j) * nb_x + (i))

int main(int argc, char *argv[]) {
	int ib, jb, kb, nb_x, nb_y, nb_z, n_x, n_y, n_z;
	int i, j, k;
	int id, numproc, proc_name_len;
	char proc_name[MPI_MAX_PROCESSOR_NAME];
	char file_path[256] = "mpi.out";

	double lx, ly, lz, hx, hy, hz, bc_down, bc_up, bc_left, bc_right, bc_back, bc_front;
	double *data, *temp, *next;
	double u_0, max_dif_l, dif, max_dif, eps;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Get_processor_name(proc_name, &proc_name_len);	

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
	next = (double *)malloc(sizeof(double) * (n_x + 2) * (n_y + 2) * (n_z + 2));

	if (id == 0) {	
		fprintf(stderr, "%f %f %f %f %f %f %f %f %f %f %f %f \n", lx, ly, lz, hx, hy, hz, bc_down, bc_up, bc_left, bc_right, bc_back, bc_front);
		fprintf(stderr, "%d %d %d %d %d %d\n",nb_x, nb_y, nb_z, n_x, n_y, n_z );
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

	memcpy(next, data, sizeof(double) * (n_x + 2) * (n_y + 2) * (n_z + 2));

    MPI_Datatype x_face, y_face, z_face, column, row;

    MPI_Type_vector(n_y, 1, n_x + 2, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);

    MPI_Type_vector(n_x, 1, 1, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);


    MPI_Type_create_hvector(n_z, 1, (n_x + 2) * (n_y + 2) * sizeof(double), column, &x_face);
    MPI_Type_commit(&x_face);

    MPI_Type_create_hvector(n_z, 1, (n_x + 2) * (n_y + 2) * sizeof(double), row, &y_face);
    MPI_Type_commit(&y_face);

    MPI_Type_create_hvector(n_y, 1, (n_x + 2) * sizeof(double), row, &z_face);
    MPI_Type_commit(&z_face);


	int it = 0;
	while(1) {
		MPI_Barrier(MPI_COMM_WORLD);
		it++;
		// X-axies (i)
        for(int dir = 0; dir < 2; dir++) {
			if ((ib + dir) & 1) {
				if (ib > 0) {
					MPI_Sendrecv(&data[_i(0, 0, 0)], 1, x_face, _ib(ib - 1, jb, kb), id,
								&data[_i(-1, 0, 0)], 1, x_face, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
                }
			} else {
				if (ib + 1 < nb_x) {
					MPI_Sendrecv(&data[_i(n_x - 1, 0, 0)], 1, x_face, _ib(ib + 1, jb, kb), id,
								&data[_i(n_x, 0, 0)], 1, x_face, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
				}
			}
		}
		// Y-axies (j)
		for(int dir = 0; dir < 2; dir++) {
			if ((jb + dir)& 1) {
				if (jb > 0) {
					MPI_Sendrecv(&data[_i(0, 0, 0)], 1, y_face, _ib(ib, jb - 1, kb), id,
								&data[_i(0, -1, 0)], 1, y_face, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
				}
			} else {
				if (jb + 1 < nb_y) {
					MPI_Sendrecv(&data[_i(0, n_y - 1, 0)], 1, y_face, _ib(ib, jb + 1, kb), id,
								&data[_i(0, n_y, 0)], 1, y_face, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
				}
			}
		}

		// Z-axies (k)
		for(int dir = 0; dir < 2; dir++) {
			if ((kb + dir)& 1) {
				if (kb > 0) {
					MPI_Sendrecv(&data[_i(0, 0, 0)], 1, z_face, _ib(ib, jb, kb - 1), id,
								&data[_i(0, 0, -1)], 1, z_face, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
				}
			} else {
				if (kb + 1 < nb_z) {
					MPI_Sendrecv(&data[_i(0, 0, n_z - 1)], 1, z_face, _ib(ib, jb, kb + 1), id,
								&data[_i(0, 0, n_z)], 1, z_face, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);

		
		max_dif_l = 0;

        #pragma omp parallel for shared(next, data) private(i, j, k, dif) reduction(max:max_dif_l)
		for(i = 0; i < n_x; i++)
			for(j = 0; j < n_y; j++)
				for(k = 0; k < n_z; k++) {
					next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
											(data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
											(data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) / 
												(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
					dif = fabs(next[_i(i, j, k)] - data[_i(i, j, k)]);
					if (dif > max_dif_l)
						max_dif_l = dif;
				}
		// fprintf(stderr, "H: %f, %f, %f\n", hx, hy, hz);
		
		

		temp = next;
		next = data;
		data = temp;

		MPI_Allreduce(&max_dif_l, &max_dif, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		if(max_dif < eps) {
			fprintf(stderr, "e %f, max_dif_l %f, max_dif%f, it%d\n", eps, max_dif_l, max_dif, it);
			fflush(stderr);	
			break;
		}
	}

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

    /*-------------------------------------------------------------------------------------------------------------------------*/

	free(data);
	free(next);
	free(str);
	// free(buffr);
	// free(buffs);

    /*-------------------------------------------------------------------------------------------------------------------------*/

	MPI_Finalize();
	return 0;
}

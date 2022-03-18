#include "mpi.h"
#include <bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

// Индексация внутри блока
#define _i(i, j, k) ((k + 1) * (dim[1] + 2) * (dim[0] + 2) + (j + 1) * (dim[0] + 2)+ i + 1)

// Индексация по блокам (процессам)
#define _ib(i, j, k) ((k) * box[1] * box[0] + (j) * box[0] + (i))
#define _ibz(id) ((id) / box[0] / box[1])
#define _iby(id) (((id) % (box[0] * box[1])) / box[0])
#define _ibx(id) ((id) % box[0])

int main(int argc, char *argv[]) {

    int id, ib, jb, kb;
    int i, j, k;

    // Input
    int dim[3];
    int box[3];
    double l[3];
    double u[6];
    double eps, u_0;
    string filename;

    double hx, hy, hz;
    double *data, *tmp, *next, *buff;
    MPI_Status status;

    int numproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if (id == 0) {
        cin >> box[0] >> box[1] >> box[2];
        cin >> dim[0] >> dim[1] >> dim[2];
        cin >> filename;
        cin >> eps;
        cin >> l[0] >> l[1] >> l[2];
        cin >> u[4] >> u[5] >> u[0] >> u[1] >> u[2] >> u[3] >> u_0;
    }

    // Передача параметров расчета всем процессам
    MPI_Bcast(dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(box, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(l, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    hx = l[0] / (double)(dim[0] * box[0]);
    hy = l[1] / (double)(dim[1] * box[1]);
    hz = l[2] / (double)(dim[2] * box[2]);

    double h2x = 1.0 / (hx * hx), h2y = 1.0 / (hy * hy), h2z = 1.0 / (hz * hz);

    ib = _ibx(id);
    jb = _iby(id);
    kb = _ibz(id);

    data = (double *)malloc(sizeof(double) * (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2));
    next = (double *)malloc(sizeof(double) * (dim[0] + 2) * (dim[1] + 2) * (dim[2] + 2));

    int mx = max(max(dim[0], dim[1]), dim[2]);
    buff = (double *)malloc(sizeof(double) * (mx + 2) * (mx + 2));


    for (i = 0; i < dim[0]; i++)
        for (j = 0; j < dim[1]; j++)
            for (k = 0; k < dim[2]; k++)
                data[_i(i, j, k)] = u_0;

    double difference = 0.0;
    do {
        // Right
        if (ib + 1 < box[0]) {
            for (k = 0; k < dim[2]; k++)
                for (j = 0; j < dim[1]; j++)
                    buff[j + k * dim[1]] = data[_i(dim[0] - 1, j, k)];
            MPI_Bsend(buff, dim[1] * dim[2], MPI_DOUBLE, _ib(ib + 1, jb, kb), 0, MPI_COMM_WORLD);
        }

        // Up
        if (jb + 1 < box[1]) {
            for (k = 0; k < dim[2]; k++)
                for (i = 0; i < dim[0]; i++)
                    buff[i + k * dim[0]] = data[_i(i, dim[1] - 1, k)];
            MPI_Bsend(buff, dim[0] * dim[2], MPI_DOUBLE, _ib(ib, jb + 1, kb), 0, MPI_COMM_WORLD);
        }

        // Back
        if (kb + 1 < box[2]) {
            for (j = 0; j < dim[1]; j++)
                for (i = 0; i < dim[0]; i++)
                    buff[i + j * dim[0]] = data[_i(i, j, dim[2] - 1)];
            MPI_Bsend(buff, dim[0] * dim[1], MPI_DOUBLE, _ib(ib, jb, kb + 1), 0, MPI_COMM_WORLD);
        }


	    if (ib > 0) {
            MPI_Recv(buff, dim[1] * dim[2], MPI_DOUBLE, _ib(ib - 1, jb, kb), 0, MPI_COMM_WORLD, &status);
            for (k = 0; k < dim[2]; k++)
                for (j = 0; j < dim[1]; j++)
                    data[_i(-1, j, k)] = buff[j + k * dim[1]];
        } else {
            for (k = 0; k < dim[2]; k++)
                for (j = 0; j < dim[1]; j++)
                    data[_i(-1, j, k)] = u[0];
        }

        if (jb > 0) {
            MPI_Recv(buff, dim[0] * dim[2], MPI_DOUBLE, _ib(ib, jb - 1, kb), 0, MPI_COMM_WORLD, &status);
            for (k = 0; k < dim[2]; k++)
                for (i = 0; i < dim[0]; i++)
                    data[_i(i, -1, k)] = buff[i + k * dim[0]];
        } else {
            for (k = 0; k < dim[2]; k++)
                for (i = 0; i < dim[0]; i++)
                    data[_i(i, -1, k)] = u[2];
        }

        if (kb > 0) {
            MPI_Recv(buff, dim[0] * dim[1], MPI_DOUBLE, _ib(ib, jb, kb - 1), 0, MPI_COMM_WORLD, &status);
            for (j = 0; j < dim[1]; j++)
                for (i = 0; i < dim[0]; i++)
                    data[_i(i, j, -1)] = buff[i + j * dim[0]];
        } else {
            for (j = 0; j < dim[1]; j++)
                for (i = 0; i < dim[0]; i++)
                    data[_i(i, j, -1)] = u[4];
        }

        // Left
        if (ib > 0) {
            for (k = 0; k < dim[2]; k++)
                for (j = 0; j < dim[1]; j++)
                    buff[j + k * dim[1]] = data[_i(0, j, k)];
            MPI_Bsend(buff, dim[1] * dim[2], MPI_DOUBLE, _ib(ib - 1, jb, kb), 0, MPI_COMM_WORLD);
        }

        // Down
        if (jb > 0) {
            for (k = 0; k < dim[2]; k++)
                for (i = 0; i < dim[0]; i++)
                    buff[i + k * dim[0]] = data[_i(i, 0, k)];
            MPI_Bsend(buff, dim[0] * dim[2], MPI_DOUBLE, _ib(ib, jb - 1, kb), 0, MPI_COMM_WORLD);
        }

        // Front
        if (kb > 0) {
            for (j = 0; j < dim[1]; j++)
                for (i = 0; i < dim[0]; i++)
                    buff[i + j * dim[0]] = data[_i(i, j, 0)];
            MPI_Bsend(buff, dim[0] * dim[1], MPI_DOUBLE, _ib(ib, jb, kb - 1), 0, MPI_COMM_WORLD);
        }

        // Data recieve
        if (ib + 1 < box[0]) {
            MPI_Recv(buff, dim[1] * dim[2], MPI_DOUBLE, _ib(ib + 1, jb, kb), 0, MPI_COMM_WORLD, &status);
            for (k = 0; k < dim[2]; k++)
                for (j = 0; j < dim[1]; j++)
                    data[_i(dim[0], j, k)] = buff[j + k * dim[1]];
        } else {
            for (k = 0; k < dim[2]; k++)
                for (j = 0; j < dim[1]; j++)
                    data[_i(dim[0], j, k)] = u[1];
        }

        if (jb + 1 < box[1]) {
            MPI_Recv(buff, dim[0] * dim[2], MPI_DOUBLE, _ib(ib, jb + 1, kb), 0, MPI_COMM_WORLD, &status);
            for (k = 0; k < dim[2]; k++)
                for (i = 0; i < dim[0]; i++)
                    data[_i(i, dim[1], k)] = buff[i + k * dim[0]];
        } else {
            for (k = 0; k < dim[2]; k++)
                for (i = 0; i < dim[0]; i++)
                    data[_i(i, dim[1], k)] = u[3];
        }

        if (kb + 1 < box[2]) {
            MPI_Recv(buff, dim[0] * dim[1], MPI_DOUBLE, _ib(ib, jb, kb + 1), 0, MPI_COMM_WORLD, &status);
            for (j = 0; j < dim[1]; j++)
                for (i = 0; i < dim[0]; i++)
                    data[_i(i, j, dim[2])] = buff[i + j * dim[0]];
        } else {
            for (j = 0; j < dim[1]; j++)
                for (i = 0; i < dim[0]; i++)
                    data[_i(i, j, dim[2])] = u[5];
        }

        difference = 0.0;
        for (i = 0; i < dim[0]; i++) {
            for (j = 0; j < dim[1]; j++) {
                for (k = 0; k < dim[2]; k++) {
                    next[_i(i, j, k)] = ((data[_i(i - 1, j, k)] + data[_i(i + 1, j, k)]) * h2x + \
                                         (data[_i(i, j - 1, k)] + data[_i(i, j + 1, k)]) * h2y + \
                                         (data[_i(i, j, k - 1)] + data[_i(i, j, k + 1)]) * h2z) / \
                                         (2 * (h2x + h2y + h2z));
                    difference = max(difference, abs(next[_i(i, j, k)] - data[_i(i, j, k)]));
                }
            }
        }
        MPI_Allreduce(&difference, &difference, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        tmp = next;
        next = data;
        data = tmp;

    } while (difference > eps);

    if (id != 0) {
        for (k = 0; k < dim[2]; k++) {
            for (j = 0; j < dim[1]; j++) {
                for (i = 0; i < dim[0]; i++)
                    buff[i] = data[_i(i, j, k)];
                MPI_Bsend(buff, dim[0], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        FILE * file;
        file = fopen(filename.c_str(), "w");

        for (kb = 0; kb < box[2]; kb++) {
            for (k = 0; k < dim[2]; k++) {
                for (jb = 0; jb < box[1]; jb++) {
                    for (j = 0; j < dim[1]; j++) {
                        for (ib = 0; ib < box[0]; ib++) {
                            if (_ib(ib, jb, kb) == 0)
                                for (i = 0; i < dim[0]; i++)
                                    buff[i] = data[_i(i, j, k)];
                            else
                                MPI_Recv(buff, dim[0], MPI_DOUBLE, _ib(ib, jb, kb), 0, MPI_COMM_WORLD, &status);
                            for (i = 0; i < dim[0]; i++)
                                fprintf(file, "%.7e ", buff[i]);
                        }
                    }
                }
            }
        }
        fclose(file);
    }

    MPI_Finalize();
    free(buff);
    free(data);
    free(next);
    return 0;
}

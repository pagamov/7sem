#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <math.h>

using namespace std;

#define _i(x,y,z) arr[(x) + (y) * (xp * xl) + (z) * (xp * xl) * (yp * yl)]
#define _in(x,y,z) next[(x) + (y) * (xp * xl) + (z) * (xp * xl) * (yp * yl)]

int main() {
    int xp, yp, zp;                                         //сетка процессов
    int xl, yl, zl;                                         //размер обработки одним процессом
    string filename;                                        //название файла
    double eps;                                             //эпсилон для проверки конца процесса
    double xL, yL, zL;                                      //размер полотка из ячеек
    double u_down, u_up, u_left, u_right, u_front, u_back;  //гранич условия
    double u_0;                                             //начальное значение
    
    cin >> xp >> yp >> zp;
    cin >> xl >> yl >> zl;
    cin >> filename;
    cin >> eps;
    cin >> xL >> yL >> zL;
    cin >> u_down >> u_up >> u_left >> u_right >> u_front >> u_back;
    cin >> u_0;
    
    double * arr = (double *)malloc(sizeof(double) * xp * xl * yp * yl * zp * zl);
    double * next =(double *)malloc(sizeof(double) * xp * xl * yp * yl * zp * zl);
    double * tmp;
    
    // инициализация начальных значений
    for (int z = 0; z < zp * zl; z++) {
        for (int y = 0; y < yp * yl; y++) {
            for (int x = 0; x < xp * xl; x++) {
                arr[x + y * (xp * xl) + z * (xp * xl) * (yp * yl)] = u_0;
            }
        }
    }
    
    double hx = xL / (double)(xp * xl), hy = yL / (double)(yp * yl), hz = zL / (double)(zp * zl);
    double h2x = hx * hx, h2y = hy * hy, h2z = hz * hz;
    
    h2x = 1.0 / h2x;    h2y = 1.0 / h2y;    h2z = 1.0 / h2z;
    
    
    double u_down_, u_up_, u_left_, u_right_, u_front_, u_back_;
    bool f = true;
    int iter = 0;
    while(f) {
        cout << "iter " << iter++ << endl;
        // считать значения в next
        for (int z = 0; z < zp * zl; z++) {
            for (int y = 0; y < yp * yl; y++) {
                for (int x = 0; x < xp * xl; x++) {
                    
                    // u_down_, u_up_, u_left_, u_right_, u_front_, u_back_;
                    
                    
                    if (x == 0) {
                        u_left_ = u_left;
                        u_right_ = _i(x + 1, y, z);
                    } else if (x == xp * xl - 1) {
                        u_left_ = _i(x - 1, y, z);
                        u_right_ = u_right;
                    } else {
                        u_left_ = _i(x - 1, y, z);
                        u_right_ =_i(x + 1, y, z);
                    }
                    
                    if (y == 0) {
                        u_front_ = u_front;
                        u_back_ = _i(x, y - 1, z);
                    } else if (y == yp * yl - 1) {
                        u_front_ =_i(x, y + 1, z);
                        u_back_ = u_back;
                    } else {
                        u_front_ =_i(x, y + 1, z);
                        u_back_ = _i(x, y - 1, z);
                    }
                    
                    if (z == 0) {
                        u_down_ =u_down;
                        u_up_ =  _i(x, y, z + 1);
                    } else if (z == zp * zl - 1) {
                        u_down_ =_i(x, y, z - 1);
                        u_up_ =  u_up;
                    } else {
                        u_down_ =_i(x, y, z - 1);
                        u_up_ =  _i(x, y, z + 1);
                    }
                    
                    
                    _in(x,y,z) = ((u_left_ + u_right_) * h2x + (u_front_ + u_back_) * h2y + (u_down_ + u_up_) * h2z) / (2 * (h2x + h2y + h2z));
                }
            }
        }
        
        // проверять ошибку
        // bool f = true;
        eps = 0.1;
        f = false;
        for (int z = 0; z < zp * zl; z++) {
            for (int y = 0; y < yp * yl; y++) {
                for (int x = 0; x < xp * xl; x++) {
                    if (abs(_in(x,y,z) - _i(x, y, z)) >= eps) {
                        f = true;
                        //cout << abs(next[x + y * (xp * xl) + z * (xp * xl) * (yp * yl)] - _i(x, y, z)) << ' ' << endl;
                    }
                }
            }
        }
        
        // swap
        tmp = next;
        next = arr;
        arr = tmp;
        
        // if (iter > 10) {
        //     break;
        // }
        
        for (int z = 0; z < zp * zl; z++) {
            for (int y = 0; y < yp * yl; y++) {
                for (int x = 0; x < xp * xl; x++) {
                    cout << arr[x + y * (xp * xl) + z * (xp * xl) * (yp * yl)] << ' ';
                }
                cout << endl;
            }
            cout << endl;
        }
        
        cout << "-/-/-/-/-/-/-/-/-" << endl;
        
        // и выходить если ошибка меньше eps
    }
    
    
    
    
    
    // запись в файл 
    // ofstream myfile;
    // myfile.open(file);
    FILE * file;
    file = fopen(filename.c_str(), "w");
    for (int z = 0; z < zp * zl; z++) {
        for (int y = 0; y < yp * yl; y++) {
            for (int x = 0; x < xp * xl; x++) {
                // myfile << std::fixed << std::setprecision(6) << arr[x + y * (xp * xl) + z * (xp * xl) * (yp * yl)] << ' ';
                fprintf(file, "%.7e ", arr[x + y * (xp * xl) + z * (xp * xl) * (yp * yl)]);
                // вопрос с выводом мантисы то ли 6 то ли 7
            }
            // myfile << endl; // потом убрать
            fprintf(file, "\n");
        }
        // myfile << endl; // потом убрать
        fprintf(file, "\n");
    }
    // myfile.close();
    fclose(file);
    free(arr);
    free(next);
    return 0;
}
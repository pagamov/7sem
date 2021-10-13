#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <math.h>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

using namespace std;

// текстурная ссылка <тип элементов, размерность, режим нормализации>
texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(uchar4 * out, int w, int h) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	uchar4 p;
    uchar4 piv;

    for (y = idy; y < h; y += offsety) {
		for (x = idx; x < w; x += offsetx) {
			p = tex2D(tex, x, y);
            
            float gx = 0;
            float gy = 0;
            
            int sx1[2] = {max(min(x+1, w-1),0),max(min(x-1, w-1),0)};
            int sy1[3] = {max(min(y, h-1),0),max(min(y+1, h-1),0),max(min(y-1, h-1),0)};
            
            for (int j=0;j<3;j++) {
                piv = tex2D(tex, sx1[0], sy1[j]);
                gx += (float)piv.x * 0.299 + (float)piv.y * 0.587 + (float)piv.z * 0.114;
                piv = tex2D(tex, sx1[1], sy1[j]);
                gx -= (float)piv.x * 0.299 + (float)piv.y * 0.587 + (float)piv.z * 0.114;
            }
            
            int sx2[3] = {max(min(x, w-1),0),max(min(x-1, w-1),0),max(min(x+1, w-1),0)};
            int sy2[2] = {max(min(y+1, h-1),0),max(min(y-1, h-1),0)};
            
            for (int j=0;j<3;j++) {
                piv = tex2D(tex, sx2[j], sy2[0]);
                gy += (float)piv.x * 0.299 + (float)piv.y * 0.587 + (float)piv.z * 0.114;
                piv = tex2D(tex, sx2[j], sy2[1]);
                gy -= (float)piv.x * 0.299 + (float)piv.y * 0.587 + (float)piv.z * 0.114;
            } 
            
            float g = sqrt(gx*gx + gy*gy);      
            unsigned char mean = (unsigned char) min(255,(int)g);
            out[y * w + x] = make_uchar4(mean, mean, mean, p.w);
		}
    }
}

int main() {
    string filename1, filename2;
    cin >> filename1 >> filename2;
	int w, h;
	FILE *fp = fopen(filename1.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	// Подготовка данных для текстуры
	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, w, h));

	CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

	// Подготовка текстурной ссылки, настройка интерфейса работы с данными
	tex.addressMode[0] = cudaAddressModeClamp;	// Политика обработки выхода за границы по каждому измерению
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;		// Без интерполяции при обращении по дробным координатам
	tex.normalized = false;						// Режим нормализации координат: без нормализации

	// Связываем интерфейс с данными
	CSC(cudaBindTextureToArray(tex, arr, ch));

	uchar4 *dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

	kernel<<<dim3(16, 16), dim3(16, 32)>>>(dev_out, w, h);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	// Отвязываем данные от текстурной ссылки
	CSC(cudaUnbindTexture(tex));

	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_out));

	fp = fopen(filename2.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(data);
	return 0;
}

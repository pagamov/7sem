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

__global__ void prewitt(uchar4 * res, int w, int h) {
    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < h; y += blockDim.y * gridDim.y) {
		for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < w; x += blockDim.x * gridDim.x) {
			uchar4 p = tex2D(tex, x, y);
            uchar4 piv;
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
			res[y * w + x] = make_uchar4(mean, mean, mean, p.w);
		}
    }
}

int main() {
    string filename1, filename2;
    cin >> filename1 >> filename2;
	int w, h;
	
	FILE * f = fopen(filename1.c_str(), "rb");
	fread(&w, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	uchar4 * data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, f);
	fclose(f);

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

	uchar4 * dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));
	
	// Prepare
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
// Start record
cudaEventRecord(start, 0);
// Do something on GPU

	prewitt<<<dim3(16, 16), dim3(16, 32)>>>(dev_out, w, h);
	
// Stop event
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
// Clean up:
cudaEventDestroy(start);
cudaEventDestroy(stop);

	CSC(cudaGetLastError());

	CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	// Отвязываем данные от текстурной ссылки
	CSC(cudaUnbindTexture(tex));

	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_out));

	f = fopen(filename2.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, f);
	fwrite(&h, sizeof(int), 1, f);
	fwrite(data, sizeof(uchar4), w * h, f);
	fclose(f);

	free(data);
	return 0;
}

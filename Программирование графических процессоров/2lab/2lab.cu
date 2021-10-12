#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

struct Pixel {unsigned char r, g, b, a;};

__global__ void prewitt (int x, int y, Pixel * pic, Pixel * res) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < x * y; i += gridDim.x * blockDim.x) {
        int yt = i / x;
        int xt = i - yt * x;
        float gx = 0;
        float gy = 0;
        
        int sx1[2] = {max(min(xt+1, x-1),0),max(min(xt-1, x-1),0)};
        int sy1[3] = {max(min(yt, y-1),0),max(min(yt+1, y-1),0),max(min(yt-1, y-1),0)};
        
        for (int j=0;j<3;j++) gx += (float)pic[sx1[0]+sy1[j]*x].r;
        for (int j=0;j<3;j++) gx -= (float)pic[sx1[1]+sy1[j]*x].r;
        
        int sx2[3] = {max(min(xt, x-1),0),max(min(xt-1, x-1),0),max(min(xt+1, x-1),0)};
        int sy2[2] = {max(min(yt+1, y-1),0),max(min(yt-1, y-1),0)};
        
        for (int j=0;j<3;j++) gy += (float)pic[sx2[j]+sy2[0]*x].r;
        for (int j=0;j<3;j++) gy -= (float)pic[sx2[j]+sy2[1]*x].r;
        
        float g = min((float)255.0,sqrt(gx*gx + gy*gy));
        
        res[i].r = (unsigned char)g;
        res[i].g = (unsigned char)g;
        res[i].b = (unsigned char)g;
        res[i].a = pic[i].a;
    }
}

class Image {
public:
    int x,y;
    Pixel * pixels;
    ~Image() {free(pixels);}
    void load(string filename) {
        FILE * in = fopen(filename.c_str(), "rb");
        // if (in == NULL) {fprintf(stderr, "cant open file", 30);}
        fread(&x,4,1,in);
        fread(&y,4,1,in);
        pixels = (Pixel *)malloc(sizeof(Pixel)*x*y);
        for (int i = 0; i < x * y; i++) {
            fread(&pixels[i].r,1,1,in);
            fread(&pixels[i].g,1,1,in);
            fread(&pixels[i].b,1,1,in);
            fread(&pixels[i].a,1,1,in);
        }
        fclose(in);
    }
    void save(string filename) {
        FILE * out = fopen(filename.c_str(), "wb");
        if (out == NULL) {cerr << "cant open file" << endl;}
        fwrite(&x,4,1,out);
        fwrite(&y,4,1,out);
        for (int i = 0; i < x * y; i++) {
            fwrite(&pixels[i].r,1,1,out);
            fwrite(&pixels[i].g,1,1,out);
            fwrite(&pixels[i].b,1,1,out);
            fwrite(&pixels[i].a,1,1,out);
        }
        fclose(out);
    }
    void ink() {
        for (int i = 0; i < x*y; i++) {
            float newx = (float)pixels[i].r * 0.299 + \
                               (float)pixels[i].g * 0.587 + \
                               (float)pixels[i].b * 0.114;
            unsigned char mean = (unsigned char) min((float)255.0,newx);
            pixels[i].r = mean; pixels[i].g = mean; pixels[i].b = mean;
        }
    }
};

int main() {
    string filename1, filename2;
    cin >> filename1 >> filename2;
    
    Image pic;
    pic.load(string(filename1));
    pic.ink();
    
    Pixel *dev_pic, *dev_res;
    cudaMalloc(&dev_pic, sizeof(Pixel) * pic.x * pic.y);
    cudaMalloc(&dev_res, sizeof(Pixel) * pic.x * pic.y);

    cudaMemcpy(dev_pic, pic.pixels, sizeof(Pixel) * pic.x * pic.y, cudaMemcpyHostToDevice);

    prewitt <<<2,2>>>(pic.x, pic.y, dev_pic, dev_res);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(pic.pixels, dev_res, sizeof(Pixel) * pic.x * pic.y, cudaMemcpyDeviceToHost);

    cudaFree(dev_pic);
    cudaFree(dev_res);
    
    pic.save(string(filename2));
    return 0;
}
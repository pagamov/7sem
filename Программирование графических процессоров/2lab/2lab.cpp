#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

class Pixel {
public:
    unsigned char r, g, b, a;
    Pixel(unsigned char r, unsigned char g, unsigned char b, unsigned char a): r(r),g(g),b(b),a(a) {}
    unsigned char mean() {
        float res = (float)r * 0.299 + (float)g * 0.587 + (float)b * 0.114;
        if (res < 0 || res > 255) {cerr << "err in mean" << endl;}
        return (unsigned char)res;
    }
};

class Image {
public:
    int x,y;
    vector <Pixel> pixels;
    void load(string filename) {
        char r, g, b, a;
        FILE * in = fopen(filename.c_str(), "rb");
        if (in == NULL) {cerr << "cant open file" << endl;}
        fread(&x,4,1,in);
        fread(&y,4,1,in);
        for (int i = 0; i < x * y; i++) {
            fread(&r,1,1,in);
            fread(&g,1,1,in);
            fread(&b,1,1,in);
            fread(&a,1,1,in);
            pixels.push_back(Pixel(r,g,b,a));
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
        unsigned char r, g, b, a;
        unsigned char mean;
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                mean = pixels[i + j * x].mean();
                if (mean < 0 || mean > 255) {cerr << "err in ink" << endl;}
                pixels[i + j * x].r = mean;
                pixels[i + j * x].g = mean;
                pixels[i + j * x].b = mean;
            }
        }
    }
    float Gx(int i, int j) {
        int right = (int)get(i + 1, j).r + (int)get(i + 1, j + 1).r + (int)get(i + 1, j - 1).r;
        int left =  (int)get(i - 1, j).r + (int)get(i - 1, j + 1).r + (int)get(i - 1, j - 1).r;
        return (float)(right - left);
    }
    float Gy(int i, int j) {
        int top =   (int)get(i, j + 1).r + (int)get(i - 1, j + 1).r + (int)get(i + 1, j + 1).r;
        int bott =  (int)get(i, j - 1).r + (int)get(i - 1, j - 1).r + (int)get(i + 1, j - 1).r;
        return (float)(-bott + top);
    }
    Pixel get(int i, int j) {
        int ip = max(min(i, x-1), 0);
        int jp = max(min(j, y-1), 0);
        return pixels[ip + x * jp];
    }
};

class BMP32 {
public:
    int x, y; //left-right bottom-top
    vector <Pixel> pixels;
    char *pre_header, *after_header;
    bool header_loaded;
    
    BMP32() {
        pre_header = (char *)malloc(sizeof(char) * 18);
        after_header = (char *)malloc(sizeof(char) * 96);
        header_loaded = false;
    }
    ~BMP32() {
        free(pre_header);
        free(after_header);
    }
    void load(string filename) {
        FILE * in = fopen(filename.c_str(), "rb");
        if (in == NULL) {cerr << "cant open file" << endl;}
        unsigned char r, g, b, a;
        fread(pre_header,18,1,in);
        // fseek(in,18,SEEK_SET); //?
        fread(&x,4,1,in);
        fread(&y,4,1,in);
        fread(after_header,96,1,in);
        
        for (int i = 0; i < x * y; i++) {
            fread(&r,1,1,in);
            fread(&g,1,1,in);
            fread(&b,1,1,in);
            fread(&a,1,1,in);
            pixels.push_back(Pixel(r,g,b,a));
        }
        fclose(in);
        header_loaded = true;
    }
    void loadBinary(string filename) {
        if (!header_loaded) {cerr << "cant find header" << endl;}
        FILE * in = fopen(filename.c_str(), "rb");
        if (in == NULL) {cerr << "cant open file" << endl;}
        unsigned char r, g, b, a;
        fread(&x,4,1,in);
        fread(&y,4,1,in);
        pixels.clear();
        for (int i = 0; i < x * y; i++) {
            fread(&r,1,1,in);
            fread(&g,1,1,in);
            fread(&b,1,1,in);
            fread(&a,1,1,in);
            pixels.push_back(Pixel(r,g,b,a));
        }
        fclose(in);
    }
    void save(string filename) {
        FILE * out = fopen(filename.c_str(), "wb");
        if (out == NULL) {cerr << "cant open file" << endl;}
        fwrite(pre_header,18,1,out);
        fwrite(&x,4,1,out);
        fwrite(&y,4,1,out);
        fwrite(after_header,96,1,out);
        for (int i = 0; i < x * y; i++) {
            fwrite(&pixels[i].r,1,1,out);
            fwrite(&pixels[i].g,1,1,out);
            fwrite(&pixels[i].b,1,1,out);
            fwrite(&pixels[i].a,1,1,out);
        }
        fclose(out);
    }
    void saveBinary(string filename) {
        if (!header_loaded) {cerr << "cant find header" << endl;}
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
    Pixel get(int i, int j) {
        int ip = max(min(i, x-1), 0);
        int jp = max(min(j, y-1), 0);
        return pixels[ip + x * jp];
    }
    void ink() {
        unsigned char r, g, b, a;
        unsigned char mean;
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                mean = pixels[i + j * x].mean();
                if (mean < 0 || mean > 255) {cerr << "err in ink" << endl;}
                pixels[i + j * x].r = mean;
                pixels[i + j * x].g = mean;
                pixels[i + j * x].b = mean;
            }
        }
    }
    float Gx(int i, int j) {
        int right = (int)get(i + 1, j).r + (int)get(i + 1, j + 1).r + (int)get(i + 1, j - 1).r;
        int left =  (int)get(i - 1, j).r + (int)get(i - 1, j + 1).r + (int)get(i - 1, j - 1).r;
        return (float)(right - left);
    }
    float Gy(int i, int j) {
        int top =   (int)get(i, j + 1).r + (int)get(i - 1, j + 1).r + (int)get(i + 1, j + 1).r;
        int bott =  (int)get(i, j - 1).r + (int)get(i - 1, j - 1).r + (int)get(i + 1, j - 1).r;
        return (float)(bott - top);
    }
};

template <typename T>
vector <Pixel> prewitt(T& im) {
    vector <Pixel> pixels;
    vector <float> Grag;
    float newc;
    unsigned char res;
    float maxGrag = 0;
    // for (int j = 0; j < im.y; j++) {
    //     for (int i = 0; i < im.x; i++) {
    //         float gx = im.Gx(i, j);
    //         float gy = im.Gy(i, j);
    //         float g = sqrt(gx*gx + gy*gy);
    //         if (g > maxGrag) {
    //             maxGrag = g;
    //         }
    //         // newc = (unsigned char) g;
    //         // pixels.push_back(Pixel(newc, newc, newc, im.pixels[i + im.x * j].a));
    //     }
    // }
    // cout << maxGrag << endl;
    for (int j = 0; j < im.y; j++) {
        for (int i = 0; i < im.x; i++) {
            float gx = im.Gx(i, j);
            float gy = im.Gy(i, j);
            float g = sqrt(gx*gx + gy*gy);
            newc = min((float)255, g);
            res = (unsigned char) newc;
            pixels.push_back(Pixel(res, res, res, im.pixels[i + im.x * j].a));
        }
    }
    return pixels;
}

int main(int argc, char ** argv) {
    // Image pic;
    BMP32 pic;
    string filename1, filename2;
    // cin >> filename1 >> filename2;
    pic.load(string("dog.bmp"));
    // clock_t start = clock(), end;
    pic.ink();
    pic.pixels = prewitt(pic);
    // end = clock();
    // cout << (end - start)/CLOCKS_PER_SEC << endl;
    pic.save(string("dogres.bmp"));
    return 0;
}
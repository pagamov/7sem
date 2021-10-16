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
    void clean() {
        for (int j = 0; j < y; j++) {
            for (int i = 0; i < x; i++) {
                pixels[i + j * x].a = 0;
            }
        }
    }
};

float dist(Pixel p, Pixel d) {
    float res;
    res = sqrt(                           \
        pow((float)p.r - (float)d.r, 2) + \
        pow((float)p.g - (float)d.g, 2) + \
        pow((float)p.b - (float)d.b, 2)   \
    );
    return res;
}

vector <Pixel> Kmean(Image im, int n, int * cl) {
    for (int y = 0; y < im.y; y++) {
        for (int x = 0; x < im.x; x++) {
            unsigned char piv = 0;
            int resClas = -1;
            float maxDist = 10000;
            for (int i = 0, clas = 0; clas < n; clas++, i+=2) {
                float pivDist = dist(                   \
                    im.pixels[x + y * im.x],            \
                    im.pixels[cl[i] + cl[i+1] * im.x]   \
                );
                if (pivDist < maxDist) {
                    resClas = clas;
                    maxDist = pivDist;
                }
            }
            im.pixels[x + y * im.x].a = (unsigned char)resClas;
        }
    }
    return im.pixels;
}

int main(int argc, char ** argv) {
    Image pic;
    string filename1, filename2;
    int n;
    
    cin >> filename1 >> filename2 >> n;
    int * cl = (int *)malloc(sizeof(int) * n * 2);
    for (int i = 0; i < n * 2; i += 2) {
        cin >> cl[i] >> cl[i + 1];
    }
    
    pic.load(filename1);
    pic.clean();
    pic.pixels = Kmean(pic, n, cl);
    pic.save(filename2);
    
    free(cl);
    return 0;
}
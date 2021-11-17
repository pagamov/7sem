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

vector <Pixel> Kmean(Image im, int n, int *newx, int *newy, int *newz) {
    for (int y = 0; y < im.y; y++) {
        for (int x = 0; x < im.x; x++) {
            unsigned char piv = 0;
            int resClas = -1;
            float maxDist = 10000;
            for (int i = 0, clas = 0; clas < n; clas++, i+=2) {
                float pivDist = dist(                   \
                    im.pixels[x + y * im.x],            \
                    Pixel(newx[i],newy[i],newz[i],0)   \
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
    srand(time(NULL));
    Image pic;
    string filename1, filename2;
    int n = 32;
    
    cin >> filename1 >> filename2;
    
    int curx[32],cury[32],curz[32];
    
    pic.load(filename1);
    pic.clean();
    
    for (int j = 0; j < n; j++) {
        curx[j] = (int)(rand() % 255);
        cury[j] = (int)(rand() % 255);
        curz[j] = (int)(rand() % 255);
    }
    
    int newx[32],newy[32],newz[32],num[32];
    
    for (int iter = 0; iter < 10; iter++) {
        pic.pixels = Kmean(pic, n, curx,cury,curz);
        
        for (int i = 0; i < n; i++) {
            num[i] = 0; newx[i] = 0; newy[i] = 0; newz[i] = 0;
        }
        
        for (int y = 0; y < pic.y; y++) {
            for (int x = 0; x < pic.x; x++) {
                int idx = pic.pixels[y * pic.x + x].a;
                newx[idx] += pic.pixels[y * pic.x + x].r;
                newy[idx] += pic.pixels[y * pic.x + x].g;
                newz[idx] += pic.pixels[y * pic.x + x].b;
                num[idx] += 1;
            }
        }
        
        for (int i = 0; i < n; i++) {
            if (num[i] != 0) {
                newx[i] /= num[i]; newy[i] /= num[i]; newz[i] /= num[i];
            } else {
                // printf("error in del num\n");
            }
        }
        
        for (int i = 0; i < n; i++) {
            curx[i] = newx[i]; cury[i] = newy[i]; curz[i] = newz[i];
        }
        printf("%d\n",iter);
    }
    
    for (int y = 0; y < pic.y; y++) {
        for (int x = 0; x < pic.x; x++) {
            int idx = pic.pixels[y * pic.x + x].a;
            pic.pixels[y * pic.x + x].a = 255;
            pic.pixels[y * pic.x + x].r = curx[idx];
            pic.pixels[y * pic.x + x].g = cury[idx];
            pic.pixels[y * pic.x + x].b = curz[idx];
        }
    }
    
    
    
    pic.save(filename2);
    
    // free(cl);
    // free(newcl);
    return 0;
}
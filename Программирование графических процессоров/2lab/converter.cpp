#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

int main(int argc, char ** argv) {
    FILE *f = fopen("data", "wb");
    int a = 3, b = 3;
    fwrite(&a, sizeof(a),1,f);
    fwrite(&b, sizeof(b),1,f);
    int d [] = {0x00030201,0x00060504,0x00090807, \
                0x00070809,0x00040506,0x00010203, \
                0x00000000,0x00141414,0x00000000};
                
    // int d [] = {0x00000000,0x00000000,0x00000000, \
    //             0x00000000,0x00808080,0x00000000, \
    //             0x00000000,0x00000000,0x00000000};
    fwrite(&d, sizeof(int) * a * b, 1, f);
    fclose(f);
    
    // int red;
    // FILE *ff = fopen(argv[1], "rb");
    // fread(&red, sizeof(int), 1, ff);
    // cout << red << endl;
    // fclose(ff);
    return 0;
}
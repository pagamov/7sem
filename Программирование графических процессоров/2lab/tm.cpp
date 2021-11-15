#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

int main(int argc, char ** argv) {
    FILE *f = fopen("data", "wb");
    int a, b;
    cin >> a >> b;
    fwrite(&a, sizeof(a),1,f);
    fwrite(&b, sizeof(b),1,f);
    // int d [] = {0x00030201,0x00060504,0x00090807, \
    //             0x00070809,0x00040506,0x00010203, \
    //             0x00000000,0x00141414,0x00000000};
                
    // int d [] = {0x00000000,0x00000000,0x00000000, \
    //             0x00000000,0x00808080,0x00000000, \
    //             0x00000000,0x00000000,0x00000000};
    int d = 0x00030201;
    for (int i = 0; i < a*b; i++) {
        fwrite(&d, sizeof(int), 1, f);
    }
    // fwrite(&d, sizeof(int) * a * b, 1, f);
    fclose(f);
    return 0;
}
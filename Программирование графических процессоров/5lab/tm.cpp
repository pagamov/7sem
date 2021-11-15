#include <iostream>
#include <vector>
#include <fstream>
#include <time.h>

using namespace std;

int main(int argc, char ** argv) {
    srand(time(NULL));
    FILE * f = fopen("data", "wb");
    int n,d;
    cin >> n;
    fwrite(&n, 4, 1, f);
    
    for (int i = 0; i < n; i++) {
        d = (int)(rand() % 1024);
        fwrite(&d, sizeof(int), 1, f);
    }
    fclose(f);
    return 0;
}
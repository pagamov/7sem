#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std;

class Test {
public:
    int n;
    vector <int> v1;
    vector <int> v2;
    
    Test(int n): n(n) { 
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            v1.push_back(rand() % 1000 + 1);
            v2.push_back(rand() % 1000 + 1);
        }
    }
    
    void save(string filename) {
        FILE * out = fopen(filename.c_str(), "w");
        if (out == NULL) {cerr << "cant open file" << endl;}
        fprintf(out, "%d\n", n);
        for (int i = 0; i < n; i++)
            fprintf(out, "%d ", v1[i]);
        for (int i = 0; i < n; i++)
            fprintf(out, "%d ", v2[i]);
        fclose(out);
    }
};

int main() {
    int n;
    cin >> n;
    Test t(n);
    t.save(string("data"));
    return 0;
}
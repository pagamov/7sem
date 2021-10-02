#include <iostream>
#include <vector>
using namespace std;

int main() {
    int c;
    vector <int> v;
    for (int i = 0; i < 6; i++) {
        cin >> c;
        v.push_back(c);
    }
    for (int i = 0; i < 6; i++) {
        cout << v[i];
    }
    cout << endl;
    
    return 0;
}
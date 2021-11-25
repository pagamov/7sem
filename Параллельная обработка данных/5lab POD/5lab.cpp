#include <iostream>
#include <fstream>

using namespace std;

void oddEvenSorting(int * array, int N) {
	for (size_t i = 0; i < N; i++) {
		for (size_t j = (i % 2) ? 0 : 1; j + 1 < N; j += 2) {
			if (array[j] > array[j + 1]) {
				std::swap(array[j], array[j + 1]);
			}
		}
	}
}

int main() {
	bool binary = true;
	int n;
	
	if (binary) {
		fread(&n, 4, 1, stdin);
	} else {
		cin >> n;
	}
	
	int * arr = (int *)malloc(n * sizeof(int));
	
	if (binary) {
		fread(arr, 4, n, stdin);
	} else {
		for (int i = 0; i < n; i++) {
			cin >> arr[i];
		}
	}
	
	oddEvenSorting(arr, n);
	
	if (binary) {
		fwrite(arr, 4, n, stdout);
	} else {
		for (int i = 0; i < n; i++) {
			cout << arr[i] << " ";
		}
		cout << endl;
	}
	free(arr);
    return 0;
}
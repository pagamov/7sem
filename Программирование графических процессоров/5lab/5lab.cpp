#include <iostream>
#include <fstream>

template < typename T, size_t N >
void oddEvenSorting(T (&array)[N]) {
	for (size_t i = 0; i < N; i++) {
	    // (i % 2) ? 0 : 1 возвращает 1, если i четное, 0, если i не четное
		for (size_t j = (i % 2) ? 0 : 1; j + 1 < N; j += 2) {
			if (array[j] > array[j + 1]) {
				std::swap(array[j], array[j + 1]);
			}
		}
	}
}

int main(int argc, char ** argv) {
    int n;
	fread(&n, 4, 1, stdin);
	int * arr = (int *)malloc(n * sizeof(int));
	fread(arr, 4, n, stdin);
	
	fwrite(arr, 4, n, stdout);	
	free(arr);
    return 0;
}
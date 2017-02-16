// Author : Nikhil Kulkarni
// Date : Feb 15, 2017
// Desc : Merge sort code from CLRS
// Notes: I am returning to c++ after almost 4 years
// last time I wrote in code in c++ was in 3rd year of my
// engineering. Need to refresh data structures and algorithms

#include <iostream>
// For assigning last value as infinity importing limits
#include <limits>
using namespace std;

int* Merge(int* arr, int p, int q, int  r) {

    // Question : In c++ does the array start with 0 or 1?
    // Ans : starts with 0 same as python

    //condition on p,q,r is that p <= q < r
    //we will divide arr into two arrays left and right one
    int n1 = q - p + 1;
    int n2 = r - q;

    // Initialising arrays L and R
    int L[n1 + 1];
    int R[n2 + 1];

    // Question : What is correct i++ or ++i?

    // Populate the arrays
    // Note: Declaration inside array is not considered
    // So here i as good as not declared
    for (int i = 0; i < n1; i++)
    {
        L[i] = arr[i];
    }
    for (int j = 0; j < n2; j++)
    {
        // stuck here to figure out what goes in bracket after A
        R[j] = arr[n1 + j];
    }

    // Assigning infinity as last elements
    L[n1] = numeric_limits<int>::max();
    R[n2] = numeric_limits<int>::max();


    // Beware C++ doesn't allow single quotes in place of double quotes

    int i = 0;
    int j = 0;
    for (int k = p; k < r; k++)
    {
        if (L[i] <= R[j])
        {
            // cout << "Comapared " << L[i] << "With" << R[j] << endl;
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            // cout << "Comapared " << L[i] << "With" << R[j] << endl;
            j++;
        }
    }
    cout << "Sorted Array" << endl;
    for ( unsigned int k = 0; k < r; k = k + 1 )
    {
        cout << arr[k] << ",";

    }
    return arr;
}

int* MergeSort(int* A, int p, int r)
{
    cout << endl;
    cout << "Inside MergeSort function with " << endl;
    cout << "P = " << p << endl;
    cout << "R = " << r << endl;
    cout << "Array" << endl;
    for ( unsigned int k = p; k < r; k = k + 1 )
    {
        cout << A[k] << ",";

    }
    if (p < r)
    {
        int q = (p + r) / 2;
        MergeSort(A, p, q);
        MergeSort(A, q + 1, r);
        Merge(A, p, q, r);
    }
}

int main ()
{
    int arr[] = {5, 2, 4, 7, 1, 3, 2, 6};
    int size = sizeof(arr) / sizeof(arr[0]);
    int* t = MergeSort(arr, 0, size);
    for ( unsigned int k = 0; k < size; k = k + 1 )
    {
        cout << t[k] << ",";

    }
    return 0;
}
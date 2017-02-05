// my second program in C++
#include <iostream>
#include <limits>
using namespace std;

int* MergeSort(int* arr, int size){
	int r = size;
	cout <<"Value of r :" << r << endl;
	int q = r/2;
	cout <<"Value of q :" << q << endl;
	int p = 0;
	int* ls;
	int* js;
	cout <<"------------------------"<< endl;
  	for( unsigned int a = 0; a < q-p; a = a + 1 )
  	{
  		cout <<"Value of a :" << a << endl;
  		ls[a] = arr[a];
  		cout <<"Value of l[a] :" << ls[a] << endl;
  	}
  	cout <<"------------------------"<< endl;
  	for( unsigned int z = 0; z < r-q; z = z + 1 )
  	{
  		cout <<"Value of z :" << z << endl;
  		js[z] = arr[z+q];
  		cout <<"Value of j[z] :" << js[z] << endl;
  	}
  	ls[q] = numeric_limits<int>::max();
  	js[r-q] = numeric_limits<int>::max();

  	int* l = MergeSort(ls,q);
  	int* j = MergeSort(js,r-q);

  	cout <<"Value of l[last] :" << q << ":"<< l[q] << endl;
  	cout <<"Value of j[last] :" << r-q << ":" << j[r-q] << endl;
  	cout <<"------------------------"<< endl;
  	cout <<"l vector" << endl;
  	cout <<"------------------------"<< endl;
  	for ( unsigned int k = 0; k < q+1; k = k + 1 )
  	{
  		//cout << k << endl;
  		cout << l[k] << endl;
  	}
	cout <<"------------------------"<< endl;
	cout <<"------------------------"<< endl;
  	cout <<"j vector" << endl;
  	cout <<"------------------------"<< endl;
  	for ( unsigned int k = 0; k < r-q+1; k = k + 1 )
  	{
  		//cout << k << endl;
  		cout << j[k] << endl;
  	}
	cout <<"------------------------"<< endl;
  	int e = 0;
  	int w = 0;
  	for ( unsigned int k = 0; k < r+1; k = k + 1 )
  	{
  		if(l[e]<=j[w]){
  			arr[k] = l[e];
  			e = e + 1;
  		}
  		else{
  			arr[k] = j[w];
  			w = w + 1;
  		}
  	}
  	return arr;

}

int main ()
{
	int arr[] = {110,2,3,4,5,6,7};
	int size = sizeof(arr)/sizeof(arr[0]);
	int* t = MergeSort(arr,size);
	cout <<"------------------------"<< endl;
  	cout <<"final sorted vector" << endl;
  	cout <<"------------------------"<< endl;
  	for ( unsigned int k = 0; k < size; k = k + 1 )
  	{
  		cout << t[k] << endl;
  	}	
}
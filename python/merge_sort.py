'''
Author : Nikhil Kulkarni
Desc   : Merge sort after failed attempts
'''


def Merge(arr, p, q, r):

    n1 = q - p + 1
    n2 = r - q

    L = arr[p - 1:q]
    R = arr[q:r]

    print 'left', L
    print 'right', R

    i = 0
    j = 0
    k = p - 1

    while (k < r - 1):
        if (L[i] <= R[j]):
            arr[k] = L[i]
            i = i + 1
        else:
            arr[k] = R[j]
            j = j + 1
        k = k + 1

    while (i < n1):
        arr[k] = L[i]
        i = i + 1
        k = k + 1

    while (j < n2):
        arr[k] = R[j]
        j = j + 1
        k = k + 1
    print 'Array ', arr

    return arr


def MergeSort(A, p, r):

    if p < r:
        q = (r + p) / 2
        print '\n\n'
        print 'P =>', p
        print 'Q =>', q
        print 'R =>', r
        A = MergeSort(A, p, q)
        A = MergeSort(A, q + 1, r)
        A = Merge(A, p, q, r)
    else:
        print "\nGoing to next MergeSort Call/Merge"
    return A


def main(A):
    size = len(A)
    print 'Array A:', A
    print 'size:', size
    A = MergeSort(A, 1, size)
    return A

if __name__ == "__main__":
    m = [5, 2, 4, 7, 1, 3, 2, 6]
    sorted_array = main(m)
    print 'sorted_array', sorted_array

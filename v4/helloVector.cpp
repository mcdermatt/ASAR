#include <iostream>
using namespace std;

struct Vector{int sz;
              double* elem;};

void vector_init(Vector& v, int s){
    v.elem = new double[s];
    v.sz = s;
}

double read_and_sum_(int s){
    Vector v;
    vector_init(v,s);

    for (int i =0; i!=s; i++){
        cin >> v.elem[i];
        }
    double sum = 0;
    for (int i=0; i!=s; ++i){
        sum+=v.elem[i]; // compute the sum of the elements
    
    }

    // cout << "sum: " << sum << endl;
    return sum;

}


int main()
{
    int S = read_and_sum_(5);
    cout << S << endl;
}
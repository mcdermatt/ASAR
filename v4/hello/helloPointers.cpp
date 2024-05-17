#include <iostream> 
using namespace std;

int main()
{
// cout << "Hello World! Welcome to your first C++ program!" << endl;

int a = 15;
auto *b = &a;                   //Pointer-- stores the memory address of something else

cout << "a  " << a << endl;
cout << "b  " << b << endl;
cout << "&a  " << &a << endl;   // memory address
cout << "*b  " << *b << endl;   // dereference operator * 
                                // used to obtain the value pointed to by a pointer
                                // in this case, should be equal to a
cout << "----------------------" << endl;

*b = 10;
cout << "a  " << a << endl;     //changing the value stored at the location pointed to by b updates a as well 
cout << "*b  " << *b << endl;
cout << "b  " << b << endl;

//point b at g
int g = 123;
b = &g;
cout << "*b  " << *b << endl;
cout << "b  " << b << endl;

cout << "----------------------" << endl;

auto &c = a;                    //Reference-- alias for another object
                                // anything done to reference also happens to parent
                                // alias can not be moved to something else??

cout << "c  " << c << endl;
cout << "&c  " << &c << endl;

c = g;
cout << "c  " << c << endl;
cout << "&c  " << &c << endl;
cout << "a  " << a << endl;

}
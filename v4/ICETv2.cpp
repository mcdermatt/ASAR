#include <GL/glew.h>
#include <GL/glut.h>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "csv-parser/single_include/csv.hpp"
#include <fstream>
#include <cmath>
#include <limits>
#include <algorithm>  // Include the algorithm header for std::sort
#include <map>
#include <execution>
#include "ThreadPool.h"

using namespace Eigen;
using namespace std;

class ICET {
public:
    ICET(int rl) : runlen(rl) {}

    void step(){
        cout << "stepping " << endl;
        cout << runlen << endl;
        runlen--;
    }

private:
    int runlen;

};

int main() {

    int rl = 5;
    ICET it(rl);
    it.step();
    it.step();

}
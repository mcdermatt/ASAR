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
#include <ctime>

using namespace Eigen;
using namespace std;

//Script to test and improve runtime for finding spherical cluster bounds 

Eigen::MatrixXf points(250000, 3);  // Declare points as a global variable
GLfloat azimuthalMin = 0.0;
GLfloat azimuthalMax = 2.0 * M_PI;
GLfloat elevationMin = -M_PI / 4.0;
GLfloat elevationMax = M_PI / 4.0;
GLfloat innerDistance = 5.0;
GLfloat outerDistance = 10.0;

//test -- set cluster bounds as [n, 6] matrix
Eigen::MatrixXd clusterBounds(1000,6);

// Function to convert Cartesian coordinates to spherical coordinates
MatrixXf cartesianToSpherical(const MatrixXf& cartesianPoints) {
    // Ensure that the input matrix has 3 columns (X, Y, Z coordinates)
    assert(cartesianPoints.cols() == 3);

    // Extract X, Y, Z columns
    VectorXf x = cartesianPoints.col(0);
    VectorXf y = cartesianPoints.col(1);
    VectorXf z = cartesianPoints.col(2);

    // Compute radius (r)
    // VectorXf r = VectorXf::Zero(cartesianPoints.rows());
    VectorXf r = cartesianPoints.rowwise().norm();

    // Compute azimuthal angle (theta)
    VectorXf theta = VectorXf::Zero(cartesianPoints.rows());
    for (int i = 0; i < cartesianPoints.rows(); ++i) {
        theta(i) = std::atan2(y(i), x(i));
        if (theta(i) < 0.0) {
            theta(i) += 2.0 * M_PI;
        }
    }
    // // Compute elevation angle (phi)
    VectorXf phi = VectorXf::Zero(cartesianPoints.rows());
    for (int i = 0; i < cartesianPoints.rows(); ++i) {
        phi(i) = std::acos(z(i) / r(i));
        // std::cout << "phi(i): \n" << phi(i) << "\n";
    }

    // Combine r, theta, phi into a new matrix
    MatrixXf sphericalPoints(cartesianPoints.rows(), 3);
    sphericalPoints << r, theta, phi;

    // Replace NaN or -NaN values with [0, 0, 0]
    for (int i = 0; i < sphericalPoints.rows(); ++i) {
        for (int j = 0; j < sphericalPoints.cols(); ++j) {
            if (std::isnan(sphericalPoints(i, j))) {
                sphericalPoints(i, j) = 1000.0;
            }
        }
    }

    return sphericalPoints;
}

template<typename Scalar>
std::pair<Scalar, Scalar> findClusterFast(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& radii, int n, Scalar thresh) {
    Scalar innerDistance = 0.0;
    Scalar outerDistance = 0.0;

    // std::cout << "points within bounds \n" << radii << std::endl;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> forwardDifference = radii.tail(radii.size() - 1) - radii.head(radii.size() - 1);
    // std::cout << "\n forward difference \n" << forwardDifference << std::endl;

    // Modify innerDistance and outerDistance based on your logic

    return {innerDistance, outerDistance};
}

// Function to find the first sufficiently large cluster of points
pair<float, float> findCluster(const MatrixXf& sphericalCoords, int n, float thresh) {
    int numPoints = sphericalCoords.rows();

    float innerDistance = 0.0;
    float outerDistance = 0.0;
    vector<Vector3f> localPoints;

    for (int i = 0; i < numPoints; i++) {
        Vector3f point = sphericalCoords.row(i);
        float r = point(0);
        // float theta = point(1);
        // float phi = point(2);

        // Filtering points based on azimuthal, elevation range, and radial distance
        // if (theta >= azimuthalMin && theta <= azimuthalMax &&
        //     phi >= elevationMin && phi <= elevationMax) {

        // Check for jumps in radial distance to identify clusters
        if (!localPoints.empty() && std::abs(localPoints.back()(0) - r) > thresh) {
            if (localPoints.size() >= n) {
                // Found a sufficiently large cluster
                innerDistance = localPoints.front()(0);
                outerDistance = localPoints.back()(0);
                cout << "Found cluster - Inner Distance: " << innerDistance << ", Outer Distance: " << outerDistance << endl;
                return {innerDistance, outerDistance};
            } else {
                // Reset the cluster if it's not large enough
                localPoints.clear();
            }
        }
        // Add the point to the current cluster
        localPoints.push_back(point);
        // }
    }
    return {innerDistance, outerDistance};
}

vector<vector<vector<int>>> sortSphericalCoordinates(const MatrixXf& sphericalCoords, int numBinsTheta, int numBinsPhi) {
    // Create a 2D vector of vectors to store point indices in each bin
    vector<vector<vector<int>>> pointIndices(numBinsTheta, vector<vector<int>>(numBinsPhi));

    // Iterate through each spherical coordinate
    for (int i = 0; i < sphericalCoords.rows(); ++i) {
        // Extract phi and theta values
        float phi = sphericalCoords(i, 1);
        float theta = sphericalCoords(i, 2);

        // Calculate bin indices
        int binTheta = static_cast<int>((theta / (2 * M_PI)) * numBinsTheta) % numBinsTheta;
        int binPhi = static_cast<int>((phi / M_PI) * numBinsPhi) % numBinsPhi;

        // Store the point index in the corresponding bin
        pointIndices[binTheta][binPhi].push_back(i);
    }

    // Return the vector of point indices
    return pointIndices;
}

int main(int argc, char** argv) {

    // load point data from .csv ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Ouster Sample Dataset
    std::string csvFilePath = "sample_data/pcap_out_000106.csv";
    // Open the CSV file
    std::ifstream file(csvFilePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the CSV file." << std::endl;
        return 1;
    }
    // Parse and process the CSV file, skipping the first row
    csv::CSVReader reader(file, csv::CSVFormat().header_row(1).trim({}));
    
    // Initialize a vector to store rows temporarily
    std::vector<Eigen::Vector3f> rows;

    csv::CSVRow row;
    reader.read_row(row); // Skip the first row
    csv::CSVRow secondRow;
    reader.read_row(secondRow); // Skip the second row
    // Parse and process the CSV file
    // Iterate over rows and fill the vector
    for (csv::CSVRow& currentRow : reader) {
        // Assuming three columns in each row
        Eigen::Vector3f rowData;
        rowData << static_cast<float>(currentRow[8].get<int>()),
                   static_cast<float>(currentRow[9].get<int>()),
                   static_cast<float>(currentRow[10].get<int>());

        // Append the row to the vector
        rows.push_back(rowData);
    }

    // Close the file before processing the vector
    file.close();

    // Preallocate memory for the dataMatrix
    Eigen::MatrixXf dataMatrix(rows.size(), 3);

    // Copy the data from the vector to the dataMatrix
    for (size_t i = 0; i < rows.size(); ++i) {
        dataMatrix.row(i) = rows[i]/1000;
    }

    points = dataMatrix;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Set the frustum parameters as needed
    azimuthalMin = 120 * (M_PI/ 180); //0-2pi
    azimuthalMax = 150 * (M_PI/ 180);
    elevationMin =  75 * (M_PI/ 180);
    elevationMax = 90 * (M_PI/ 180);
    int n = 30; // Size of the cluster
    float thresh = 0.1; // Threshold for radial distance
    Eigen::MatrixXf pointsSpherical = cartesianToSpherical(points);
    std::cout << "pointsSpherical: \n" << pointsSpherical.rows() << "\n";

    // Sort sphericalCoords based on radial distance
    vector<int> index(pointsSpherical.rows());
    iota(index.begin(), index.end(), 0);
    sort(index.begin(), index.end(), [&](int a, int b) {
        return pointsSpherical(a, 0) < pointsSpherical(b, 0); // Sort by radial distance
    });

    // Create a sorted matrix using the sorted indices
    MatrixXf sortedPointsSpherical(pointsSpherical.rows(), pointsSpherical.cols());
    for (int i = 0; i < pointsSpherical.rows(); i++) {
        sortedPointsSpherical.row(i) = pointsSpherical.row(index[i]);
    }

    // MatrixXf every10thPoint = points.topRows(points.rows() / 1000).bottomRows(points.rows() / 1000);

    // set up spherical voxel grid ~~~~~~~~~~~~~~~~~~~~~~~~~
    int azimBins = 40;
    int elevBins = 25;
    float azimMin = 0;
    float azimMax = 2*M_PI;
    float elevMin = M_PI/4;
    float elevMax = 5*M_PI/4;

    int totalBins = azimBins*elevBins; 

    auto before = std::chrono::system_clock::now();
    auto beforeMs = std::chrono::time_point_cast<std::chrono::milliseconds>(before);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // for (int i = 0; i < totalBins; i++){
    //     //get [azimMin_i, azimMax_i, elevMin_i, elevMax_i, inner_i, outer_i]
    //     float azimMin_i = (i % azimBins) * (azimMax - azimMin) / azimBins;  
    //     float azimMax_i = ((i+1) % azimBins) * (azimMax - azimMin) / azimBins;  
    //     if ((i+1) % azimBins == 0){
    //         azimMax_i = 2*M_PI - 0.00001;
    //     }
    //     float elevMin_i = floor(i / azimBins) * (elevMax - elevMin) / elevBins;
    //     float elevMax_i = (floor(i / azimBins) + 1) * (elevMax - elevMin) / elevBins;

    //     MatrixXf test = sortedPointsSpherical.block(100*i, 0, 1000, points.cols());

    //     // // old slow method
    //     // pair<float, float> clusterDistances = findCluster(test, azimMin_i, azimMax_i, elevMin_i, elevMax_i, n, thresh);
    //     // innerDistance = clusterDistances.first;
    //     // outerDistance = clusterDistances.second;

    //     // testing new faster(?) method
    //     pair<float, float> clusterDistances = findClusterFast(sortedPointsSpherical.col(0).cast<double>(), n, thresh);

    //     clusterBounds.row(i) << azimMin_i, azimMax_i, elevMin_i, elevMax_i, innerDistance, outerDistance;
    //     // cout << clusterBounds.row(i) << " i " << i << endl;
    //     // cout << clusterBounds(i,0) << "  " << clusterBounds(i,1) << " test " << i << endl;
    // }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    int numBinsPhi = 40;  // Adjust the number of bins as needed
    int numBinsTheta = 25; // Adjust the number of bins as needed

    vector<vector<vector<int>>> pointIndices = sortSphericalCoordinates(sortedPointsSpherical, numBinsPhi, numBinsTheta);

    for (int phi = 0; phi < numBinsPhi; phi++){
        for (int theta = 0; theta< numBinsTheta; theta++){
            // Retrieve the point indices
            const vector<int>& indices = pointIndices[phi][theta];

            // cout << "Dimensions of sortedPointsSpherical: " << sortedPointsSpherical.rows() << " x " << sortedPointsSpherical.cols() << endl;
            // cout << "Size of indices: " << indices.size() << endl;

            // only calculate inner/outer bounds if there are a sufficient number of points in the spike 
            if (indices.size() > 30) {
                // // Print the point indices
                // cout << "Point indices from [binsPhi = " << desiredPhi << "][binsTheta = " << desiredTheta << "]: ";
                // for (int index : indices) {
                //     // cout << index << " ";
                //     cout << sortedPointsSpherical.row(index) << endl;
                // }
                // cout << endl;

                // Use the indices to access the corresponding rows in sortedPointsSpherical
                MatrixXf selectedPoints = MatrixXf::Zero(indices.size(), sortedPointsSpherical.cols());
                for (int i = 0; i < indices.size(); ++i) {
                    selectedPoints.row(i) = sortedPointsSpherical.row(indices[i]);
                }
                // cout << "Points from [binsPhi = " << desiredPhi << "][binsTheta = " << desiredTheta << "]: \n" << endl;
                // cout << selectedPoints << endl;

                // Call the fast function
                // std::pair<double, double> clusterDistances = findClusterFast<double>(selectedPoints.col(0).cast<double>(), n, thresh);
                // Access the results
                // double innerDistance = clusterDistances.first;
                // double outerDistance = clusterDistances.second;

                // call the old function for finding cluster distances
                pair<float, float> clusterDistances = findCluster(selectedPoints, n, thresh);
                innerDistance = clusterDistances.first;
                outerDistance = clusterDistances.second;
            }
            // use -1 value as a flag for unoccupied voxles
            else{
                float innerDistance = -1;
                float outerDistance = -1;
                
            }
            //convert [desiredPhi][desiredTheta] to azimMin, azimMax, elevMin, elevMax
            float azimMin_i =  (static_cast<float>(theta) / numBinsTheta) * (2 * M_PI) ;
            float azimMax_i =  (static_cast<float>(theta+1) / numBinsTheta) * (2 * M_PI) ;
            float elevMin_i =  (static_cast<float>(phi) / numBinsPhi) * (M_PI) ;
            float elevMax_i =  (static_cast<float>(phi+1) / numBinsPhi) * (M_PI) ;

            // cout << "\n azimMin: " << azimMin_i << endl;
            // cout << "azimMax: " << azimMax_i << endl;
            // cout << "phiMin: " << elevMin_i << endl;
            // cout << "phiMax: " << elevMax_i << endl;

            clusterBounds.row(numBinsTheta*phi + theta ) << azimMin_i, azimMax_i, elevMin_i, elevMax_i, innerDistance, outerDistance;

        }

    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    auto after = std::chrono::system_clock::now();
    auto afterMs = std::chrono::time_point_cast<std::chrono::milliseconds>(after);

    auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(afterMs - beforeMs).count();

    cout << "Elapsed time: " << elapsedTimeMs << " ms" << endl;
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    return 0;
}

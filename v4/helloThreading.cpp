#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <queue>
#include "ThreadPool.h"

using namespace std;

void do_work(const int vec_size){

    vector<int> random_numbers(vec_size);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1,1000);
    for (int &num : random_numbers) {num = distrib(gen);}
    vector<int> &a = random_numbers;
    sort(a.begin(), a.end());
}


int main() {
    const int num_threads = 4;
    const int num_calls = 128;
    const int vec_size = 1'000;
    cout << num_calls << " function calls for vec size " << vec_size << endl;

    //for single thread ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    auto start = chrono::high_resolution_clock::now();
    for (int count = 0; count < num_calls; count++){
        do_work(vec_size);
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    std::cout << "Took: " << duration.count() << " seconds using single thread" << std::endl;

    //multi-thread~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // create and delete threads after each function call
    auto start_multi = chrono::high_resolution_clock::now();
    mutex mtx;
    condition_variable cv;


    int count = 0;
    while (count < num_calls){
        vector<thread> threads;
        for (int i = 0; i< num_threads; i++){
            threads.emplace_back(do_work, vec_size);
        }
        for (auto &thread:threads){
            if (thread.joinable()){
                thread.join();
                count += 1;
                // cout << count << endl;
                // // Break out of the loop if count reaches 10 // causes leak!!!!
                // if (count >= 10) {
                //     break;
                // }
            }
        }
    }

    auto end_multi = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_multi = end_multi - start_multi;
    std::cout << "Took: " << duration_multi.count() << " seconds using " << num_threads << " threads" << std::endl;
    cout << count << endl;

    //thread pool~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // reuses fixed number of threads to avoid overhead of creating and destroying each
    auto start_pool = chrono::high_resolution_clock::now();

    ThreadPool pool(4);
    vector<future<void>> futures;
    int c = 0;

    while (c < num_calls) {
        for (int i = 0; i < num_threads && c < num_calls; ++i) { // 5 tasks per iteration
            futures.push_back(pool.enqueue(do_work, vec_size));
            c++;
        }
    }
    // Wait for all tasks to complete
    for (auto &fut : futures) {
        fut.get();
    }
    auto end_pool = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_pool = end_pool - start_pool;
    std::cout << "Took: " << duration_pool.count() << " seconds using pool with " << num_threads << " threads" << std::endl;
    cout << c << endl;


    return 0;

}
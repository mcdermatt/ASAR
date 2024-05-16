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

using namespace std;

void do_work(const int vec_size){

    vector<int> random_numbers(vec_size);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1,1000);
    for (int &num : random_numbers) {num = distrib(gen);}
    vector<int> &a = random_numbers;
    sort(a.begin(), a.end());

    // for (int i = 0; i< 10; i++){
    //     cout << a[i] << " ";
    // }
    // cout << endl;

}

class ThreadPool {
public:
    ThreadPool(size_t numThreads);
    ~ThreadPool();

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queueMutex;
    std::condition_variable condition;
    std::atomic<bool> stop;

    void workerThread();
};

ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back(&ThreadPool::workerThread, this);
    }
}

ThreadPool::~ThreadPool() {
    stop.store(true);
    condition.notify_all();
    for (std::thread &worker : workers) {
        worker.join();
    }
}

void ThreadPool::workerThread() {
    while (!stop) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            condition.wait(lock, [this] { return stop.load() || !tasks.empty(); });
            if (stop.load() && tasks.empty())
                return;
            task = std::move(tasks.front());
            tasks.pop();
        }
        task();
    }
}

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
    using returnType = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<returnType()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<returnType> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}


int main() {
    const int num_threads = 4;
    const int num_calls = 256;
    const int vec_size = 100'000;
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
                // // Break out of the loop if count reaches 10 // causes leak!
                // if (count >= 10) {
                //     break;
                // }
            }
        }
    }

    auto end_multi = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_multi = end_multi - start_multi;
    std::cout << "Took: " << duration_multi.count() << " seconds using " << num_threads << " threads" << std::endl;

    //thread pool~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // reuses fixed number of threads to avoid overhead of creating and destroying each
    auto start_pool = chrono::high_resolution_clock::now();

    ThreadPool pool(4);
    vector<future<void>> futures;
    int c = 0;

    while (c < num_calls) {
        for (int i = 0; i < num_threads; ++i) { // 5 tasks per iteration
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


    return 0;

}
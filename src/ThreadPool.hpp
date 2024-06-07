#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

template <typename T> class BPopQueue {
public:
  T pop() {
    std::unique_lock<std::mutex> mlock(mutex);
    while (queue.empty())
      cond.wait(mlock);

    auto item = queue.front();
    queue.pop();
    return item;
  }

  void push(T const &item) {
    std::unique_lock<std::mutex> mlock(mutex);
    queue.push(item);
    mlock.unlock();
    cond.notify_one();
  }

private:
  std::queue<T> queue;
  std::mutex mutex;
  std::condition_variable cond;
};

class ThreadPool {

public:
  ThreadPool(int size) {
    for (auto i = 0; i < size; i++)
      threads.push_back(std::thread(&ThreadPool::doWork, this));
  }

  template <typename F, typename... Args>
  auto queueWork(F &&f, Args &&...args) -> std::future<decltype(f(args...))> {
    auto work = std::make_shared<std::packaged_task<decltype(f(args...))()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    auto wrapper = new std::function<bool()>([work] {
      (*work)();
      return false;
    });

    jobs.push(wrapper);
    return work->get_future();
  }

  void terminate() {
    auto termFn = new std::function<bool()>([] { return true; });
    for (auto it = threads.begin(); it != threads.end(); it++)
      jobs.push(termFn);

    for (auto it = threads.begin(); it != threads.end(); it++)
      it->join();
  }

  ~ThreadPool() { terminate(); }

private:
  void doWork() {
    while (true) {
      auto func = jobs.pop();
      if ((*func)()) {
        // delete func;
        return;
      }
      // delete func;
    }
  }

  std::vector<std::thread> threads;
  BPopQueue<std::function<bool()> *> jobs;
};

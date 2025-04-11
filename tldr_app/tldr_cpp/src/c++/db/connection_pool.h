#ifndef TLDR_CPP_CONNECTION_POOL_H
#define TLDR_CPP_CONNECTION_POOL_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <iostream>

namespace tldr {
    template<typename ConnectionType>
    class ConnectionPool {
    public:
        using CreateFunc = std::function<ConnectionType*(const std::string &)>;
        using CloseFunc = std::function<void(ConnectionType *)>;

        // Default constructor
        ConnectionPool() : create_conn_(nullptr), close_conn_(nullptr) {
        }

        // Constructor takes:
        // - connection string or path
        // - pool size
        // - function to create connection
        // - function to close connection
        ConnectionPool(
            const std::string &conn_str,
            size_t pool_size,
            CreateFunc create_conn,
            CloseFunc close_conn
        ) : conn_str_(conn_str), create_conn_(create_conn), close_conn_(close_conn) {
            for (size_t i = 0; i < pool_size; ++i) {
                try {
                    auto conn = create_conn(conn_str_);
                    pool_.push(conn);
                } catch (const std::exception &e) {
                    std::cerr << "Failed to create connection: " << e.what() << std::endl;
                }
            }
        }

        // Destructor to clean up remaining connections
        ~ConnectionPool() {
            while (!pool_.empty()) {
                auto conn = pool_.front();
                pool_.pop();
                if (close_conn_) {
                    close_conn_(conn);
                }
            }
        }

        // Acquire a connection from the pool
        ConnectionType *acquire() {
            std::unique_lock<std::mutex> lock(mutex_);

            // Wait until a connection is available
            cond_var_.wait(lock, [this] {
                return !pool_.empty();
            });

            // Get and remove the connection from the pool
            auto conn = pool_.front();
            pool_.pop();
            return conn;
        }

        // Release a connection back to the pool
        void release(ConnectionType *conn) {
            std::lock_guard<std::mutex> lock(mutex_);
            pool_.push(conn);
            cond_var_.notify_one();
        }

        // Check if the pool is empty
        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return pool_.empty();
        }

    private:
        std::string conn_str_;
        std::queue<ConnectionType *> pool_;
        std::mutex mutex_;
        std::condition_variable cond_var_;
        CreateFunc create_conn_;
        CloseFunc close_conn_;
    };
}

#endif // TLDR_CPP_CONNECTION_POOL_H

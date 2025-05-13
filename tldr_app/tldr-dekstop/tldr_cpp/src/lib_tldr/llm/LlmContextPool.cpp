#include "LlmContextPool.h"
#include <iostream>

namespace tldr {

LlmContextPool::LlmContextPool(llama_model* model, size_t initial_size, size_t max_size, const llama_context_params& ctx_params)
    : model_(model), max_size_(max_size), ctx_params_(ctx_params) {
    
    // Create initial contexts
    for (size_t i = 0; i < initial_size; ++i) {
        llama_context* ctx = create_context();
        if (ctx) {
            available_contexts_.push(ctx);
            all_contexts_.push_back(ctx);
        }
    }
}

LlmContextPool::~LlmContextPool() {
    clear();
}

std::shared_ptr<ContextHandle> LlmContextPool::acquire_context() {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Wait for an available context if none are available and we're at max capacity
    while (available_contexts_.empty() && all_contexts_.size() >= max_size_) {
        cv_.wait(lock);
    }
    
    llama_context* ctx = nullptr;
    
    // If we have an available context, use it
    if (!available_contexts_.empty()) {
        ctx = available_contexts_.front();
        available_contexts_.pop();
    } else {
        // Otherwise create a new one
        ctx = create_context();
        if (ctx) {
            all_contexts_.push_back(ctx);
        }
    }
    
    if (!ctx) {
        std::cerr << "Failed to acquire context from pool" << std::endl;
        return nullptr;
    }
    
    // Clear the KV cache before reusing the context
    llama_kv_cache_clear(ctx);
    
    return std::make_shared<ContextHandle>(ctx, this);
}

void LlmContextPool::release_context(llama_context* ctx) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Add the context back to the available queue
    available_contexts_.push(ctx);
    
    // Notify one waiting thread that a context is available
    cv_.notify_one();
}

void LlmContextPool::clear() {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Free all contexts
    for (auto ctx : all_contexts_) {
        if (ctx) {
            llama_free(ctx);
        }
    }
    
    // Clear the containers
    all_contexts_.clear();
    
    // Clear the queue (no easy way to clear a std::queue)
    std::queue<llama_context*> empty;
    std::swap(available_contexts_, empty);
}

llama_context* LlmContextPool::create_context() {
    llama_context* ctx = llama_new_context_with_model(model_, ctx_params_);
    if (!ctx) {
        std::cerr << "Failed to create new context" << std::endl;
    }
    return ctx;
}

// ContextHandle implementation
ContextHandle::ContextHandle(llama_context* ctx, LlmContextPool* pool)
    : ctx_(ctx), pool_(pool) {
}

ContextHandle::~ContextHandle() {
    if (ctx_ && pool_) {
        pool_->release_context(ctx_);
    }
}

ContextHandle::ContextHandle(ContextHandle&& other) noexcept
    : ctx_(other.ctx_), pool_(other.pool_) {
    other.ctx_ = nullptr;
    other.pool_ = nullptr;
}

ContextHandle& ContextHandle::operator=(ContextHandle&& other) noexcept {
    if (this != &other) {
        if (ctx_ && pool_) {
            pool_->release_context(ctx_);
        }
        
        ctx_ = other.ctx_;
        pool_ = other.pool_;
        
        other.ctx_ = nullptr;
        other.pool_ = nullptr;
    }
    return *this;
}

llama_context* ContextHandle::get() const {
    return ctx_;
}

} // namespace tldr

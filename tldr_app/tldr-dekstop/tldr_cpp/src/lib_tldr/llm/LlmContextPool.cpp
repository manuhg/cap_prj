#include "LlmContextPool.h"
#include <iostream>
#include <algorithm>

namespace tldr {

LlmContextPool::LlmContextPool(llama_model* model, size_t initial_size, size_t max_size, const llama_context_params& ctx_params, size_t max_uses)
    : model_(model), max_size_(max_size), max_uses_(max_uses), ctx_params_(ctx_params) {
    
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

llama_context* LlmContextPool::createContext() {
    llama_context *ctx = create_context();
    if (ctx) {
        std::cout<<"Creating new LLM context ... "<<std::endl;
        all_contexts_.push_back(ctx);
        // Initialize usage count for new context
        context_uses_[ctx] = 0;
    }
    return ctx;
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
        ctx = createContext();
    }

    if (!ctx) {
        std::cerr << "Failed to acquire context from pool" << std::endl;
        return nullptr;
    }
    
    // Increment usage count
    context_uses_[ctx]++;
    
    return std::make_shared<ContextHandle>(ctx, this);
}

void LlmContextPool::release_context(llama_context* ctx) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Check if context should be destroyed based on usage count
    if (max_uses_ > 0 && context_uses_[ctx] >= max_uses_) {
        // Remove from all_contexts_
        all_contexts_.erase(std::remove(all_contexts_.begin(), all_contexts_.end(), ctx), all_contexts_.end());
        
        // Remove from usage tracking
        context_uses_.erase(ctx);
        
        // Free the context
        llama_free(ctx);
        
        std::cerr << "Destroyed context after " << max_uses_ << " uses" << std::endl;
        if (max_uses_==1 && all_contexts_.size() < max_size_) {
            createContext();
        }
    } else {
        // Add the context back to the available queue
        available_contexts_.push(ctx);
    }
    
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
    
    // Clear the usage tracking
    context_uses_.clear();
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

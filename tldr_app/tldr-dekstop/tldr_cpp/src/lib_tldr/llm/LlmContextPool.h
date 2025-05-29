#ifndef LLM_CONTEXT_POOL_H
#define LLM_CONTEXT_POOL_H

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <unordered_map>
#include "llama.h"

namespace tldr {

// Forward declarations
class ContextHandle;

/**
 * A pool of llama contexts that can be reused
 */
class LlmContextPool {
public:
    /**
     * Create a new context pool for a given model
     * @param model The model to create contexts for
     * @param initial_size Initial number of contexts to create
     * @param max_size Maximum number of contexts in the pool
     * @param ctx_params Parameters for context creation
     */
    LlmContextPool(llama_model* model, size_t initial_size, size_t max_size, const llama_context_params& ctx_params);
    
    /**
     * Destructor - frees all contexts
     */
    ~LlmContextPool();
    llama_context *createContext();

    /**
     * Get a context from the pool, creating a new one if necessary
     * @return A handle to a context that will be returned to the pool when it goes out of scope
     */
    std::shared_ptr<ContextHandle> acquire_context();
    
    /**
     * Return a context to the pool
     * @param ctx The context to return
     */
    void release_context(llama_context* ctx);
    
    /**
     * Clear all contexts from the pool
     */
    void clear();

private:
    llama_model* model_;
    size_t max_size_;
    llama_context_params ctx_params_;
    
    std::vector<llama_context*> all_contexts_; // All contexts created by this pool
    std::queue<llama_context*> available_contexts_; // Contexts available for use
    
    std::mutex mutex_;
    std::condition_variable cv_;
    
    /**
     * Create a new context
     * @return A new context
     */
    llama_context* create_context();
};

/**
 * A handle to a context that will be returned to the pool when it goes out of scope
 */
class ContextHandle {
public:
    ContextHandle(llama_context* ctx, LlmContextPool* pool);
    ~ContextHandle();
    
    // Prevent copying
    ContextHandle(const ContextHandle&) = delete;
    ContextHandle& operator=(const ContextHandle&) = delete;
    
    // Allow moving
    ContextHandle(ContextHandle&& other) noexcept;
    ContextHandle& operator=(ContextHandle&& other) noexcept;
    
    /**
     * Get the underlying context
     * @return The context
     */
    llama_context* get() const;

private:
    llama_context* ctx_;
    LlmContextPool* pool_;
};

} // namespace tldr

#endif // LLM_CONTEXT_POOL_H

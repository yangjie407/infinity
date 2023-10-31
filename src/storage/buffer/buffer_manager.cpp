module;

#include "concurrentqueue.h"

import stl;
import file_worker;
import third_party;
import local_file_system;
import infinity_assert;
import infinity_exception;
import buffer_obj;

module buffer_manager;

namespace infinity {
BufferManager::BufferManager(u64 memory_limit, SharedPtr<String> base_dir, SharedPtr<String> temp_dir)
    : memory_limit_(memory_limit), base_dir_(Move(base_dir)), temp_dir_(Move(temp_dir)), current_memory_size_(0) {
    LocalFileSystem fs;
    if (!fs.Exists(*base_dir_)) {
        fs.CreateDirectory(*base_dir_);
    }
    if (!fs.Exists(*temp_dir_)) {
        fs.CreateDirectory(*temp_dir_);
    }
}

BufferObj *BufferManager::Allocate(UniquePtr<FileWorker> file_worker) {
    String file_path = file_worker->GetFilePath();
    auto buffer_obj = MakeUnique<BufferObj>(this, true, Move(file_worker));

    rw_locker_.lock();
    auto [iter, insert_ok] = buffer_map_.emplace(Move(file_path), Move(buffer_obj));
    rw_locker_.unlock();

    if (!insert_ok) {
        Error<StorageException>("Buffer handle already exists. Use GET instead.", __FILE_NAME__, __LINE__);
    }
    return iter->second.get();
}

BufferObj *BufferManager::Get(UniquePtr<FileWorker> file_worker) {
    String file_path = file_worker->GetFilePath();

    rw_locker_.lock_shared();
    auto iter1 = buffer_map_.find(file_path);
    rw_locker_.unlock_shared();

    if (iter1 != buffer_map_.end()) {
        return iter1->second.get();
    }

    // Cannot find BufferHandle in buffer_map, read from disk
    auto buffer_obj = MakeUnique<BufferObj>(this, false, Move(file_worker));

    rw_locker_.lock();
    auto [iter2, insert_ok] = buffer_map_.emplace(Move(file_path), Move(buffer_obj));
    // If insert_ok is false, it means another thread has inserted the same buffer handle. Return it.
    rw_locker_.unlock();

    return iter2->second.get();
}

void BufferManager::RequestSpace(SizeT need_size, BufferObj *buffer_obj) {
    while (current_memory_size_ + need_size > memory_limit_) {
        BufferObj *buffer_obj1 = nullptr;
        if (gc_queue_.try_dequeue(buffer_obj1)) {
            if (buffer_obj == buffer_obj1) {
                Assert<StorageException>(buffer_obj1->status() == BufferStatus::kFreed, "Bug.", __FILE_NAME__, __LINE__);
                // prevent dead lock
                continue;
            }
            if (buffer_obj1->Free()) {
                current_memory_size_ -= buffer_obj1->GetBufferSize();
            }
        } else {
            throw StorageException("Out of memory.");
        }
    }
    current_memory_size_ += need_size;
}

void BufferManager::PushGCQueue(BufferObj *buffer_obj) {
    // gc_queue_ is lock-free. No lock is needed.
    gc_queue_.enqueue(buffer_obj);
}

} // namespace infinity
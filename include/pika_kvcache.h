// Copyright (c) 2024, PikiwiDB Contributors.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIKA_KVCACHE_H_
#define PIKA_KVCACHE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "include/pika_command.h"
#include "storage/storage.h"

namespace pikiwidb {

// KV cache page header structure (32 bytes) - Page-oriented design
struct KVCachePageHeader {
  uint32_t magic;           // 'PVKP' (PikiwiDB vLLM KV Page) (4 bytes)
  uint8_t version;          // Format version (1 byte)
  uint8_t dtype;            // 1=FP16, 2=FP32, 3=INT8 (1 byte)
  uint16_t page_size;       // Page size (tokens per page, e.g. 128) (2 bytes)
  uint16_t head_dim;        // Head dimension (2 bytes)
  uint16_t layer_idx;       // Layer index (2 bytes)
  uint32_t page_id;         // Page ID (4 bytes)
  uint8_t head_idx;         // Head index (1 byte)
  uint8_t kv_type;          // 0=K, 1=V (1 byte)
  uint16_t reserved1;       // Reserved (2 bytes)
  uint32_t timestamp;       // Timestamp for TTL/GC (4 bytes)
  uint32_t data_size;       // Actual data size in bytes (4 bytes)
  uint32_t crc32;           // CRC32 checksum (4 bytes)
};

static_assert(sizeof(KVCachePageHeader) == 32, "KVCachePageHeader must be 32 bytes");

// KV cache page constants
constexpr uint32_t KVCACHE_PAGE_MAGIC = 0x50564B50;  // 'PVKP'
constexpr uint8_t KVCACHE_PAGE_VERSION = 1;

// Data type constants
enum class KVCacheDataType : uint8_t {
  FP16 = 1,
  FP32 = 2,
  INT8 = 3
};

// Page-oriented key builder for vLLM PagedAttention
class KVCachePageKeyBuilder {
public:
  // Build key for a single KV page: kv:<req_id>:<layer>:<head>:<page_id>:<kv_type>
  static std::string BuildPageKey(const std::string& req_id,
                                 uint16_t layer_idx,
                                 uint8_t head_idx,
                                 uint32_t page_id,
                                 uint8_t kv_type);  // 0=K, 1=V
  
  // Build batch key prefix for multiple pages: kv:<req_id>:<layer>:*
  static std::string BuildBatchKeyPrefix(const std::string& req_id,
                                        uint16_t layer_idx);
  
  // Build TTL index key for request cleanup: ttl:<req_id>
  static std::string BuildTTLKey(const std::string& req_id);
  
  // Parse page key and extract components
  static bool ParsePageKey(const std::string& key,
                          std::string& req_id,
                          uint16_t& layer_idx,
                          uint8_t& head_idx,
                          uint32_t& page_id,
                          uint8_t& kv_type);
};

// KV cache page serializer/deserializer for vLLM PagedAttention
class KVCachePage {
public:
  KVCachePage() = default;
  
  // Create page from tensor data (FP16/FP32 raw bytes)
  bool CreatePage(const std::string& req_id,
                  uint16_t layer_idx,
                  uint8_t head_idx,
                  uint32_t page_id,
                  uint8_t kv_type,
                  uint8_t dtype,
                  uint16_t page_size,
                  uint16_t head_dim,
                  const void* tensor_data,
                  size_t data_size);
  
  // Parse page from serialized string
  bool ParsePage(const std::string& serialized_data);
  
  // Get serialized page data
  std::string GetSerializedData() const;
  
  // Get raw tensor data pointer (for zero-copy operations)
  const uint8_t* GetTensorData() const { return tensor_data_.data(); }
  size_t GetTensorDataSize() const { return tensor_data_.size(); }
  
  // Getters
  const KVCachePageHeader& GetHeader() const { return header_; }
  
  // Validate page integrity
  bool ValidatePage() const;
  
  // Static helper: Calculate tensor data size
  static size_t CalculateTensorSize(uint8_t dtype, uint16_t page_size, uint16_t head_dim);

private:
  KVCachePageHeader header_;
  std::vector<uint8_t> tensor_data_;
  
  uint32_t CalculateCRC32(const void* data, size_t size) const;
};

}  // namespace pikiwidb

// Page-oriented KV cache commands for vLLM PagedAttention
class KVPageSetCmd : public Cmd {
public:
  KVPageSetCmd(const std::string& name, int arity, uint32_t flag)
      : Cmd(name, arity, flag, 0) {}
  
  std::vector<std::string> current_key() const override {
    std::vector<std::string> res;
    res.push_back(key_);
    return res;
  }
  
  void Do() override;
  void DoThroughDB() override;
  void DoUpdateCache() override;
  void Split(const HintKeys& hint_keys) override {};
  void Merge() override {};
  Cmd* Clone() override { return new KVPageSetCmd(*this); }

private:
  std::string key_;
  std::string req_id_;
  uint16_t layer_idx_;
  uint8_t head_idx_;
  uint32_t page_id_;
  uint8_t kv_type_;
  uint8_t dtype_;
  uint16_t page_size_;
  uint16_t head_dim_;
  uint32_t ttl_seconds_;
  std::string tensor_data_;
  
  void DoInitial() override;
  rocksdb::Status s_;
};

class KVPageGetCmd : public Cmd {
public:
  KVPageGetCmd(const std::string& name, int arity, uint32_t flag)
      : Cmd(name, arity, flag, 0) {}
  
  std::vector<std::string> current_key() const override {
    std::vector<std::string> res;
    res.push_back(key_);
    return res;
  }
  
  void Do() override;
  void DoThroughDB() override;
  void ReadCache() override;
  void DoUpdateCache() override;
  void Split(const HintKeys& hint_keys) override {};
  void Merge() override {};
  Cmd* Clone() override { return new KVPageGetCmd(*this); }

private:
  std::string key_;
  std::string value_;
  
  void DoInitial() override;
  rocksdb::Status s_;
};

#endif  // PIKA_KVCACHE_H_

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

// ==================== KVBlock for vLLM Block-level KV Cache ====================

namespace pikiwidb {

// KVBlock key builder for vLLM block-level storage
class KVBlockKeyBuilder {
public:
  // Build key for a single KV block: kvblock:{ns}:{layer_id}:{block_id}:{k_type}
  static std::string BuildBlockKey(const std::string& ns,
                                  uint16_t layer_id,
                                  uint32_t block_id,
                                  uint8_t k_type);  // 0=K, 1=V
  
  // Parse block key and extract components
  static bool ParseBlockKey(const std::string& key,
                           std::string& ns,
                           uint16_t& layer_id,
                           uint32_t& block_id,
                           uint8_t& k_type);
};

// Simple KVBlock for storing raw K/V tensor data
class KVBlock {
public:
  KVBlock() = default;
  
  // Create block from raw tensor data
  bool CreateBlock(const void* tensor_data, size_t data_size);
  
  // Parse block from serialized string (just raw data)
  bool ParseBlock(const std::string& serialized_data);
  
  // Get serialized block data (just raw tensor bytes)
  std::string GetSerializedData() const;
  
  // Get raw tensor data pointer
  const uint8_t* GetTensorData() const { return tensor_data_.data(); }
  size_t GetTensorDataSize() const { return tensor_data_.size(); }

private:
  std::vector<uint8_t> tensor_data_;
};

}  // namespace pikiwidb

// Block-level KV cache commands for vLLM
class KVBlockSetCmd : public Cmd {
public:
  KVBlockSetCmd(const std::string& name, int arity, uint32_t flag)
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
  Cmd* Clone() override { return new KVBlockSetCmd(*this); }

private:
  std::string key_;
  std::string ns_;
  uint16_t layer_id_;
  uint32_t block_id_;
  uint8_t k_type_;
  std::string tensor_data_;
  
  void DoInitial() override;
  rocksdb::Status s_;
};

class KVBlockGetCmd : public Cmd {
public:
  KVBlockGetCmd(const std::string& name, int arity, uint32_t flag)
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
  Cmd* Clone() override { return new KVBlockGetCmd(*this); }

private:
  std::string key_;
  std::string value_;
  
  void DoInitial() override;
  rocksdb::Status s_;
};

class KVBlockMSetCmd : public Cmd {
public:
  KVBlockMSetCmd(const std::string& name, int arity, uint32_t flag)
      : Cmd(name, arity, flag, 0) {}
  
  std::vector<std::string> current_key() const override {
    return keys_;
  }
  
  void Do() override;
  void DoThroughDB() override;
  void DoUpdateCache() override;
  void Split(const HintKeys& hint_keys) override {};
  void Merge() override {};
  Cmd* Clone() override { return new KVBlockMSetCmd(*this); }

private:
  struct BlockData {
    std::string key;
    std::string ns;
    uint16_t layer_id;
    uint32_t block_id;
    uint8_t k_type;
    std::string tensor_data;
  };
  
  std::vector<std::string> keys_;
  std::vector<BlockData> blocks_;
  
  void DoInitial() override;
  std::vector<rocksdb::Status> statuses_;
};

class KVBlockMGetCmd : public Cmd {
public:
  KVBlockMGetCmd(const std::string& name, int arity, uint32_t flag)
      : Cmd(name, arity, flag, 0) {}
  
  std::vector<std::string> current_key() const override {
    return keys_;
  }
  
  void Do() override;
  void ReadCache() override;
  void DoThroughDB() override;
  void DoUpdateCache() override;
  void Split(const HintKeys& hint_keys) override {};
  void Merge() override {};
  Cmd* Clone() override { return new KVBlockMGetCmd(*this); }

private:
  std::vector<std::string> keys_;
  std::vector<std::string> values_;
  
  void DoInitial() override;
  std::vector<rocksdb::Status> statuses_;
};

class KVBlockExistsCmd : public Cmd {
public:
  KVBlockExistsCmd(const std::string& name, int arity, uint32_t flag)
      : Cmd(name, arity, flag, 0) {}
  
  std::vector<std::string> current_key() const override {
    std::vector<std::string> res;
    res.push_back(key_);
    return res;
  }
  
  void Do() override;
  void ReadCache() override;
  void DoThroughDB() override;
  void Split(const HintKeys& hint_keys) override {};
  void Merge() override {};
  Cmd* Clone() override { return new KVBlockExistsCmd(*this); }

private:
  std::string key_;
  
  void DoInitial() override;
  rocksdb::Status s_;
};

#endif  // PIKA_KVCACHE_H_

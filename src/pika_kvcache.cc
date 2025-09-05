#include "include/pika_kvcache.h"

#include <algorithm>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "include/pika_db.h"
#include "src/storage/include/storage/storage.h"
#include "pstd/include/pstd_string.h"

namespace pikiwidb {

// KVBlock key builder implementations
std::string KVBlockKeyBuilder::BuildBlockKey(const std::string& ns,
                                            uint16_t layer_id,
                                            uint32_t block_hash,
                                            uint8_t k_type) {
  std::ostringstream oss;
  oss << "kvblock:" << ns << ":" << layer_id << ":" << block_hash << ":" << static_cast<int>(k_type);
  return oss.str();
}

bool KVBlockKeyBuilder::ParseBlockKey(const std::string& key,
                                     std::string& ns,
                                     uint16_t& layer_id,
                                     uint32_t& block_hash,
                                     uint8_t& k_type) {
  // Parse key format: kvblock:{ns}:{layer_id}:{block_hash}:{k_type}
  std::vector<std::string> parts;
  std::string current;
  
  for (char c : key) {
    if (c == ':') {
      parts.push_back(current);
      current.clear();
    } else {
      current += c;
    }
  }
  if (!current.empty()) {
    parts.push_back(current);
  }
  
  if (parts.size() != 5 || parts[0] != "kvblock") {
    return false;
  }
  
  try {
    ns = parts[1];
    layer_id = static_cast<uint16_t>(std::stoul(parts[2]));
    block_hash = static_cast<uint32_t>(std::stoul(parts[3]));
    k_type = static_cast<uint8_t>(std::stoul(parts[4]));
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// KVBlock implementations
bool KVBlock::CreateBlock(const void* tensor_data, size_t data_size) {
  tensor_data_.resize(data_size);
  if (data_size > 0 && tensor_data != nullptr) {
    std::memcpy(tensor_data_.data(), tensor_data, data_size);
  }
  return true;
}

bool KVBlock::ParseBlock(const std::string& serialized_data) {
  tensor_data_.resize(serialized_data.size());
  if (!serialized_data.empty()) {
    std::memcpy(tensor_data_.data(), serialized_data.data(), serialized_data.size());
  }
  return true;
}

std::string KVBlock::GetSerializedData() const {
  return std::string(reinterpret_cast<const char*>(tensor_data_.data()), tensor_data_.size());
}

}  // namespace pikiwidb

// ======================= KVBlockSetCmd =======================
void KVBlockSetCmd::DoInitial() {
  if (!CheckArg(argv_.size())) {
    res_.SetRes(CmdRes::kWrongNum, "kvblockset");
    return;
  }
  
  // KVBLOCKSET <ns> <layer_id> <block_hash> <k_type> <tensor_data>
  if (argv_.size() != 6) {
    res_.SetRes(CmdRes::kWrongNum, "kvblockset requires 5 arguments");
    return;
  }
  
  try {
    ns_ = argv_[1];
    layer_id_ = static_cast<uint16_t>(std::stoul(argv_[2]));
    block_hash_ = static_cast<uint32_t>(std::stoul(argv_[3]));
    k_type_ = static_cast<uint8_t>(std::stoul(argv_[4]));
    tensor_data_ = argv_[5];
    
    // Build the block key
    key_ = pikiwidb::KVBlockKeyBuilder::BuildBlockKey(ns_, layer_id_, block_hash_, k_type_);
    
  } catch (const std::exception& e) {
    res_.SetRes(CmdRes::kInvalidInt, "Invalid argument format");
    return;
  }
}

void KVBlockSetCmd::Do() {
  // Create KV block (just store raw data)
  pikiwidb::KVBlock block;
  if (!block.CreateBlock(tensor_data_.data(), tensor_data_.size())) {
    res_.SetRes(CmdRes::kErrOther, "Failed to create KV block");
    return;
  }
  
  std::string block_data = block.GetSerializedData();
  
  STAGE_TIMER_GUARD(storage_duration_ms, true);
  s_ = db_->storage()->Set(key_, block_data);
  
  if (s_.ok()) {
    res_.SetRes(CmdRes::kOk);
  } else {
    res_.SetRes(CmdRes::kErrOther, s_.ToString());
  }
}

void KVBlockSetCmd::DoThroughDB() {
  Do();
}

void KVBlockSetCmd::DoUpdateCache() {
  // TODO
}

// ======================= KVBlockGetCmd =======================
void KVBlockGetCmd::DoInitial() {
  if (!CheckArg(argv_.size())) {
    res_.SetRes(CmdRes::kWrongNum, "kvblockget");
    return;
  }
  
  // KVBLOCKGET <ns> <layer_id> <block_hash> <k_type>
  if (argv_.size() != 5) {
    res_.SetRes(CmdRes::kWrongNum, "kvblockget requires 4 arguments");
    return;
  }
  
  try {
    std::string ns = argv_[1];
    uint16_t layer_id = static_cast<uint16_t>(std::stoul(argv_[2]));
    uint32_t block_hash = static_cast<uint32_t>(std::stoul(argv_[3]));
    uint8_t k_type = static_cast<uint8_t>(std::stoul(argv_[4]));
    
    // Build the block key
    key_ = pikiwidb::KVBlockKeyBuilder::BuildBlockKey(ns, layer_id, block_hash, k_type);
    
  } catch (const std::exception& e) {
    res_.SetRes(CmdRes::kInvalidInt, "Invalid argument format");
    return;
  }
}

void KVBlockGetCmd::Do() {
  STAGE_TIMER_GUARD(storage_duration_ms, true);
  s_ = db_->storage()->Get(key_, &value_);
  
  if (s_.ok()) {
    // Parse and validate the block
    pikiwidb::KVBlock block;
    if (block.ParseBlock(value_)) {
      // Return the raw tensor data
      const uint8_t* tensor_data = block.GetTensorData();
      size_t tensor_size = block.GetTensorDataSize();
      res_.AppendStringLenUint64(tensor_size);
      res_.AppendContent(std::string(reinterpret_cast<const char*>(tensor_data), tensor_size));
    } else {
      res_.SetRes(CmdRes::kErrOther, "Invalid KV block format");
    }
  } else if (s_.IsNotFound()) {
    res_.AppendStringLen(-1);
  } else {
    res_.SetRes(CmdRes::kErrOther, s_.ToString());
  }
}

void KVBlockGetCmd::DoThroughDB() {
  res_.clear();
  Do();
}

void KVBlockGetCmd::ReadCache() {
  res_.SetRes(CmdRes::kCacheMiss);
}

void KVBlockGetCmd::DoUpdateCache() {
  // TODO
}

// ======================= KVBlockMSetCmd =======================
void KVBlockMSetCmd::DoInitial() {
  if (!CheckArg(argv_.size())) {
    res_.SetRes(CmdRes::kWrongNum, "kvblockmset");
    return;
  }
  
  // KVBLOCKMSET <num_blocks> <ns1> <layer_id1> <block_hash1> <k_type1> <data1> ...
  if (argv_.size() < 2) {
    res_.SetRes(CmdRes::kWrongNum, "kvblockmset requires at least 1 argument");
    return;
  }
  
  try {
    size_t num_blocks = std::stoul(argv_[1]);
    if (argv_.size() != 2 + num_blocks * 5) {
      res_.SetRes(CmdRes::kWrongNum, "kvblockmset argument count mismatch");
      return;
    }
    
    keys_.clear();
    blocks_.clear();
    keys_.reserve(num_blocks);
    blocks_.reserve(num_blocks);
    
    for (size_t i = 0; i < num_blocks; ++i) {
      size_t base_idx = 2 + i * 5;
      BlockData block_data;
      
      block_data.ns = argv_[base_idx];
      block_data.layer_id = static_cast<uint16_t>(std::stoul(argv_[base_idx + 1]));
      block_data.block_hash = static_cast<uint32_t>(std::stoul(argv_[base_idx + 2]));
      block_data.k_type = static_cast<uint8_t>(std::stoul(argv_[base_idx + 3]));
      block_data.tensor_data = argv_[base_idx + 4];
      
      block_data.key = pikiwidb::KVBlockKeyBuilder::BuildBlockKey(
          block_data.ns, block_data.layer_id, block_data.block_hash, block_data.k_type);
      
      keys_.push_back(block_data.key);
      blocks_.push_back(block_data);
    }
    
  } catch (const std::exception& e) {
    res_.SetRes(CmdRes::kInvalidInt, "Invalid argument format");
    return;
  }
}

void KVBlockMSetCmd::Do() {
  STAGE_TIMER_GUARD(storage_duration_ms, true);
  
  statuses_.clear();
  statuses_.resize(blocks_.size());
  
  // Batch set all blocks
  for (size_t i = 0; i < blocks_.size(); ++i) {
    const BlockData& block_data = blocks_[i];
    
    // Create KV block
    pikiwidb::KVBlock block;
    if (!block.CreateBlock(block_data.tensor_data.data(), block_data.tensor_data.size())) {
      statuses_[i] = rocksdb::Status::InvalidArgument("Failed to create block");
      continue;
    }
    
    std::string serialized_data = block.GetSerializedData();
    statuses_[i] = db_->storage()->Set(block_data.key, serialized_data);
  }
  
  // Check if all operations succeeded
  bool all_success = true;
  for (const auto& status : statuses_) {
    if (!status.ok()) {
      all_success = false;
      break;
    }
  }
  
  if (all_success) {
    res_.SetRes(CmdRes::kOk);
  } else {
    res_.SetRes(CmdRes::kErrOther, "Some block set operations failed");
  }
}

void KVBlockMSetCmd::DoThroughDB() {
  Do();
}

void KVBlockMSetCmd::DoUpdateCache() {
  // TODO
}

// ======================= KVBlockMGetCmd =======================
void KVBlockMGetCmd::DoInitial() {
  if (!CheckArg(argv_.size())) {
    res_.SetRes(CmdRes::kWrongNum, "kvblockmget");
    return;
  }
  
  // KVBLOCKMGET <num_keys> <ns1> <layer_id1> <block_hash1> <k_type1> ...
  if (argv_.size() < 2) {
    res_.SetRes(CmdRes::kWrongNum, "kvblockmget requires at least 1 argument");
    return;
  }
  
  try {
    size_t num_keys = std::stoul(argv_[1]);
    if (argv_.size() != 2 + num_keys * 4) {
      res_.SetRes(CmdRes::kWrongNum, "kvblockmget argument count mismatch");
      return;
    }
    
    keys_.clear();
    keys_.reserve(num_keys);
    
    for (size_t i = 0; i < num_keys; ++i) {
      size_t base_idx = 2 + i * 4;
      std::string ns = argv_[base_idx];
      uint16_t layer_id = static_cast<uint16_t>(std::stoul(argv_[base_idx + 1]));
      uint32_t block_hash = static_cast<uint32_t>(std::stoul(argv_[base_idx + 2]));
      uint8_t k_type = static_cast<uint8_t>(std::stoul(argv_[base_idx + 3]));
      
      std::string key = pikiwidb::KVBlockKeyBuilder::BuildBlockKey(ns, layer_id, block_hash, k_type);
      keys_.push_back(key);
    }
    
  } catch (const std::exception& e) {
    res_.SetRes(CmdRes::kInvalidInt, "Invalid argument format");
    return;
  }
}

void KVBlockMGetCmd::Do() {
  STAGE_TIMER_GUARD(storage_duration_ms, true);
  
  values_.clear();
  values_.resize(keys_.size());
  statuses_.clear();
  statuses_.resize(keys_.size());
  
  // Batch get all keys
  for (size_t i = 0; i < keys_.size(); ++i) {
    statuses_[i] = db_->storage()->Get(keys_[i], &values_[i]);
  }
  
  // Build response array
  res_.AppendArrayLen(keys_.size());
  
  for (size_t i = 0; i < keys_.size(); ++i) {
    if (statuses_[i].ok()) {
      // Parse and validate the block
      pikiwidb::KVBlock block;
      if (block.ParseBlock(values_[i])) {
        // Return the raw tensor data
        const uint8_t* tensor_data = block.GetTensorData();
        size_t tensor_size = block.GetTensorDataSize();
        res_.AppendStringLenUint64(tensor_size);
        res_.AppendContent(std::string(reinterpret_cast<const char*>(tensor_data), tensor_size));
      } else {
        res_.AppendStringLen(-1);  // Invalid block format
      }
    } else {
      res_.AppendStringLen(-1);  // Key not found or error
    }
  }
}

void KVBlockMGetCmd::DoThroughDB() {
  res_.clear();
  Do();
}

void KVBlockMGetCmd::ReadCache() {
  res_.SetRes(CmdRes::kCacheMiss);
}

void KVBlockMGetCmd::DoUpdateCache() {
  // TODO
}

// ======================= KVBlockExistsCmd =======================
void KVBlockExistsCmd::DoInitial() {
  if (!CheckArg(argv_.size())) {
    res_.SetRes(CmdRes::kWrongNum, "kvblockexists");
    return;
  }
  
  // KVBLOCKEXISTS <ns> <layer_id> <block_hash> <k_type>
  if (argv_.size() != 5) {
    res_.SetRes(CmdRes::kWrongNum, "kvblockexists requires 4 arguments");
    return;
  }
  
  try {
    std::string ns = argv_[1];
    uint16_t layer_id = static_cast<uint16_t>(std::stoul(argv_[2]));
    uint32_t block_hash = static_cast<uint32_t>(std::stoul(argv_[3]));
    uint8_t k_type = static_cast<uint8_t>(std::stoul(argv_[4]));
    
    // Build the block key
    key_ = pikiwidb::KVBlockKeyBuilder::BuildBlockKey(ns, layer_id, block_hash, k_type);
    
  } catch (const std::exception& e) {
    res_.SetRes(CmdRes::kInvalidInt, "Invalid argument format");
    return;
  }
}

void KVBlockExistsCmd::Do() {
  STAGE_TIMER_GUARD(storage_duration_ms, true);
  std::string value;
  s_ = db_->storage()->Get(key_, &value);
  
  if (s_.ok()) {
    res_.AppendInteger(1);  // Block exists
  } else if (s_.IsNotFound()) {
    res_.AppendInteger(0);  // Block does not exist
  } else {
    res_.SetRes(CmdRes::kErrOther, s_.ToString());
  }
}

void KVBlockExistsCmd::DoThroughDB() {
  res_.clear();
  Do();
}

void KVBlockExistsCmd::ReadCache() {
  res_.SetRes(CmdRes::kCacheMiss);
}
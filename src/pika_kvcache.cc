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

// Page-oriented key builder implementations
std::string KVCachePageKeyBuilder::BuildPageKey(const std::string& req_id,
                                               uint16_t layer_idx,
                                               uint8_t head_idx,
                                               uint32_t page_id,
                                               uint8_t kv_type) {
  std::ostringstream oss;
  oss << "kv:" << req_id << ":" << layer_idx << ":" 
      << static_cast<int>(head_idx) << ":" << page_id << ":" 
      << static_cast<int>(kv_type);
  return oss.str();
}

std::string KVCachePageKeyBuilder::BuildBatchKeyPrefix(const std::string& req_id,
                                                      uint16_t layer_idx) {
  std::ostringstream oss;
  oss << "kv:" << req_id << ":" << layer_idx << ":*";
  return oss.str();
}

std::string KVCachePageKeyBuilder::BuildTTLKey(const std::string& req_id) {
  return "ttl:" + req_id;
}

bool KVCachePageKeyBuilder::ParsePageKey(const std::string& key,
                                        std::string& req_id,
                                        uint16_t& layer_idx,
                                        uint8_t& head_idx,
                                        uint32_t& page_id,
                                        uint8_t& kv_type) {
  // Parse key format: kv:<req_id>:<layer>:<head>:<page_id>:<kv_type>
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
  
  if (parts.size() != 6 || parts[0] != "kv") {
    return false;
  }
  
  try {
    req_id = parts[1];
    layer_idx = static_cast<uint16_t>(std::stoul(parts[2]));
    head_idx = static_cast<uint8_t>(std::stoul(parts[3]));
    page_id = static_cast<uint32_t>(std::stoul(parts[4]));
    kv_type = static_cast<uint8_t>(std::stoul(parts[5]));
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// KV cache page implementations
bool KVCachePage::CreatePage(const std::string& req_id,
                            uint16_t layer_idx,
                            uint8_t head_idx,
                            uint32_t page_id,
                            uint8_t kv_type,
                            uint8_t dtype,
                            uint16_t page_size,
                            uint16_t head_dim,
                            const void* tensor_data,
                            size_t data_size) {
  // Initialize header
  header_.magic = KVCACHE_PAGE_MAGIC;
  header_.version = KVCACHE_PAGE_VERSION;
  header_.dtype = dtype;
  header_.page_size = page_size;
  header_.head_dim = head_dim;
  header_.layer_idx = layer_idx;
  header_.page_id = page_id;
  header_.head_idx = head_idx;
  header_.kv_type = kv_type;
  header_.timestamp = static_cast<uint32_t>(std::time(nullptr));
  header_.data_size = static_cast<uint32_t>(data_size);
  
  // Copy tensor data
  tensor_data_.resize(data_size);
  if (data_size > 0 && tensor_data != nullptr) {
    std::memcpy(tensor_data_.data(), tensor_data, data_size);
  }
  
  // Calculate CRC32
  header_.crc32 = CalculateCRC32(tensor_data_.data(), tensor_data_.size());
  
  return true;
}

bool KVCachePage::ParsePage(const std::string& serialized_data) {
  if (serialized_data.size() < sizeof(KVCachePageHeader)) {
    return false;
  }
  
  // Parse header
  std::memcpy(&header_, serialized_data.data(), sizeof(KVCachePageHeader));
  
  // Validate magic and version
  if (header_.magic != KVCACHE_PAGE_MAGIC || header_.version != KVCACHE_PAGE_VERSION) {
    return false;
  }
  
  // Extract tensor data
  size_t expected_data_size = serialized_data.size() - sizeof(KVCachePageHeader);
  if (expected_data_size != header_.data_size) {
    return false;
  }
  
  tensor_data_.resize(expected_data_size);
  if (expected_data_size > 0) {
    std::memcpy(tensor_data_.data(), 
                serialized_data.data() + sizeof(KVCachePageHeader), 
                expected_data_size);
  }
  
  return ValidatePage();
}

std::string KVCachePage::GetSerializedData() const {
  std::string result;
  result.resize(sizeof(KVCachePageHeader) + tensor_data_.size());
  
  // Copy header
  std::memcpy(result.data(), &header_, sizeof(KVCachePageHeader));
  
  // Copy tensor data
  if (!tensor_data_.empty()) {
    std::memcpy(result.data() + sizeof(KVCachePageHeader), 
                tensor_data_.data(), tensor_data_.size());
  }
  
  return result;
}

bool KVCachePage::ValidatePage() const {
  // Validate CRC32
  uint32_t calculated_crc = CalculateCRC32(tensor_data_.data(), tensor_data_.size());
  return calculated_crc == header_.crc32;
}

size_t KVCachePage::CalculateTensorSize(uint8_t dtype, uint16_t page_size, uint16_t head_dim) {
  size_t element_size;
  switch (dtype) {
    case static_cast<uint8_t>(KVCacheDataType::FP16):
      element_size = 2;
      break;
    case static_cast<uint8_t>(KVCacheDataType::FP32):
      element_size = 4;
      break;
    case static_cast<uint8_t>(KVCacheDataType::INT8):
      element_size = 1;
      break;
    default:
      return 0;
  }
  return page_size * head_dim * element_size;
}

uint32_t KVCachePage::CalculateCRC32(const void* data, size_t size) const {
  // Simple CRC32 implementation
  uint32_t crc = 0xFFFFFFFF;
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  
  for (size_t i = 0; i < size; ++i) {
    crc ^= bytes[i];
    for (int j = 0; j < 8; ++j) {
      if (crc & 1) {
        crc = (crc >> 1) ^ 0xEDB88320;
      } else {
        crc >>= 1;
      }
    }
  }
  
  return ~crc;
}

}  // namespace pikiwidb

// ======================= KVPageSetCmd =======================
void KVPageSetCmd::DoInitial() {
  if (!CheckArg(argv_.size())) {
    res_.SetRes(CmdRes::kWrongNum, "kvpageset");
    return;
  }
  
  // KVPAGESET <req_id> <layer> <head> <page_id> <kv_type> <dtype> <page_size> <head_dim> <ttl> <tensor_data>
  if (argv_.size() != 11) {
    res_.SetRes(CmdRes::kWrongNum, "kvpageset requires 10 arguments");
    return;
  }
  
  try {
    req_id_ = argv_[1];
    layer_idx_ = static_cast<uint16_t>(std::stoul(argv_[2]));
    head_idx_ = static_cast<uint8_t>(std::stoul(argv_[3]));
    page_id_ = static_cast<uint32_t>(std::stoul(argv_[4]));
    kv_type_ = static_cast<uint8_t>(std::stoul(argv_[5]));
    dtype_ = static_cast<uint8_t>(std::stoul(argv_[6]));
    page_size_ = static_cast<uint16_t>(std::stoul(argv_[7]));
    head_dim_ = static_cast<uint16_t>(std::stoul(argv_[8]));
    ttl_seconds_ = static_cast<uint32_t>(std::stoul(argv_[9]));
    tensor_data_ = argv_[10];
    
    // Build the page key
    key_ = pikiwidb::KVCachePageKeyBuilder::BuildPageKey(req_id_, layer_idx_, 
                                                        head_idx_, page_id_, kv_type_);
    
  } catch (const std::exception& e) {
    res_.SetRes(CmdRes::kInvalidInt, "Invalid argument format");
    return;
  }
}

void KVPageSetCmd::Do() {
  // Create KV cache page
  pikiwidb::KVCachePage page;
  if (!page.CreatePage(req_id_, layer_idx_, head_idx_, page_id_, kv_type_,
                      dtype_, page_size_, head_dim_, 
                      tensor_data_.data(), tensor_data_.size())) {
    res_.SetRes(CmdRes::kErrOther, "Failed to create KV cache page");
    return;
  }
  
  std::string page_data = page.GetSerializedData();
  
  STAGE_TIMER_GUARD(storage_duration_ms, true);
  s_ = db_->storage()->Set(key_, page_data);
  
  if (s_.ok()) {
    res_.SetRes(CmdRes::kOk);
    
    // Set TTL if specified
    if (ttl_seconds_ > 0) {
      // Note: PikiwiDB doesn't have native TTL, this is a placeholder
      // In production, you might want to use a separate TTL mechanism
    }
  } else {
    res_.SetRes(CmdRes::kErrOther, s_.ToString());
  }
}

void KVPageSetCmd::DoThroughDB() {
  Do();
}

void KVPageSetCmd::DoUpdateCache() {
    // TODO
}


// ======================= KVPageGetCmd =======================
void KVPageGetCmd::DoInitial() {
  LOG(INFO) << "KVPageGetCmd::DoInitial() - argc: " << argv_.size();
  LOG(INFO) << "KVPageGetCmd::DoInitial() - arity_: " << arity_;
  
  bool check_result = CheckArg(argv_.size());
  LOG(INFO) << "KVPageGetCmd::DoInitial() - CheckArg result: " << check_result;
  
  if (!check_result) {
    LOG(ERROR) << "KVPageGetCmd::DoInitial() - CheckArg failed, argc=" << argv_.size() << ", arity=" << arity_;
    res_.SetRes(CmdRes::kWrongNum, "kvpageget");
    return;
  }
  
  // KVPAGEGET <req_id> <layer> <head> <page_id> <kv_type>
  if (argv_.size() != 6) {
    LOG(ERROR) << "KVPageGetCmd::DoInitial() - Wrong argc: " << argv_.size() << ", expected 6";
    res_.SetRes(CmdRes::kWrongNum, "kvpageget requires 5 arguments");
    return;
  }
  
  try {
    std::string req_id = argv_[1];
    uint16_t layer_idx = static_cast<uint16_t>(std::stoul(argv_[2]));
    uint8_t head_idx = static_cast<uint8_t>(std::stoul(argv_[3]));
    uint32_t page_id = static_cast<uint32_t>(std::stoul(argv_[4]));
    uint8_t kv_type = static_cast<uint8_t>(std::stoul(argv_[5]));
    
    // Build the page key
    key_ = pikiwidb::KVCachePageKeyBuilder::BuildPageKey(req_id, layer_idx, 
                                                        head_idx, page_id, kv_type);
    
    LOG(INFO) << "KVPageGetCmd::DoInitial() - Built key: " << key_;
    
  } catch (const std::exception& e) {
    LOG(ERROR) << "KVPageGetCmd::DoInitial() - Exception: " << e.what();
    res_.SetRes(CmdRes::kInvalidInt, "Invalid argument format");
    return;
  }
  
  LOG(INFO) << "KVPageGetCmd::DoInitial() - Success";
}

void KVPageGetCmd::Do() {
  LOG(INFO) << "KVPageGetCmd::Do() - Start";
  STAGE_TIMER_GUARD(storage_duration_ms, true);
  s_ = db_->storage()->Get(key_, &value_);

  LOG(INFO) << "KVPageGetCmd::Do() - Key: " << key_;
  LOG(INFO) << "KVPageGetCmd::Do() - Storage result: " << s_.ToString();
  
  if (s_.ok()) {
    LOG(INFO) << "KVPageGetCmd::Do() - Found data, size: " << value_.size();
    // Parse and validate the page
    pikiwidb::KVCachePage page;

    if (page.ParsePage(value_)) {
      // Return the raw tensor data (without header)
      LOG(INFO) << "KVPageGetCmd::Do() - Page parsed successfully";
      const uint8_t* tensor_data = page.GetTensorData();
      size_t tensor_size = page.GetTensorDataSize();
      LOG(INFO) << "KVPageGetCmd::Do() - Returning tensor data, size: " << tensor_size;
      res_.AppendStringLenUint64(tensor_size);
      res_.AppendContent(std::string(reinterpret_cast<const char*>(tensor_data), tensor_size));
    } else {
      LOG(ERROR) << "KVPageGetCmd::Do() - Invalid KV cache page format";
      res_.SetRes(CmdRes::kErrOther, "Invalid KV cache page format");
    }
  } else if (s_.IsNotFound()) {
    LOG(INFO) << "KVPageGetCmd::Do() - Key not found, returning null";
    res_.AppendStringLen(-1);
  } else {
    LOG(ERROR) << "KVPageGetCmd::Do() - Storage error: " << s_.ToString();
    res_.SetRes(CmdRes::kErrOther, s_.ToString());
  }
  
  LOG(INFO) << "KVPageGetCmd::Do() - End";
}

void KVPageGetCmd::DoThroughDB() {
  res_.clear();
  Do();
}

void KVPageGetCmd::ReadCache() {
  // TODO
  res_.SetRes(CmdRes::kCacheMiss);
}

void KVPageGetCmd::DoUpdateCache() {
  // TODO
}
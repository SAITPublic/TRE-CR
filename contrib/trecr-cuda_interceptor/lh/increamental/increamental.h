#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <memory.h>
#include <dirent.h>
#include <iostream>

#include <unordered_map>
#include <string>
#include <fstream>
#include <set>
#include "procmapsarea.h"
#include "dmtcpalloc.h"
#include "util.h"

using namespace dmtcp;

// struct gpu_status {
//   void * mem_addr;
//   size_t mem_len;
//   std::string hash;
//   bool is_changed;
//   Area* area_ptr;

//   gpu_status(void * mem_addr, size_t mem_len, std::string hash)
//         : mem_addr(mem_addr),
//           mem_len(mem_len),
//           hash(hash),
//           is_changed(false),
//           area_ptr(nullptr) {}
// };

class Increamental
{
public:
  static bool use_increamental;

  static std::unordered_map<void *, void *> gpu_cpu_map; // 用来map gpu和cpu mem

  static std::unordered_map<void *, std::string> address_hash_map; // 用来比较hash

  static std::unordered_map<void *, size_t> address_size_map; // 用来记录大小

  static std::unordered_map<void *, bool> address_change_map; // 用来记录本轮是否改变

  static std::set<void *> mem_set; // 用来记录mem是否存在

  static std::string ckpt_filename;

  bool UpdateGpuStatus(void *address, size_t len, std::string hash);

  void WriteMemMaptoFile();

  void ReadMemMapfromFile();

  int GetCkptFile(void *addr, pid_t pid);

  void ReadCkptFile(std::vector<int> &gpu_fd_v, pid_t pid);

  void ReadCkptFile(pid_t pid);

  void FreeAllCpuMem();
};

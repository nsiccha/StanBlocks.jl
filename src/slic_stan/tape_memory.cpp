// The reverse-mode tape memory of a StanBlocks model library, and the C
// functions that hand it back to the operating system. StanBlocks links this
// file into every BridgeStan library it builds (`_artifact_library` in
// build.jl), as one more source of the link step, so it is compiled with the
// model's own flags. `tape_memory.jl` calls these functions from Julia.
//
// Stan Math records a gradient on the calling thread's arena: the `memalloc_`
// (a `stack_alloc`) of the thread's `ChainableStack`. `recover_memory()` rewinds
// the arena after every gradient and keeps all of its blocks, each twice the
// size of the one before, so a thread holds, for as long as it lives, about
// twice the largest tape it has recorded. Each model library carries its own
// copy of Stan Math and so its own arena per thread. BridgeStan's C API has no
// call that returns this memory.
//
// Nothing here changes how Stan allocates. A model library also links
// BridgeStan's own object, which carries Stan Math's inline arena code
// compiled without this file, so both copies must stay the same. A release
// instead tells the kernel that the pages of the arena's existing blocks hold
// nothing worth keeping: the blocks stay allocated and owned by the arena, and
// the next gradient writes into them again.
//
// Every function acts on the CALLING thread's arena in THIS library only.

#include <stan/math/rev/core.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#define STANBLOCKS_TAPE_PAGES 1
#else
#define STANBLOCKS_TAPE_PAGES 0
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#define STANBLOCKS_TAPE_API extern "C" __declspec(dllexport)
#else
#define STANBLOCKS_TAPE_API extern "C" __attribute__((visibility("default")))
#endif

namespace stanblocks_tape {

// The arena keeps its block pointers and block sizes in private members. The
// names in an explicit instantiation are exempt from access checking, so
// instantiating `exposed` with a pointer to such a member defines a friend
// function that returns it. A Stan Math that renames these members fails to
// compile this file, and with it the library build, naming this file.
template <typename Tag, typename Tag::type Member>
struct exposed {
  friend typename Tag::type member_pointer(Tag) { return Member; }
};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif
struct block_list {
  using type = std::vector<char*> stan::math::stack_alloc::*;
  friend type member_pointer(block_list);
};
struct block_sizes {
  using type = std::vector<std::size_t> stan::math::stack_alloc::*;
  friend type member_pointer(block_sizes);
};
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

template struct exposed<block_list, &stan::math::stack_alloc::blocks_>;
template struct exposed<block_sizes, &stan::math::stack_alloc::sizes_>;

using tape_t = std::remove_pointer_t<decltype(stan::math::ChainableStack::instance_)>;

inline const std::vector<char*>& blocks(const tape_t& tape) {
  return tape.memalloc_.*member_pointer(block_list());
}
inline const std::vector<std::size_t>& sizes(const tape_t& tape) {
  return tape.memalloc_.*member_pointer(block_sizes());
}

// Whether a release is pending its reuse on this thread (one tape per thread
// with STAN_THREADS, one per process without, as the tape itself).
static STAN_THREADS_DEF bool released = false;

// Nothing is recorded on the tape and nothing is allocated on its arena: no
// gradient (nested or not) is in progress on this thread.
inline bool idle(const tape_t& tape) {
  return tape.var_stack_.empty() && tape.var_nochain_stack_.empty()
         && tape.var_alloc_stack_.empty() && tape.nested_var_stack_sizes_.empty()
         && !tape.memalloc_.in_stack(blocks(tape).front());
}

enum class advice { discard, reuse };

// The whole pages inside [begin, begin + bytes), so that nothing outside the
// allocation, such as an allocator's header, is touched. Linux discards them
// at once with MADV_DONTNEED and needs no reuse. macOS charges a page to the
// process until it is told it is reusable (MADV_FREE_REUSABLE); a page
// written again after that is charged again only once it is told so
// (MADV_FREE_REUSE).
inline void advise(const void* begin, std::size_t bytes, advice what) noexcept {
#if STANBLOCKS_TAPE_PAGES
  if (begin == nullptr || bytes == 0) {
    return;
  }
  const auto page = static_cast<std::uintptr_t>(::sysconf(_SC_PAGESIZE));
  const auto start = reinterpret_cast<std::uintptr_t>(begin);
  const std::uintptr_t first = (start + page - 1) / page * page;
  const std::uintptr_t end = (start + bytes) / page * page;
  if (end <= first) {
    return;
  }
  void* pages = reinterpret_cast<void*>(first);
#if defined(__APPLE__)
  ::madvise(pages, end - first, what == advice::discard ? MADV_FREE_REUSABLE : MADV_FREE_REUSE);
#else
  if (what == advice::discard) {
    ::madvise(pages, end - first, MADV_DONTNEED);
  }
#endif
#endif
}

// The storage a pointer list keeps beyond its elements: all of it once the
// list is empty.
template <typename T>
inline void advise_capacity(const std::vector<T>& list, advice what) noexcept {
  if (list.capacity() > 0 && (what == advice::reuse || list.empty())) {
    advise(list.data(), list.capacity() * sizeof(T), what);
  }
}

inline void advise_tape(const tape_t& tape, advice what) noexcept {
  const auto& b = blocks(tape);
  const auto& s = sizes(tape);
  for (std::size_t i = 0; i < b.size() && i < s.size(); ++i) {
    advise(b[i], s[i], what);
  }
  advise_capacity(tape.var_stack_, what);
  advise_capacity(tape.var_nochain_stack_, what);
}

template <typename T>
inline std::size_t capacity_bytes(const std::vector<T>& list) {
  return list.capacity() * sizeof(T);
}

#if defined(__linux__)
// The bytes of the whole pages inside [begin, begin + bytes) that are in
// memory now.
inline long long resident_bytes(const void* begin, std::size_t bytes) {
  if (begin == nullptr || bytes == 0) {
    return 0;
  }
  const auto page = static_cast<std::uintptr_t>(::sysconf(_SC_PAGESIZE));
  const auto start = reinterpret_cast<std::uintptr_t>(begin);
  const std::uintptr_t first = (start + page - 1) / page * page;
  const std::uintptr_t end = (start + bytes) / page * page;
  if (end <= first) {
    return 0;
  }
  std::vector<unsigned char> present((end - first) / page);
  if (::mincore(reinterpret_cast<void*>(first), end - first, present.data()) != 0) {
    return -1;
  }
  long long count = 0;
  for (unsigned char p : present) {
    count += p & 1;
  }
  return count * static_cast<long long>(page);
}
#endif

}  // namespace stanblocks_tape

// Hand the pages of the calling thread's tape memory in this library back to
// the system: the arena's blocks and the spare storage of its two pointer
// lists. Nothing is freed; the next gradient on this thread writes into the
// same blocks. Returns 1 when released, 0 when this thread has no tape in this
// library, -1 when a gradient is in progress on it (nothing is touched then),
// and -2 where pages cannot be handed back (neither Linux nor macOS).
STANBLOCKS_TAPE_API int stanblocks_tape_release() noexcept {
  auto* tape = stan::math::ChainableStack::instance_;
  if (tape == nullptr) {
    return 0;
  }
  if (!stanblocks_tape::idle(*tape)) {
    return -1;
  }
#if STANBLOCKS_TAPE_PAGES
  stanblocks_tape::advise_tape(*tape, stanblocks_tape::advice::discard);
  stanblocks_tape::released = true;
  return 1;
#else
  return -2;
#endif
}

// Say that the calling thread's tape memory in this library is in use again,
// before the first gradient after a release. Only macOS needs it, for what it
// charges to the process; everywhere it is cheap, and a call without a
// release since the last one does nothing. Returns 1 after a release, else 0.
STANBLOCKS_TAPE_API int stanblocks_tape_reuse() noexcept {
  if (!stanblocks_tape::released) {
    return 0;
  }
  stanblocks_tape::released = false;
  auto* tape = stan::math::ChainableStack::instance_;
  if (tape != nullptr) {
    stanblocks_tape::advise_tape(*tape, stanblocks_tape::advice::reuse);
  }
  return 1;
}

// The bytes the calling thread's tape in this library has reserved (its
// blocks and the storage of its pointer lists) or, with `resident` nonzero,
// how many of the arena's block bytes are in memory now. -1 when this thread
// has no tape in this library, or when residency cannot be measured (it is
// measured on Linux only).
STANBLOCKS_TAPE_API long long stanblocks_tape_bytes(int resident) noexcept {
  auto* tape = stan::math::ChainableStack::instance_;
  if (tape == nullptr) {
    return -1;
  }
  const auto& b = stanblocks_tape::blocks(*tape);
  const auto& s = stanblocks_tape::sizes(*tape);
  long long total = 0;
  if (resident == 0) {
    for (std::size_t i = 0; i < s.size(); ++i) {
      total += static_cast<long long>(s[i]);
    }
    return total + static_cast<long long>(stanblocks_tape::capacity_bytes(tape->var_stack_)
                                          + stanblocks_tape::capacity_bytes(tape->var_nochain_stack_)
                                          + stanblocks_tape::capacity_bytes(tape->var_alloc_stack_));
  }
#if defined(__linux__)
  for (std::size_t i = 0; i < b.size() && i < s.size(); ++i) {
    const long long bytes = stanblocks_tape::resident_bytes(b[i], s[i]);
    if (bytes < 0) {
      return -1;
    }
    total += bytes;
  }
  return total;
#else
  (void)b;
  return -1;
#endif
}

// Whether each thread has its own tape (a library built with STAN_THREADS),
// or one tape serves the whole process.
STANBLOCKS_TAPE_API int stanblocks_tape_per_thread() noexcept {
#ifdef STAN_THREADS
  return 1;
#else
  return 0;
#endif
}

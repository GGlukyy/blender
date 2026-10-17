/* SPDX-FileCopyrightText: 2025 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "BLI_filereader.hh"

#include <unordered_map>

#include "BLO_core_bhead.hh"

namespace blender {

static BHead bhead_from_bhead4(const BHead4 &bhead4)
{
  BHead bhead;
  bhead.code = bhead4.code;
  bhead.len = bhead4.len;
  bhead.old = reinterpret_cast<const void *>(uintptr_t(bhead4.old));
  bhead.SDNAnr = bhead4.SDNAnr;
  bhead.nr = bhead4.nr;
  return bhead;
}

/* Truncation collision tracking (32-bit builds reading 64-bit files), see
 * #BLO_bhead_truncation_check_begin. Only the thread opening the file touches this. */
static thread_local std::unordered_map<uint32_t, uint64_t> *g_trunc_seen = nullptr;
static thread_local int64_t g_trunc_collisions = 0;

void BLO_bhead_truncation_check_begin()
{
  delete g_trunc_seen;
  g_trunc_seen = new std::unordered_map<uint32_t, uint64_t>();
  g_trunc_collisions = 0;
}

int64_t BLO_bhead_truncation_check_end()
{
  delete g_trunc_seen;
  g_trunc_seen = nullptr;
  return g_trunc_collisions;
}

static const void *old_ptr_from_uint64_ptr(const uint64_t ptr)
{
  if constexpr (sizeof(void *) == 8) {
    return reinterpret_cast<const void *>(ptr);
  }
  else {
    const uint32_t key = uint32_from_uint64_ptr(ptr);
    if (g_trunc_seen && ptr != 0) {
      /* Same raw address twice is legitimate; DIFFERENT raw addresses with the same key are the
       * corrupting collision. */
      const auto [it, inserted] = g_trunc_seen->emplace(key, ptr);
      if (!inserted && it->second != ptr) {
        g_trunc_collisions++;
      }
    }
    return reinterpret_cast<const void *>(uintptr_t(key));
  }
}

static BHead bhead_from_small_bhead8(const SmallBHead8 &small_bhead8)
{
  BHead bhead;
  bhead.code = small_bhead8.code;
  bhead.len = small_bhead8.len;
  bhead.old = old_ptr_from_uint64_ptr(small_bhead8.old);
  bhead.SDNAnr = small_bhead8.SDNAnr;
  bhead.nr = small_bhead8.nr;
  return bhead;
}

static BHead bhead_from_large_bhead8(const LargeBHead8 &large_bhead8)
{
  BHead bhead;
  bhead.code = large_bhead8.code;
  bhead.len = large_bhead8.len;
  bhead.old = old_ptr_from_uint64_ptr(large_bhead8.old);
  bhead.SDNAnr = large_bhead8.SDNAnr;
  bhead.nr = large_bhead8.nr;
  return bhead;
}

std::optional<BHead> BLO_readfile_read_bhead(FileReader *file, const BHeadType type)
{
  /* NOTE: this is endianness-sensitive. */
  /* The various BHead data below would need to be switched if the read blend-file was saved from a
   * system using a different endianness. */
  switch (type) {
    case BHeadType::BHead4: {
      BHead4 bhead4{};
      bhead4.code = BLO_CODE_DATA;
      const int64_t readsize = file->read(file, &bhead4, sizeof(bhead4));
      if (readsize == sizeof(bhead4) || bhead4.code == BLO_CODE_ENDB) {
        return bhead_from_bhead4(bhead4);
      }
      break;
    }
    case BHeadType::SmallBHead8: {
      SmallBHead8 small_bhead8{};
      small_bhead8.code = BLO_CODE_DATA;
      const int64_t readsize = file->read(file, &small_bhead8, sizeof(small_bhead8));
      if (readsize == sizeof(small_bhead8) || small_bhead8.code == BLO_CODE_ENDB) {
        return bhead_from_small_bhead8(small_bhead8);
      }
      break;
    }
    case BHeadType::LargeBHead8: {
      LargeBHead8 large_bhead8{};
      large_bhead8.code = BLO_CODE_DATA;
      const int64_t readsize = file->read(file, &large_bhead8, sizeof(large_bhead8));
      if (readsize == sizeof(large_bhead8) || large_bhead8.code == BLO_CODE_ENDB) {
        return bhead_from_large_bhead8(large_bhead8);
      }
      break;
    }
  }
  return std::nullopt;
}

}  // namespace blender

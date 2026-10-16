/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * WebGPU index buffer: CPU shadow (data_, owned by base) + a WGPUBuffer
 * (Index | CopyDst | Storage) uploaded on demand.
 */

#pragma once

#include "GPU_index_buffer.hh"

#include <webgpu/webgpu.h>

namespace blender::gpu {

class WebGPUIndexBuf : public IndexBuf {
 private:
  WGPUBuffer buffer_ = nullptr;
  bool uploaded_ = false;
  size_t buffer_size_ = 0;
  uint64_t gen_ = 0;

 public:
  ~WebGPUIndexBuf() override;

  void upload_data() override;
  void bind_as_ssbo(uint binding) override;
  void read(uint32_t *data) const override;
  void update_sub(uint start, uint len, const void *data) override;

  /* CPU shadow (null for subranges) + content generation, for point-sprite de-indexing. */
  const void *cpu_data() const
  {
    return is_subrange_ ? nullptr : data_;
  }
  bool is_subrange() const
  {
    return is_subrange_;
  }
  uint32_t index_len_raw() const
  {
    return index_len_;
  }
  uint64_t gen() const
  {
    return gen_;
  }

  WGPUBuffer wgpu_buffer()
  {
    /* Subrange ibos (per-material slices of a mesh's tris ibo) own NO data —
     * they reference src_ with an index offset. Use the parent's GPU buffer
     * (uploading from a subrange would read the null/dangling data_). */
    if (is_subrange_) {
      return static_cast<WebGPUIndexBuf *>(src_)->wgpu_buffer();
    }
    /* First use: create AND fill the GPU buffer. Checking `uploaded_` (not just
     * buffer_ == null) matters because bind_as_ssbo can create the buffer via
     * ensure_buffer WITHOUT uploading — the indices would stay zero forever
     * (every triangle degenerate: the invisible-cube bug). */
    if (!uploaded_) {
      upload_data();
    }
    return buffer_;
  }
  /* Byte offset of this (sub)range's first index inside wgpu_buffer(). */
  uint64_t wgpu_index_offset_bytes() const
  {
    return uint64_t(index_start_) * (is_32bit() ? 4 : 2);
  }
  WGPUIndexFormat wgpu_index_format() const
  {
    return is_32bit() ? WGPUIndexFormat_Uint32 : WGPUIndexFormat_Uint16;
  }

 private:
  void ensure_buffer();
  void strip_restart_indices() override {}
};

}  // namespace blender::gpu

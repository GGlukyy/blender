/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * WebGPU vertex buffer: a CPU shadow (data_) plus a real WGPUBuffer (VERTEX |
 * COPY_DST | STORAGE) created+uploaded on upload_data(). Device-less mode keeps
 * only the shadow.
 */

#pragma once

#include "BLI_sys_types.hh"

#include "MEM_guardedalloc.h"

#include "GPU_vertex_buffer.hh"

#include <webgpu/webgpu.h>

namespace blender::gpu {

class WebGPUContext;

class WebGPUVertexBuffer : public VertBuf {
 private:
  WGPUBuffer buffer_ = nullptr;
  size_t buffer_size_ = 0;
  uint64_t gen_ = 0;

 public:
  ~WebGPUVertexBuffer() override;

  void bind_as_ssbo(uint binding) override;
  void bind_as_texture(uint binding) override;
  void wrap_handle(uint64_t /*handle*/) override {}
  void update_sub(uint start, uint len, const void *data) override;
  void copy_sub(VertBuf &source_buf,
                uint source_first_vertex,
                uint dest_first_vertex,
                uint vertex_len) override;
  void read(void *data) const override;

  /* CPU shadow + content generation (point-sprite de-indexing cache, webgpu_batch.cc). */
  const uchar *cpu_data() const
  {
    return data_;
  }
  uint64_t gen() const
  {
    return gen_;
  }

  WGPUBuffer wgpu_buffer()
  {
    /* Flush pending CPU data to the GPU before use — draws pull buffers via this
     * accessor and nothing else triggers VertBuf::upload() on this backend.
     * Without this the WGPUBuffer exists but holds zeros (degenerate geometry,
     * no fragments rasterized anywhere). */
    if (flag & GPU_VERTBUF_DATA_DIRTY) {
      this->upload();
    }
    ensure_buffer();
    return buffer_;
  }

 protected:
  void acquire_data() override;
  void resize_data() override;
  void release_data() override;
  void upload_data() override;

 private:
  void ensure_buffer();
  /* Swap in a fresh WGPUBuffer when updating while a pass records (see impl). */
  bool cow_if_pass_open(WebGPUContext *ctx);
};

}  // namespace blender::gpu

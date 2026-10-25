/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "GPU_matrix.hh"
#include "GPU_shader.hh"

#include "webgpu_batch.hh"
#include "webgpu_context.hh"
#include "webgpu_immediate.hh"
#include "webgpu_shader.hh"

namespace blender::gpu {

/* ?perf=1: immediate-mode buffers created this frame (reset in present_backbuffer). */
int g_stat_imm_buffers = 0;
long long g_stat_imm_bytes = 0;

uchar *WebGPUImmediate::begin()
{
  const size_t bytes_needed = size_t(vertex_len) * vertex_format.stride;
  /* +1 vertex of slack for the LINE_LOOP -> LINE_STRIP close in end(). */
  data_.resize(bytes_needed + vertex_format.stride);
  return data_.data();
}

void WebGPUImmediate::end()
{
  if (vertex_idx == 0) {
    return;
  }
  WebGPUContext *ctx = WebGPUContext::get();
  if (ctx == nullptr || ctx->device() == nullptr) {
    return;
  }

  if (prim_type == GPU_PRIM_LINE_LOOP) {
    /* Close the loop by appending the first vertex (slack reserved in begin()). */
    std::memcpy(data_.data() + size_t(vertex_idx) * vertex_format.stride,
                data_.data(),
                vertex_format.stride);
    prim_type = GPU_PRIM_LINE_STRIP;
    vertex_idx += 1;
  }
  else if (prim_type == GPU_PRIM_TRI_FAN && vertex_idx >= 3) {
    /* WebGPU has no triangle-fan topology; mapping a fan onto a list drops all
     * but the first triangle (the splash image rendered as a lone diagonal
     * half). Expand fan (v0, vi, vi+1) into an explicit triangle list. */
    const size_t stride = vertex_format.stride;
    const uint tri_count = vertex_idx - 2;
    std::vector<uchar> fan(3 * size_t(tri_count) * stride);
    for (uint t = 0; t < tri_count; t++) {
      std::memcpy(fan.data() + (3 * size_t(t) + 0) * stride, data_.data(), stride);
      std::memcpy(fan.data() + (3 * size_t(t) + 1) * stride,
                  data_.data() + size_t(t + 1) * stride,
                  stride);
      std::memcpy(fan.data() + (3 * size_t(t) + 2) * stride,
                  data_.data() + size_t(t + 2) * stride,
                  stride);
    }
    data_.assign(fan.begin(), fan.end());
    prim_type = GPU_PRIM_TRIS;
    vertex_idx = 3 * tri_count;
  }

  const size_t used = size_t(vertex_idx) * vertex_format.stride;
  const size_t buf_size = (used + 3) & ~size_t(3);

  /* Ring buffer (H1-A). Offsets only grow while this context has recorded-but-unsubmitted
   * commands (they may still read earlier ring contents); once nothing is pending the cursor
   * rewinds, and queue writes are ordered after the earlier submits. */
  static const bool use_ring = [] {
    const char *e = getenv("WGPU_IMM_RING");
    return e == nullptr || e[0] != '0';
  }();
  WGPUBuffer buf = nullptr;
  uint64_t off = 0;
  if (use_ring) {
    if (!ctx->pass_open()) {
      ring_off_ = 0;
    }
    constexpr uint64_t max_ring = 32ull << 20;
    off = (ring_off_ + 255) & ~uint64_t(255);
    if (ring_ != nullptr && off + buf_size > ring_cap_ && ring_cap_ >= max_ring) {
      /* Full at the cap: submit pending work, then reuse from the start (no allocation). */
      ctx->submit();
      ring_off_ = 0;
      off = 0;
    }
    if (ring_ == nullptr || off + buf_size > ring_cap_) {
      uint64_t cap = std::max<uint64_t>({ring_cap_ * 2, uint64_t(buf_size), 1ull << 20});
      cap = (cap + 255) & ~uint64_t(255);
      WGPUBufferDescriptor rd = {};
      rd.label = {"imm_ring", WGPU_STRLEN};
      rd.size = cap;
      rd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
      WGPUBuffer nb = wgpuDeviceCreateBuffer(ctx->device(), &rd);
      if (nb == nullptr) {
        return;
      }
      if (ring_ != nullptr) {
        /* Recorded commands hold their own reference; destroy is sequenced after them. */
        wgpu_scrub_buffer_all_contexts(ring_);
        wgpu_buffer_free(ring_);
      }
      ring_ = nb;
      ring_cap_ = cap;
      off = 0;
    }
    buf = ring_;
    wgpu_counted_write(ctx->queue(), buf, off, data_.data(), buf_size);
    ring_off_ = off + buf_size;
  }
  else {
    /* Legacy path: fresh buffer per call. */
    WGPUBufferDescriptor bd = {};
    bd.label = {"immediate", WGPU_STRLEN};
    bd.size = buf_size;
    bd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    buf = wgpuDeviceCreateBuffer(ctx->device(), &bd);
    if (buf == nullptr) {
      return;
    }
    wgpu_counted_write(ctx->queue(), buf, 0, data_.data(), buf_size);
  }
  g_stat_imm_buffers++;
  g_stat_imm_bytes += buf_size;

  if (shader->is_polyline) {
    /* Polyline shaders read positions/colors from SSBOs and expand to triangles
     * (the wide-line workaround). Bind the whole ring at the polyline slots; the shader finds
     * its data through the offset passed to polyline_draw_workaround (as Metal does). */
    ctx->bind_ssbo(GPU_SSBO_POLYLINE_POS_BUF_SLOT, buf);
    ctx->bind_ssbo(GPU_SSBO_POLYLINE_COL_BUF_SLOT, buf);
    ctx->bind_ssbo(GPU_SSBO_INDEX_BUF_SLOT, buf);
    this->polyline_draw_workaround(off);
    ctx->unbind_ssbo(GPU_SSBO_POLYLINE_POS_BUF_SLOT);
    ctx->unbind_ssbo(GPU_SSBO_POLYLINE_COL_BUF_SLOT);
    ctx->unbind_ssbo(GPU_SSBO_INDEX_BUF_SLOT);
  }
  else {
    GPU_matrix_bind(ctx->shader);
    webgpu_immediate_draw(prim_type, vertex_format, buf, vertex_idx, off, buf_size);
  }

  if (!use_ring) {
    ctx->scrub_buffer(buf);
    wgpu_buffer_free(buf);
  }
}

WebGPUImmediate::~WebGPUImmediate()
{
  if (ring_ != nullptr) {
    wgpu_scrub_buffer_all_contexts(ring_);
    wgpu_buffer_free(ring_);
  }
}

}  // namespace blender::gpu

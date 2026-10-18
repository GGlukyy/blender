/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include <cstring>

#include "webgpu_context.hh"
#include "webgpu_vertex_buffer.hh"

namespace blender::gpu {

WebGPUVertexBuffer::~WebGPUVertexBuffer()
{
  if (WebGPUContext *ctx = static_cast<WebGPUContext *>(Context::get())) {
    ctx->scrub_buffer(buffer_);
  }
  if (buffer_) {
    wgpuBufferRelease(buffer_);
  }
  MEM_SAFE_DELETE(data_);
}

void WebGPUVertexBuffer::acquire_data()
{
  if (usage_ == GPU_USAGE_DEVICE_ONLY) {
    return;
  }
  MEM_SAFE_DELETE(data_);
  data_ = MEM_new_array_uninitialized<uchar>(this->size_alloc_get(), __func__);
}

void WebGPUVertexBuffer::resize_data()
{
  if (usage_ == GPU_USAGE_DEVICE_ONLY) {
    return;
  }
  data_ = static_cast<uchar *>(
      MEM_realloc_uninitialized(data_, sizeof(uchar) * this->size_alloc_get()));
}

void WebGPUVertexBuffer::release_data()
{
  if (buffer_) {
    wgpuBufferRelease(buffer_);
    buffer_ = nullptr;
    buffer_size_ = 0;
  }
  MEM_SAFE_DELETE(data_);
}

void WebGPUVertexBuffer::ensure_buffer()
{
  const size_t needed = this->size_used_get();
  if (buffer_ && buffer_size_ >= needed) {
    return;
  }
  WebGPUContext *ctx = WebGPUContext::get();
  if (ctx == nullptr || ctx->device() == nullptr || needed == 0) {
    return;
  }
  if (buffer_) {
    wgpuBufferRelease(buffer_);
    buffer_ = nullptr;
  }
  /* WGPU requires buffer size be a multiple of 4. */
  buffer_size_ = (needed + 3) & ~size_t(3);
  WGPUBufferDescriptor desc = {};
  desc.size = buffer_size_;
  desc.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage |
               WGPUBufferUsage_CopySrc;
  buffer_ = wgpuDeviceCreateBuffer(ctx->device(), &desc);
}

/* Copy-on-write: updating a live buffer while a pass is recording would either
 * retroactively change what earlier draws read (queue writes execute before the
 * pass's submit) or force a full submit per update (pathological — BLF text
 * updates a VBO per run). A fresh buffer isolates the recorded draws: the pass
 * encoder holds refs to the old one. Requires the CPU shadow (full contents). */
bool WebGPUVertexBuffer::cow_if_pass_open(WebGPUContext *ctx)
{
  if (!ctx->pass_open() || buffer_ == nullptr || data_ == nullptr) {
    return false;
  }
  WGPUBufferDescriptor desc = {};
  desc.size = buffer_size_;
  desc.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage |
               WGPUBufferUsage_CopySrc;
  WGPUBuffer fresh = wgpuDeviceCreateBuffer(ctx->device(), &desc);
  if (fresh == nullptr) {
    return false;
  }
  ctx->rebind_buffer(buffer_, fresh); /* VBOs can be bound as SSBOs. */
  wgpuBufferRelease(buffer_);
  buffer_ = fresh;
  return true;
}

void WebGPUVertexBuffer::upload_data()
{
  gen_ = webgpu_buffer_gen_next();
  ensure_buffer();
  WebGPUContext *ctx = WebGPUContext::get();
  if (buffer_ && ctx && ctx->queue() && data_) {
    const size_t used = this->size_used_get();
    cow_if_pass_open(ctx);
    wgpu_counted_write(ctx->queue(), buffer_, 0, data_, (used + 3) & ~size_t(3));
  }
}

void WebGPUVertexBuffer::update_sub(uint start, uint len, const void *data)
{
  gen_ = webgpu_buffer_gen_next();
  if (data_) {
    memcpy(data_ + start, data, len);
  }
  ensure_buffer();
  WebGPUContext *ctx = WebGPUContext::get();
  if (buffer_ && ctx && ctx->queue()) {
    if (cow_if_pass_open(ctx)) {
      /* Fresh buffer: upload the FULL shadow (it includes this sub-range). */
      wgpu_counted_write(
          ctx->queue(), buffer_, 0, data_, (this->size_used_get() + 3) & ~size_t(3));
    }
    else {
      /* No shadow to rebuild from: preserve GL ordering the expensive way. */
      ctx->flush_if_pass_open("vbo_sub");
      wgpu_counted_write(ctx->queue(), buffer_, start, data, (len + 3) & ~size_t(3));
    }
  }
}

void WebGPUVertexBuffer::bind_as_ssbo(uint binding)
{
  WebGPUContext *ctx = WebGPUContext::get();
  if (ctx) {
    ctx->bind_ssbo(int(binding), wgpu_buffer());
  }
}

void WebGPUVertexBuffer::bind_as_texture(uint binding)
{
  /* Texel buffer: the shader reads this buffer through a read-only storage binding
   * (see promote_texel_buffers in webgpu_shader.cc). */
  WebGPUContext *ctx = WebGPUContext::get();
  static const bool dbg = getenv("WGPU_LOG_TEXBUF") != nullptr;
  if (dbg && data_) {
    const float *f = reinterpret_cast<const float *>(data_);
    fprintf(stderr,
            "WGPU_TEXBUF bind slot=%u used=%zu bytes first=[%g %g %g %g | %g %g %g %g | %g %g %g %g]\n",
            binding,
            this->size_used_get(),
            f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11]);
    if (this->size_used_get() > 792 * 16 + 48) {
      const float *g = f + 792 * 4;
      fprintf(stderr, "WGPU_TEXBUF texel792=[%g %g %g %g | %d %d %d %d | %g %g %g %g]\n", g[0], g[1], g[2], g[3],
              reinterpret_cast<const int *>(g)[4], reinterpret_cast<const int *>(g)[5],
              reinterpret_cast<const int *>(g)[6], reinterpret_cast<const int *>(g)[7], g[8], g[9], g[10], g[11]);
    }
    fflush(stderr);
  }
  if (ctx) {
    ctx->bind_texel_buffer(int(binding), wgpu_buffer(), int(GPU_vertbuf_get_format(this)->attrs[0].type.size()));
  }
}

void WebGPUVertexBuffer::read(void *data) const
{
  /* GPU->host readback needs an async map (no JSPI). Return the CPU shadow. */
  if (data_ && data) {
    memcpy(data, data_, this->size_used_get());
  }
}

}  // namespace blender::gpu

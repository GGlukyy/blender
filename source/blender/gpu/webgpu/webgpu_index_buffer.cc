/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include <cstring>

#include "webgpu_context.hh"
#include "webgpu_index_buffer.hh"

namespace blender::gpu {

WebGPUIndexBuf::~WebGPUIndexBuf()
{
  if (WebGPUContext *ctx = static_cast<WebGPUContext *>(Context::get())) {
    wgpu_scrub_buffer_all_contexts(buffer_);
  }
  if (buffer_) {
    wgpu_buffer_free(buffer_);
  }
}

void WebGPUIndexBuf::ensure_buffer()
{
  const size_t needed = this->size_get();
  if (buffer_ && buffer_size_ >= needed) {
    return;
  }
  WebGPUContext *ctx = WebGPUContext::get();
  if (ctx == nullptr || ctx->device() == nullptr || needed == 0 || data_ == nullptr) {
    return;
  }
  if (buffer_) {
    wgpu_buffer_free(buffer_);
    buffer_ = nullptr;
  }
  buffer_size_ = (needed + 3) & ~size_t(3);
  WGPUBufferDescriptor desc = {};
  desc.size = buffer_size_;
  desc.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage |
               WGPUBufferUsage_CopySrc;
  buffer_ = wgpu_buffer_acquire(ctx->device(), &desc);
}

void WebGPUIndexBuf::upload_data()
{
  gen_ = webgpu_buffer_gen_next();
  if (is_subrange_) {
    static_cast<WebGPUIndexBuf *>(src_)->upload_data();
    return;
  }
  ensure_buffer();
  WebGPUContext *ctx = WebGPUContext::get();
  {
    static int s_log = 0;
    if (s_log < 20) {
      s_log++;
      const uint16_t *u = static_cast<const uint16_t *>(data_);
      fprintf(stderr, "WGPU_IBUP this=%p buf=%p data=%p len=%u d=[%u %u %u %u %u %u]\n",
              (void *)this, (void *)buffer_, (void *)data_, index_len_,
              u ? u[0] : 0, u ? u[1] : 0, u ? u[2] : 0, u ? u[3] : 0, u ? u[4] : 0,
              u ? u[5] : 0);
      fflush(stderr);
    }
  }
  if (buffer_ && ctx && ctx->queue() && data_) {
    ctx->flush_if_pass_open("ibo");
    wgpu_counted_write(ctx->queue(), buffer_, 0, data_, (this->size_get() + 3) & ~size_t(3));
    uploaded_ = true;
  }
}

void WebGPUIndexBuf::bind_as_ssbo(uint binding)
{
  if (is_subrange_) {
    src_->bind_as_ssbo(binding);
    return;
  }
  if (!uploaded_) {
    upload_data();
  }
  WebGPUContext *ctx = WebGPUContext::get();
  if (ctx) {
    ctx->bind_ssbo(int(binding), wgpu_buffer());
  }
}

void WebGPUIndexBuf::read(uint32_t *data) const
{
  if (data_ && data) {
    memcpy(data, data_, this->size_get());
  }
}

void WebGPUIndexBuf::update_sub(uint start, uint len, const void *data)
{
  gen_ = webgpu_buffer_gen_next();
  ensure_buffer();
  WebGPUContext *ctx = WebGPUContext::get();
  if (buffer_ && ctx && ctx->queue()) {
    ctx->flush_if_pass_open("ibo_sub"); wgpu_counted_write(ctx->queue(), buffer_, start, data, (len + 3) & ~size_t(3));
  }
}

}  // namespace blender::gpu

namespace blender::gpu {
void WebGPUIndexBuf::copy_sub(IndexBuf &source_buf,
                              uint source_first_index,
                              uint dest_first_index,
                              uint index_len)
{
  /* N1b (5.3): new pure virtual. Copy through the CPU shadow (no GPU->GPU copy node on this backend). */
  const uchar *src = static_cast<const uchar *>(static_cast<WebGPUIndexBuf &>(source_buf).cpu_data());
  if (!src) {
    fprintf(stderr, "WebGPUIndexBuf::copy_sub: source has no CPU shadow, skipped\n");
    return;
  }
  const uint size = to_bytesize(index_type_);
  this->update_sub(dest_first_index * size, index_len * size, src + size_t(source_first_index) * size);
}
}  // namespace blender::gpu

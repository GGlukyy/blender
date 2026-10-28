/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * WebGPU backend context. Holds the WGPUDevice/queue plus the per-frame command
 * encoder, the active render pass, the resource binding tables (UBO/SSBO/texture/
 * image — WebGPU binds resources through bind groups at draw time, not via global
 * state), and a render-pipeline cache. On Emscripten the device is created in JS
 * and handed over via emscripten_webgpu_get_device() (no JSPI).
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "gpu_context_private.hh"

#include <webgpu/webgpu.h>

#include "webgpu_framebuffer.hh"
#include "webgpu_shader.hh"

#include <emscripten/emscripten.h>

/* Tells the page that a (possibly long) shader/pipeline compile starts (rate limited). */
void wgpu_note_compile();

/* Deferred GPU buffer destruction (sprint "iPad vydrží" M2). emdawnwebgpu's wgpuBufferRelease only
 * drops the JS handle; the GPUBuffer memory is freed when the JS GC gets to it, which Safari does far
 * too late (thousands of immediate/staging buffers per minute -> GPU process killed). So buffers that
 * Blender OWNS are queued here and destroy()ed right after the next queue submit (destroy before the
 * submit would invalidate commands still being recorded). Never use on AddRef'd (shared) handles. */
void wgpu_buffer_free(WGPUBuffer b);
WGPUBuffer wgpu_buffer_acquire(WGPUDevice dev, const WGPUBufferDescriptor *d); /* K3 pool */
/* Same for textures that Blender owns (NOT texture views sharing a source via AddRef). */
void wgpu_texture_free(WGPUTexture t);
void wgpu_flush_buffer_frees();
/* Drop `buf` from the bind tables of EVERY context (frees can come from another context/thread). */
void wgpu_scrub_buffer_all_contexts(WGPUBuffer buf);

/* Perf counters for buffer uploads (all queue writes go through wgpu_counted_write). */
inline long long g_stat_writes = 0, g_stat_write_bytes = 0, g_stat_write_big = 0;
inline void wgpu_counted_write(WGPUQueue q, WGPUBuffer b, uint64_t off, const void *d, size_t n)
{
  g_stat_writes++;
  g_stat_write_bytes += n;
  g_stat_write_big += (n > 65536);
  wgpuQueueWriteBuffer(q, b, off, d, n);
}

namespace blender::gpu {

/* Dynamic offsets for the push-constant UBO binding (H1-B). The bind group (and its cache key) carries
 * offset 0; every draw/dispatch MUST pass these to SetBindGroup, or all draws read the first slice. */
struct BindDynOffsets {
  uint32_t v[4] = {};
  uint32_t n = 0;
};

/** ?perf=1 telemetry: wall time spent in WebGPU API recording/submission per frame (outermost
 * scope only, so nested timers are not double counted). Reset per frame in present_backbuffer. */
extern double g_stat_api_ms;
extern int g_stat_api_depth;
/** True when the page was opened with ?perf=1 (ENV.WGPU_PERF); timers cost ~4% otherwise. */
extern bool g_perf_enabled;
struct WebGPUApiTimer {
  double t0 = 0.0;
  WebGPUApiTimer()
  {
    if (g_perf_enabled && g_stat_api_depth++ == 0) {
      t0 = emscripten_get_now();
    }
  }
  ~WebGPUApiTimer()
  {
    if (g_perf_enabled && --g_stat_api_depth == 0) {
      g_stat_api_ms += emscripten_get_now() - t0;
    }
  }
};

class WebGPUTexture;

/* WebGPU baseline per-stage binding counts. */
/* Blender-side slot spaces (EEVEE uses texture slots up to 19, see
 * GBUF_*_TEX_SLOT in eevee_defines.hh — these are table indices, not WebGPU
 * binding counts, so oversizing is free). */
#define WEBGPU_MAX_UBO 32
#define WEBGPU_MAX_SSBO 32
#define WEBGPU_MAX_TEX 32
#define WEBGPU_MAX_IMAGE 16

class WebGPUContext : public Context {
 private:
  WGPUDevice device_ = nullptr;
  WGPUQueue queue_ = nullptr;

  /* Per-frame command recording. Created lazily; flushed on submit(). */
  WGPUCommandEncoder encoder_ = nullptr;
  uint64_t encoder_seq_ = 0; /* g_free_seq when encoder_ was created (see wgpu_flush_buffer_frees). */
  WGPURenderPassEncoder render_pass_ = nullptr;
  WebGPUFrameBuffer *render_pass_fb_ = nullptr;

  /* Resource binding tables (group 0). Populated by *Buf::bind / texture_bind,
   * consumed when a bind group is assembled at draw/dispatch time. */
  WGPUBuffer bound_ubo_[WEBGPU_MAX_UBO] = {};
  WGPUBuffer bound_ssbo_[WEBGPU_MAX_SSBO] = {};
  WebGPUTexture *bound_tex_[WEBGPU_MAX_TEX] = {};
  /* Texel buffers (samplerBuffer): vertex buffers bound as textures are read as storage buffers. */
  WGPUBuffer bound_texbuf_[WEBGPU_MAX_TEX] = {};
  uint8_t bound_texbuf_bytes_[WEBGPU_MAX_TEX] = {};
  /* Sampler state captured at bind time (filtering/extend per unit). */
  GPUSamplerState bound_tex_state_[WEBGPU_MAX_TEX] = {};
  /* Cache: GPUSamplerState key -> WGPUSampler (see sampler_for_state). */
  Map<uint32_t, WGPUSampler> state_samplers_;
  WebGPUTexture *bound_image_[WEBGPU_MAX_IMAGE] = {};

  /* Render-pipeline cache keyed by (shader, vertex/format/prim/target) hash. */
  std::unordered_map<uint64_t, WGPURenderPipeline> render_pipelines_;
  std::unordered_map<uint64_t, WGPUComputePipeline> compute_pipelines_;

  WGPUSampler default_sampler_ = nullptr;
  WGPUSampler nearest_sampler_ = nullptr;
  std::unordered_map<uint64_t, WGPUTextureView> dummy_storage_views_;
  /* Bind groups keyed on shader + resolved entries (see build_bind_group).
   * Every entry resource is explicitly AddRef'd while cached: the bind group's
   * own internal refs keep the underlying JS objects alive but NOT the C-side
   * handle IDs — a freed handle ID gets recycled (the per-frame icon textures
   * did), making two different resources hash identically and serving a STALE
   * bind group ("all icons render the same"). */
  struct CachedBindGroup {
    WGPUBindGroup bg;
    std::vector<WGPUBindGroupEntry> entries;
    uint64_t used = 0; /* LRU clock value of the last hit (H1 step 3). */
  };
  std::unordered_map<uint64_t, CachedBindGroup> bind_group_cache_;
  uint64_t bg_clock_ = 0;
  void cached_bind_group_release(CachedBindGroup &c)
  {
    for (const WGPUBindGroupEntry &e : c.entries) {
      if (e.buffer) {
        wgpuBufferRelease(e.buffer);
      }
      if (e.textureView) {
        wgpuTextureViewRelease(e.textureView);
      }
      if (e.sampler) {
        wgpuSamplerRelease(e.sampler);
      }
    }
    wgpuBindGroupRelease(c.bg);
  }
  std::unordered_map<uint64_t, WGPUTexture> snapshot_textures_;
  std::unordered_map<uint64_t, WGPUBuffer> snapshot_buffers_;

  /* Debug capture: the WGPU handle (AddRef'd at draw time so it survives the
   * pooled texture being freed/reused) of the first content color texture; copied
   * to a persistent readback buffer right after its render pass ends, so JS can
   * map it after the render (see the webgpu_capture_* exports). */
  WGPUTexture capture_src_ = nullptr;
  uint32_t capture_w_ = 0, capture_h_ = 0;
  WGPUTextureFormat capture_fmt_ = WGPUTextureFormat_Undefined;
  bool capture_done_ = false;
  /* Set when a new capture target was recorded; render_pass_end copies the
   * texture once the active pass ends, then clears this. Lets later draws to the
   * final combined framebuffer overwrite earlier captures (last-wins). */
  bool capture_pending_ = false;

 public:
  WebGPUContext(GHOST_IWindow *ghost_window, GHOST_IContext *ghost_context);
  ~WebGPUContext() override;

  static WebGPUContext *get() { return static_cast<WebGPUContext *>(Context::get()); }

  void activate() override
  {
    immActivate();
  }
  void deactivate() override
  {
    immDeactivate();
    /* K3b: an idle context must not keep an unsubmitted encoder (it blocks freeing buffers). */
    submit();
  }
  void begin_frame() override {}
  void end_frame() override {}
  void flush() override;
  void finish() override;

  void memory_statistics_get(int *r_total_mem, int *r_free_mem) override
  {
    *r_total_mem = 0;
    *r_free_mem = 0;
  }

  void debug_group_begin(const char * /*name*/, int /*index*/) override {}
  void debug_group_end() override {}
  bool debug_capture_begin(const char * /*title*/) override { return false; }
  void debug_capture_end() override {}
  void *debug_capture_scope_create(const char * /*name*/) override { return nullptr; }
  bool debug_capture_scope_begin(void * /*scope*/) override { return false; }
  void debug_capture_scope_end(void * /*scope*/) override {}
  void debug_unbind_all_ubo() override {}
  void debug_unbind_all_ssbo() override {}

  WGPUDevice device() const { return device_; }
  WGPUQueue queue() const { return queue_; }

  /* --- command / render-pass lifecycle --- */
  WGPUCommandEncoder ensure_encoder();
  /* Begin a render pass on `fb` if one is not already active for it (ends any
   * other active pass first). No-op without a device. */
  void render_pass_ensure(WebGPUFrameBuffer &fb);
  void render_pass_end();

  /* Per-open-render-pass buffer usage tracking. WebGPU validates usage scopes
   * per pass: a buffer bound WRITABLE by one draw and READ-ONLY by another in
   * the same pass invalidates the whole command buffer (EEVEE's volume
   * occupancy prepass -> material pass relies on GL barriers for this).
   * build_bind_group() records the incoming draw's buffers; record_draw ends
   * the open pass first when they conflict with what the pass already used. */
  std::unordered_map<WGPUBuffer, bool> pass_buffer_usage_; /* buffer -> writable */
  std::vector<std::pair<WGPUBuffer, bool>> pending_draw_buffers_;
  bool pending_draw_conflicts()
  {
    for (const auto &pb : pending_draw_buffers_) {
      auto it = pass_buffer_usage_.find(pb.first);
      if (it != pass_buffer_usage_.end() && (it->second || pb.second) &&
          it->second != pb.second) {
        return true;
      }
    }
    return false;
  }
  void commit_pending_draw_buffers()
  {
    for (const auto &pb : pending_draw_buffers_) {
      auto it = pass_buffer_usage_.find(pb.first);
      if (it == pass_buffer_usage_.end()) {
        pass_buffer_usage_[pb.first] = pb.second;
      }
      else {
        it->second = it->second || pb.second;
      }
    }
  }
  /* Finish + submit the current command encoder (one command buffer per pass:
   * WebGPU drops the entire buffer when any one command is invalid). */
  void flush_encoder();
  /* GL-order guard for buffer/texture uploads: a wgpuQueueWrite* executes before
   * any LATER-submitted command buffer, so a write issued while a render pass is
   * still being recorded retroactively changes what the pass's EARLIER draws
   * read. Ending + submitting the open pass first restores GL semantics (the
   * write only affects subsequent draws). */
  void flush_if_pass_open(const char *why = "?")
  {
    /* Passes batch into one encoder now, so an in-place write is hazardous not
     * only while a pass is OPEN but whenever the encoder holds any recorded
     * (unsubmitted) pass — queue writes execute before that eventual submit. */
    if (render_pass_ != nullptr || encoder_ != nullptr) {
      extern int g_stat_flushes;
      extern void webgpu_stat_flush_why(const char *why);
      g_stat_flushes++;
      webgpu_stat_flush_why(why);
      render_pass_end();
      flush_encoder();
    }
  }
  /* True when recorded-but-unsubmitted GPU work may reference live buffers
   * (open pass OR batched encoder content) — the copy-on-write trigger. */
  /* Buffers freed with a sequence number <= this cannot be referenced by our recorded work. */
  uint64_t recording_seq() const
  {
    if (encoder_ != nullptr) {
      return encoder_seq_;
    }
    return render_pass_ != nullptr ? 0 : UINT64_MAX;
  }
  bool pass_open() const
  {
    return render_pass_ != nullptr || encoder_ != nullptr;
  }
  /* Swap every bind-table entry holding `old_buf` to `new_buf` (copy-on-write
   * buffer replacement while bound). */
  void rebind_buffer(WGPUBuffer old_buf, WGPUBuffer new_buf)
  {
    if (old_buf == nullptr) {
      return;
    }
    for (int i = 0; i < WEBGPU_MAX_UBO; i++) {
      if (bound_ubo_[i] == old_buf) {
        table_set(bound_ubo_[i], new_buf);
      }
    }
    for (int i = 0; i < WEBGPU_MAX_SSBO; i++) {
      if (bound_ssbo_[i] == old_buf) {
        table_set(bound_ssbo_[i], new_buf);
      }
    }
    for (int i = 0; i < WEBGPU_MAX_TEX; i++) {
      if (bound_texbuf_[i] == old_buf) {
        table_set(bound_texbuf_[i], new_buf);
      }
    }
  }
  /* DEBUG: copy `size` bytes of `buf` into the persistent capture buffer (mapped
   * from JS after the render; h==1 marks it as a raw uint32 dump). */
  void debug_capture_buffer(WGPUBuffer buf, size_t size);
  void debug_capture_stencil(WGPUTexture tex, uint32_t w, uint32_t h);
  WGPUBuffer bound_ssbo_get(int slot) const
  {
    return (slot >= 0 && slot < WEBGPU_MAX_SSBO) ? bound_ssbo_[slot] : nullptr;
  }
  WebGPUTexture *bound_image_get(int slot) const
  {
    return (slot >= 0 && slot < WEBGPU_MAX_IMAGE) ? bound_image_[slot] : nullptr;
  }
  WebGPUTexture *bound_tex_get(int slot) const
  {
    return (slot >= 0 && slot < WEBGPU_MAX_TEX) ? bound_tex_[slot] : nullptr;
  }
  WGPUBuffer bound_ubo_get(int slot) const
  {
    return (slot >= 0 && slot < WEBGPU_MAX_UBO) ? bound_ubo_[slot] : nullptr;
  }
  WGPURenderPassEncoder render_pass() const { return render_pass_; }
  /* Submit recorded commands to the queue (ends any active pass first). */
  void submit();

  /* Debug capture: remember the FIRST content color texture (first-wins). AddRefs
   * the WGPU handle so it stays valid even after the pooled gpu::Texture is freed.
   * Implemented in the .cc (needs the full WebGPUTexture type). */
  void set_capture_target(WebGPUTexture *t);
  bool has_capture_target() const { return capture_src_ != nullptr; }

  /* Synchronous GPU->host readback of a texture region. Copies the texture to a
   * mappable buffer, submits, and (on Emscripten, via Asyncify) pumps the browser
   * event loop until the async map completes, then converts the texels into
   * r_data in the requested CPU format. Rows are flipped to OpenGL (bottom-up)
   * convention to match what Blender's readback callers expect. Returns false if
   * the format is unsupported or the map fails. This is what makes EEVEE's
   * Film::read_result / the `gpu` module's read_color actually return pixels. */
  /* Block on a buffer map via JSPI (wgpuInstanceWaitAny) and copy it out. */
  bool map_read_sync(WGPUBuffer buf, size_t size, void *dst);
  /* Synchronous small readback (select-id sample, depth pick): copy region →
   * scratch buffer → blocking map → convert. Only formats those callers use. */
  bool read_small_sync(WGPUTexture tex,
                       WGPUTextureFormat fmt,
                       int x,
                       int y,
                       int w,
                       int h,
                       eGPUDataFormat dst_format,
                       int channels,
                       void *r_data,
                       int layer = 0,
                       int mip = 0);
  bool read_color_sync(WGPUTexture tex,
                       WGPUTextureFormat fmt,
                       int x,
                       int y,
                       int w,
                       int h,
                       eGPUDataFormat dst_format,
                       int channels,
                       void *r_data,
                       int layer = 0,
                       int mip = 0);
  /* Look up the storage image (read/write image2D) currently bound to `name` for
   * `shader`. EEVEE's world/background fragment shader writes its radiance into
   * the "rp_color_img" storage image rather than the framebuffer color
   * attachment, so this lets the debug capture read the actually-rendered color
   * without the (compute) film composite. Returns nullptr if unbound. */
  WebGPUTexture *bound_storage_image(WebGPUShader *shader, const char *name);

  /* Assemble a group-0 bind group for `shader` from the current binding tables,
   * driven by the WGSL-parsed `bindings` (the exact set the pipeline expects),
   * using the pipeline's auto-derived layout `bgl`. Returns null if no device or
   * if a required resource is unbound (the draw is then skipped). */
  WGPUBindGroup build_bind_group(WebGPUShader *shader,
                                 const std::vector<WgslResourceBinding> &bindings,
                                 WGPUBindGroupLayout bgl,
                                 BindDynOffsets *r_dyn = nullptr);

  /* A cached default linear/repeat sampler used for all sampler bindings until
   * per-binding sampler state is wired. */
  WGPUSampler default_sampler();
  WGPUSampler nearest_sampler();
  /* Render-pass downsample chain filling mips 1..N of a color texture from mip 0
   * (linear 2x2 average per level, per layer). Uint/depth formats are skipped. */
  void generate_mipmaps(WebGPUTexture *t);

  /* --- Window backbuffer + canvas presentation (GUI path) -----------------
   * back_left gets a real color+depth attachment sized to the GHOST window
   * (backbuffer_ensure, called from the window's activateDrawingContext); at
   * swapBuffers the color texture is copied into the canvas' WebGPU surface
   * current texture (present_backbuffer). The frame appears when the thread
   * yields to the browser event loop (emscripten-driven WM main loop). */
  void backbuffer_ensure(int w, int h);
  void present_backbuffer(int w, int h);

 private:
  WGPUSurface surface_ = nullptr;
  int surface_w_ = 0, surface_h_ = 0;
  gpu::Texture *backbuffer_color_ = nullptr;
  gpu::Texture *backbuffer_depth_ = nullptr;
  int backbuffer_w_ = 0, backbuffer_h_ = 0;
  WGPUBuffer null_attr_buffer_ = nullptr;
  /* Transient uniform arena (see uniform_arena_alloc). The cursor RESETS every
   * present (uniform_arena_reset) so push-constant slices land at the SAME
   * offsets frame after frame — that is what lets the bind-group cache hit for
   * push-constant groups (keys include the offset). The epoch invalidates
   * shader-cached slices so no one keeps pointing into reused space. */
  WGPUBuffer uni_arena_buf_ = nullptr;
  uint64_t uni_arena_off_ = 0, uni_arena_cap_ = 0;
  uint32_t uni_arena_epoch_ = 1;
  /* Full chunks stay valid until the epoch ends: shaders cache their push-constant slice (and the
   * chunk handle) per epoch, so a chunk may only be destroyed at uniform_arena_reset(). */
  std::vector<WGPUBuffer> uni_arena_retired_;

 public:
  /* 1x1 dummy storage view of an exact format/dimension, for bindings whose
   * bound texture format mismatches the shader's declared image format. */
  WGPUTextureView dummy_storage_view(WGPUTextureFormat format,
                                     WGPUTextureViewDimension dim,
                                     uint32_t variant = 0);
  /* Snapshot copy of a texture for sampled bindings that alias a storage-written
   * texture in the same bind group (invalid in WebGPU; GL semantics = read the
   * pre-pass contents). */
  WGPUTextureView snapshot_for_sampling(WebGPUTexture *t);
  /* Copy of a buffer for the read-only side of a writable alias. */
  WGPUBuffer snapshot_buffer(WGPUBuffer src, uint64_t size);
  /* Small zeroed vertex buffer feeding shader input locations the batch's VBOs
   * don't cover (bound at arrayStride 0 → every vertex reads zeros, matching
   * GL's default-attribute behavior; WebGPU rejects unfed locations outright). */
  WGPUBuffer null_attr_buffer();
  /* Bump allocator over rolling uniform-buffer chunks for per-draw uniform data
   * (push constants). Each caller gets a fresh 256-B-aligned slice, so
   * queue.writeBuffer into it can NEVER retroactively affect an already
   * recorded draw — no pass flush/submit needed. Full chunks are released (bind
   * groups/command buffers keep them alive until the GPU is done). */
  bool uniform_arena_alloc(uint64_t size, WGPUBuffer &r_buf, uint64_t &r_off);
  uint32_t uniform_arena_epoch() const
  {
    return uni_arena_epoch_;
  }
  void uniform_arena_reset()
  {
    /* Safe to reuse: everything referencing the old slices was submitted at
     * present, and queue writes enqueued afterwards execute after them. */
    uni_arena_off_ = 0;
    uni_arena_epoch_++;
    for (WGPUBuffer b : uni_arena_retired_) {
      wgpu_buffer_free(b);
    }
    uni_arena_retired_.clear();
  }
  /* Async self-readback of the film texture (GUI F12): pixels are injected
   * into the RenderResult from the main loop when the map completes. */
  void film_capture_async(WGPUTexture tex, WGPUTextureFormat fmt, int w, int h);
  /* Viewport depth snapshot (3D cursor / auto-depth picking). Depth cannot be read back
   * synchronously (no JSPI), so a CPU copy of the viewport depth texture is refreshed
   * asynchronously after the viewport has been idle; FrameBuffer::read(DEPTH) serves from it.
   * depth_snapshot_tick() runs once per present; depth_snapshot_capture() from the idle timer. */
  void depth_snapshot_tick();
  void depth_snapshot_capture();
  /* --- Occlusion queries (gizmo picking) ---------------------------------
   * WebGPU occlusion results only arrive via an ASYNC buffer map, but
   * QueryPool::get_occlusion_result is synchronous. We return the PREVIOUS
   * completed round's results (a global cache) and kick off the resolve+map for
   * the current round. Gizmo highlight re-runs on every cursor move, so results
   * are one mouse-event stale — imperceptible, and it makes gizmo hover/drag
   * functional. */
  WGPUQuerySet occ_qset_ = nullptr; /* Persistent; attached while a pool is active. */
  uint32_t occ_qset_capacity_ = 0;
  bool occ_attach_ = false;
  int occ_query_pending_ = -1; /* Query index between pool begin/end_query. */
  bool occ_query_open_in_pass_ = false;
  WGPUQuerySet occlusion_query_set() const { return occ_attach_ ? occ_qset_ : nullptr; }
  void occlusion_pool_begin(uint32_t capacity);
  void occlusion_query_begin(int index) { occ_query_pending_ = index; }
  void occlusion_query_end();
  /* Called by both draw paths right before recording their draw. */
  void occlusion_query_draw_hook()
  {
    if (occ_query_pending_ >= 0 && occ_query_pending_ < int(occ_qset_capacity_) &&
        !occ_query_open_in_pass_ && render_pass_ != nullptr && occ_attach_ &&
        occ_qset_ != nullptr)
    {
      wgpuRenderPassEncoderBeginOcclusionQuery(render_pass_, uint32_t(occ_query_pending_));
      occ_query_open_in_pass_ = true;
    }
  }
  /* Resolve + async map the current round; fill r_values from the last
   * COMPLETED round. Ends the pool (detaches the query set from passes). */
  void occlusion_pool_read(uint32_t count, uint32_t *r_values);

  /* Drop all cached bind groups (called when a shader dies — its pointer keys
   * the cache and a recycled allocation must not hit stale groups). */
  void clear_bind_group_cache()
  {
    for (auto &kv : bind_group_cache_) {
      cached_bind_group_release(kv.second);
    }
    bind_group_cache_.clear();
  }
  static void clear_bind_group_cache_all_contexts();
  WebGPUTexture *bound_texture_for_binding(ShaderInterface *iface,
                                           const std::vector<WgslResourceBinding> &bindings,
                                           uint32_t binding);

  /* Build an explicit bind-group layout (+ pipeline layout wrapping it) from the
   * WGSL-parsed bindings, so the pipeline's layout contains EXACTLY those bindings
   * (auto-layout strips unused ones → mismatch with our complete bind group). The
   * caller owns the returned objects and should release them after pipeline
   * creation (the pipeline retains its own refs). */
  WGPUBindGroupLayout make_bind_group_layout(const std::vector<WgslResourceBinding> &bindings,
                                             bool is_compute,
                                             ShaderInterface *iface = nullptr);
  WGPUPipelineLayout make_pipeline_layout(WGPUBindGroupLayout bgl);

  /* --- resource binding tables --- */
  /* The bind tables hold REAL WebGPU references: Blender releases/reallocates
   * buffers while they are still bound (GL name semantics tolerate this), so
   * without a ref the table would hand stale handles to CreateBindGroup. */
  static void table_set(WGPUBuffer &slot_ref, WGPUBuffer buf)
  {
    if (slot_ref == buf) {
      return;
    }
    if (buf) {
      wgpuBufferAddRef(buf);
    }
    if (slot_ref) {
      wgpuBufferRelease(slot_ref);
    }
    slot_ref = buf;
  }
  void bind_ubo(int slot, WGPUBuffer buf)
  {
    if (slot >= 0 && slot < WEBGPU_MAX_UBO) {
      table_set(bound_ubo_[slot], buf);
    }
  }
  void unbind_ubo(int slot)
  {
    if (slot >= 0 && slot < WEBGPU_MAX_UBO) {
      table_set(bound_ubo_[slot], nullptr);
    }
  }
  void bind_ssbo(int slot, WGPUBuffer buf)
  {
    if (slot >= 0 && slot < WEBGPU_MAX_SSBO) {
      table_set(bound_ssbo_[slot], buf);
    }
  }
  void bind_texel_buffer(int slot, WGPUBuffer buf, int element_bytes)
  {
    if (slot >= 0 && slot < WEBGPU_MAX_TEX) {
      table_set(bound_texbuf_[slot], buf);
      bound_texbuf_bytes_[slot] = uint8_t(element_bytes);
    }
  }
  void unbind_ssbo(int slot)
  {
    if (slot >= 0 && slot < WEBGPU_MAX_SSBO) {
      table_set(bound_ssbo_[slot], nullptr);
    }
  }
  void bind_texture_state(int unit, GPUSamplerState state)
  {
    if (unit >= 0 && unit < WEBGPU_MAX_TEX) {
      bound_tex_state_[unit] = state;
    }
  }
  GPUSamplerState bound_tex_state_get(int slot) const
  {
    return (slot >= 0 && slot < WEBGPU_MAX_TEX) ? bound_tex_state_[slot] : GPUSamplerState();
  }
  /* Sampler matching a GPUSamplerState (cached; see webgpu_context.cc). */
  WGPUSampler sampler_for_state(GPUSamplerState state);
  void bind_texture(int unit, WebGPUTexture *tex)
  {
    if (unit >= 0 && unit < WEBGPU_MAX_TEX) {
      bound_tex_[unit] = tex;
    }
  }
  /* Blender frees resources without unbinding (harmless in GL). Scrub stale
   * pointers from the binding tables so draw-time bind-group assembly never
   * touches destroyed WebGPU objects. Called from the resource destructors. */
  void scrub_texture(WebGPUTexture *tex)
  {
    for (int i = 0; i < WEBGPU_MAX_TEX; i++) {
      if (bound_tex_[i] == tex) {
        bound_tex_[i] = nullptr;
      }
    }
    for (int i = 0; i < WEBGPU_MAX_IMAGE; i++) {
      if (bound_image_[i] == tex) {
        bound_image_[i] = nullptr;
      }
    }
  }
  void scrub_buffer(WGPUBuffer buf)
  {
    if (buf == nullptr) {
      return;
    }
    for (int i = 0; i < WEBGPU_MAX_UBO; i++) {
      if (bound_ubo_[i] == buf) {
        table_set(bound_ubo_[i], nullptr);
      }
    }
    for (int i = 0; i < WEBGPU_MAX_SSBO; i++) {
      if (bound_ssbo_[i] == buf) {
        table_set(bound_ssbo_[i], nullptr);
      }
    }
    for (int i = 0; i < WEBGPU_MAX_TEX; i++) {
      if (bound_texbuf_[i] == buf) {
        table_set(bound_texbuf_[i], nullptr);
      }
    }
  }
  /* Texture wrappers are CPU objects: a freed wrapper leaves a dangling pointer
   * in EVERY context's table, not just the active one. */
  static void scrub_texture_all_contexts(WebGPUTexture *tex);
  void bind_image(int unit, WebGPUTexture *tex)
  {
    if (unit >= 0 && unit < WEBGPU_MAX_IMAGE) {
      bound_image_[unit] = tex;
    }
  }
  WGPUBuffer ubo_at(int slot) const { return bound_ubo_[slot]; }
  WGPUBuffer ssbo_at(int slot) const { return bound_ssbo_[slot]; }
  WebGPUTexture *tex_at(int unit) const { return bound_tex_[unit]; }
  WebGPUTexture *image_at(int unit) const { return bound_image_[unit]; }

  /* --- pipeline cache --- */
  WGPURenderPipeline render_pipeline_get(uint64_t key) const
  {
    auto it = render_pipelines_.find(key);
    return it == render_pipelines_.end() ? nullptr : it->second;
  }
  void render_pipeline_put(uint64_t key, WGPURenderPipeline p) { render_pipelines_[key] = p; }
  WGPUComputePipeline compute_pipeline_get(uint64_t key) const
  {
    auto it = compute_pipelines_.find(key);
    return it == compute_pipelines_.end() ? nullptr : it->second;
  }
  void compute_pipeline_put(uint64_t key, WGPUComputePipeline p) { compute_pipelines_[key] = p; }
};

/* Monotonic content-generation counter for vertex/index buffers (cache invalidation). */
uint64_t webgpu_buffer_gen_next();

/* Viewport depth snapshot hooks (webgpu_context.cc). */
void wgpu_depth_snapshot_note_written(const WebGPUTexture *tex);
void wgpu_depth_snapshot_forget(const WebGPUTexture *tex);
bool wgpu_depth_snapshot_read(const WebGPUTexture *tex, const int area[4], float *r_data);

}  // namespace blender::gpu

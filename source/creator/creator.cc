/* SPDX-FileCopyrightText: 2001-2002 NaN Holding BV. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup creator
 */

#include <cstdio>
#include <cstdlib>
#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#  include <emscripten/wasmfs.h>
#  include <pthread.h>
#  include <sys/stat.h>
/* Zero-copy asset provider backend (demo/provider_backend.cpp, from gecko-wasm).
 * Mounted at /assets when the demo sets BLENDER_WEB_ASSET_PROVIDER. */
extern "C" backend_t wasmfs_create_provider_backend(int mountId);

/* PROXY_TO_PTHREAD device bootstrap (demo/provider path only). gecko's async
 * FsProvider requires the engine OFF the browser main thread, so the demo runs
 * blender's main() on a pthread. But emdawnwebgpu WebGPU objects are per-thread
 * JS objects: a device made on the main thread is unusable here. So acquire the
 * device ON this pthread (async requestAdapter/requestDevice), stash it in this
 * thread's Module.preinitializedWebGPUDevice (which the WebGPU backend then
 * imports via emscripten_webgpu_get_device), and re-enter main(). blender's
 * init is synchronous and can't await, so main() kicks the request off, keeps
 * the runtime live, and returns; the spontaneous device callback re-invokes
 * main() once the device is ready. */
int main(int argc, const char **argv); /* forward decl for the re-entry below */
static int g_web_bargc = 0;
static const char **g_web_bargv = nullptr;
static bool g_web_device_ready = false;
extern "C" EMSCRIPTEN_KEEPALIVE void blender_web_device_lost()
{
  /* stderr is forwarded to the page (Module.printErr) — demo/src/main.js scan() shows a banner. */
  fprintf(stderr, "WEBGPU_LOST device lost\n");
  fflush(stderr);
}
extern "C" EMSCRIPTEN_KEEPALIVE void blender_web_device_ready()
{
  g_web_device_ready = true;
  main(g_web_bargc, g_web_bargv);
}

/* Mount a JS FsProvider (Module.geckoProviders[mountId]) read-only at `path` at
 * runtime — used by the drag-drop open-folder fallback (demo/src/main.js) so a
 * dropped folder is read on demand through the ProviderBackend rather than
 * copied. Returns 0 on success. */
extern "C" EMSCRIPTEN_KEEPALIVE int blender_web_mount_provider(int mountId, const char *path)
{
  backend_t b = wasmfs_create_provider_backend(mountId);
  if (b == nullptr) {
    return -1;
  }
  return wasmfs_create_directory(path, 0555, b);
}
#endif
#include <cstring>

#ifdef WIN32
#  ifdef WIN32_LEAN_AND_MEAN
#    undef WIN32_LEAN_AND_MEAN
#  endif
#  include "utfconv.hh"
#  include <windows.h>
#  ifdef WITH_CPU_CHECK
#    pragma comment(linker, "/include:cpu_check_win32")
#  endif
#endif

#if defined(WITH_TBB_MALLOC) && defined(_MSC_VER) && defined(NDEBUG)
#  pragma comment(lib, "tbbmalloc_proxy.lib")
#  pragma comment(linker, "/include:__TBB_malloc_proxy")
#endif

#include "MEM_guardedalloc.h"

#include "CLG_log.h"

#include "DNA_genfile.h"

#include "BLI_endian_defines.hh"
#include "BLI_fftw.hh"
#include "BLI_path_utils.hh"
#include "BLI_string.hh"
#include "BLI_system.hh"
#include "BLI_task_c.hh"
#include "BLI_threads.hh"
#include "BLI_utildefines.hh"

/* Mostly initialization functions. */
#include "BKE_appdir.hh"
#include "BKE_blender.hh"
#include "BKE_brush.hh"
#include "BKE_callbacks.hh"
#include "BKE_context.hh"
#include "BKE_cpp_types.hh"
#include "BKE_global.hh"
#include "BKE_idtype.hh"
#include "BKE_material.hh"
#include "BKE_modifier.hh"
#include "BKE_node.hh"
#include "BKE_particle.h"
#include "BKE_shader_fx.hh"
#include "BKE_sound.hh"
#include "BKE_vfont.hh"
#include "BKE_volume.hh"

#ifndef WITH_PYTHON_MODULE
#  include "BLI_args.hh"
#endif

#include "DEG_depsgraph.hh"

#include "IMB_imbuf.hh" /* For #IMB_init. */

#include "MOV_util.hh"

#include "RE_engine.h"
#include "RE_texture.h"

#include "ED_datafiles.h"

#include "SEQ_modifier.hh"

#include "WM_api.hh"

#include "RNA_define.hh"

#include "FN_init.hh"

#ifdef WITH_OPENGL_BACKEND
#  include "GPU_compilation_subprocess.hh"
#endif

#ifdef WITH_FREESTYLE
#  include "FRS_freestyle.h"
#endif

#include <csignal>

#ifdef __FreeBSD__
#  include <floatingpoint.h>
#endif

#ifdef WITH_BINRELOC
#  include "binreloc.h"
#endif

#ifdef WITH_LIBMV
#  include "libmv-capi.h"
#endif

#ifdef WITH_CYCLES
#  include "CCL_api.h"
#endif

#if defined(WITH_PYTHON_MODULE) && defined(__APPLE__)
/* Environment is not available in macOS shared libraries. */
#  include <crt_externs.h>
char **environ = nullptr;
#endif

#if defined(WITH_TBB_MALLOC) && defined(__linux__)
#  include <tbb/scalable_allocator.h>
#endif

#include "creator_intern.h" /* Own include. */

BLI_STATIC_ASSERT(ENDIAN_ORDER == L_ENDIAN, "Blender only builds on little endian systems")

/* -------------------------------------------------------------------- */
/** \name GMP Allocator Workaround
 * \{ */

#if (defined(WITH_TBB_MALLOC) && defined(_MSC_VER) && defined(NDEBUG) && defined(WITH_GMP)) || \
    defined(DOXYGEN)
#  include "gmp.h"
#  include "tbb/scalable_allocator.h"

void *gmp_alloc(size_t size)
{
  return scalable_malloc(size);
}
void *gmp_realloc(void *ptr, size_t /*old_size*/, size_t new_size)
{
  return scalable_realloc(ptr, new_size);
}

void gmp_free(void *ptr, size_t /*size*/)
{
  scalable_free(ptr);
}
/**
 * Use TBB's scalable_allocator on Windows.
 * `TBBmalloc` correctly captures all allocations already,
 * however, GMP is built with MINGW since it doesn't build with MSVC,
 * which TBB has issues hooking into automatically.
 */
void gmp_blender_init_allocator()
{
  mp_set_memory_functions(gmp_alloc, gmp_realloc, gmp_free);
}
#endif

/** \} */

/* -------------------------------------------------------------------- */
/** \name Local Defines
 * \{ */

/* When building as a Python module, don't use special argument handling
 * so the module loading logic can control the `argv` & `argc`. */
#if defined(WIN32) && !defined(WITH_PYTHON_MODULE)
#  define USE_WIN32_UNICODE_ARGS
#endif

/** \} */

namespace blender {

/* -------------------------------------------------------------------- */
/** \name Local Application State
 * \{ */

/* Written to by `creator_args.cc`. */
ApplicationState app_state = []() {
  ApplicationState app_state{};
  app_state.signal.use_crash_handler = true;
  app_state.signal.use_console_crash_handler = false;
  app_state.signal.use_abort_handler = true;
  app_state.exit_code_on_error.python = 0;
  app_state.main_arg_deferred = nullptr;
  return app_state;
}();

/** \} */

/* -------------------------------------------------------------------- */
/** \name Application Level Callbacks
 *
 * Initialize callbacks for the modules that need them.
 * \{ */

static void callback_mem_error(const char *errorStr)
{
  fputs(errorStr, stderr);
  fflush(stderr);
}

static void main_callback_setup()
{
  /* Error output from the guarded allocation routines. */
  MEM_set_error_callback(callback_mem_error);
}

/** Data to free when Blender exits early on. */
struct CreatorAtExitData_EarlyExit {
  bContext *C;
};

/** Free data on early exit (if Python calls `sys.exit()` while parsing args for eg). */
struct CreatorAtExitData {
#ifndef WITH_PYTHON_MODULE
  bArgs *ba;
#endif

#ifdef USE_WIN32_UNICODE_ARGS
  char **argv;
  int argv_num;
#endif

  /**
   * When non-null, run additional exit logic.
   * Cleared once early initialization is over.
   */
  CreatorAtExitData_EarlyExit *early_exit = nullptr;
};

static void callback_main_atexit(void *user_data)
{
  CreatorAtExitData *app_init_data = static_cast<CreatorAtExitData *>(user_data);

#ifndef WITH_PYTHON_MODULE
  if (app_init_data->ba) {
    BLI_args_destroy(app_init_data->ba);
    app_init_data->ba = nullptr;
  }
#endif

#ifdef USE_WIN32_UNICODE_ARGS
  if (app_init_data->argv) {
    while (app_init_data->argv_num) {
      free((void *)app_init_data->argv[--app_init_data->argv_num]);
    }
    free((void *)app_init_data->argv);
    app_init_data->argv = nullptr;
  }
#endif

  if (CreatorAtExitData_EarlyExit *early_exit = app_init_data->early_exit) {
    CTX_free(early_exit->C);

    DEG_free_node_types();

    BKE_blender_globals_clear();
    BKE_appdir_exit();

    DNA_sdna_current_free();

    CLG_exit();
  }
}

static void callback_clg_fatal(void *fp)
{
  BLI_system_backtrace(static_cast<FILE *>(fp));
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name LD_PRELOAD for Linux
 * \{ */

static void restore_ld_preload()
{
  /* LD_PRELOAD may have been modified on startup for Blender. However
   * we don't want it for other executables launched from Blender. */
  const char *restore_ld_preload = BLI_getenv("BLENDER_RESTORE_LD_PRELOAD");
  if (restore_ld_preload) {
    BLI_setenv("LD_PRELOAD", restore_ld_preload);
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Blender as a Stand-Alone Python Module (bpy)
 *
 * While not officially supported, this can be useful for Python developers.
 * See: https://developer.blender.org/docs/handbook/building_blender/python_module/
 * \{ */

#ifdef WITH_PYTHON_MODULE
static void *main_python_evil_C = nullptr;

/* Called in `bpy_interface.cc` when building as a Python module. */
int main_python_enter(int argc, const char **argv);

void main_python_exit()
{
  WM_exit_ex((bContext *)main_python_evil_C, true, false);
  main_python_evil_C = nullptr;
}

/* Rename the `main(..)` function, allowing Python initialization to call it. */
#  define main blender::main_python_enter
#endif /* WITH_PYTHON_MODULE */

/** \} */

}  // namespace blender

/* -------------------------------------------------------------------- */
/** \name Main Function
 * \{ */

#if defined(__APPLE__)
extern "C" int GHOST_HACK_getFirstFile(char buf[]);
#endif

/**
 * Blender's main function responsibilities are:
 * - setup subsystems.
 * - handle arguments.
 * - run #WM_main() event loop,
 *   or exit immediately when running in background-mode.
 */
int main(int argc,
#ifdef USE_WIN32_UNICODE_ARGS
         const char ** /*argv_c*/
#else
         const char **argv
#endif
)
{
  using namespace blender;

#ifdef __EMSCRIPTEN__
  /* PROXY_TO_PTHREAD device bootstrap (see top of file). On first entry (demo
   * provider mode, device not yet acquired on this pthread), asynchronously
   * request an adapter+device HERE, stash it in this thread's
   * Module.preinitializedWebGPUDevice, and re-enter main() from the callback.
   * Same feature/limit request as demo/src/main.js. */
  if (getenv("BLENDER_WEB_ASSET_PROVIDER") && !g_web_device_ready) {
    g_web_bargc = argc;
    g_web_bargv = argv;
    /* Test aid (P3): WGPU_CAP_MAX_BUFFER / WGPU_CAP_MAX_SSBO (bytes) clamp the device limits to what
     * iPad Safari offers (~268435456 / 134217728), to reproduce its high-poly limits on a desktop GPU. */
    const char *cap_buf_env = getenv("WGPU_CAP_MAX_BUFFER");
    const char *cap_ssbo_env = getenv("WGPU_CAP_MAX_SSBO");
    const double cap_buf = cap_buf_env ? atof(cap_buf_env) : 0.0;
    const double cap_ssbo = cap_ssbo_env ? atof(cap_ssbo_env) : 0.0;
    /* WGPU_CAP_MAX_SSBO_STAGE (count): clamp maxStorageBuffersPerShaderStage (Safari may offer far
     * fewer than a desktop GPU; texel buffers and EEVEE use many storage bindings). */
    const char *cap_stage_env = getenv("WGPU_CAP_MAX_SSBO_STAGE");
    const double cap_stage = cap_stage_env ? atof(cap_stage_env) : 0.0;
    EM_ASM({
      (async () => {
        try {
          const adapter = await navigator.gpu.requestAdapter();
          const wanted = ["dual-source-blending", "float32-filterable",
            "rg11b10ufloat-renderable", "bgra8unorm-storage", "shader-f16",
            "depth32float-stencil8", "texture-formats-tier1", "texture-formats-tier2",
            "indirect-first-instance", "clip-distances"];
          const requiredFeatures = wanted.filter((f) => adapter.features.has(f));
          const wantLimits = ["maxStorageTexturesPerShaderStage",
            "maxStorageBuffersPerShaderStage", "maxSampledTexturesPerShaderStage",
            "maxSamplersPerShaderStage", "maxUniformBuffersPerShaderStage",
            "maxComputeWorkgroupSizeX", "maxComputeWorkgroupSizeY",
            "maxComputeWorkgroupSizeZ", "maxComputeInvocationsPerWorkgroup",
            "maxComputeWorkgroupStorageSize", "maxComputeWorkgroupsPerDimension",
            "maxStorageBufferBindingSize", "maxUniformBufferBindingSize",
            "maxBufferSize", "maxBindGroups", "maxBindingsPerBindGroup",
            "maxTextureDimension2D", "maxTextureDimension3D", "maxTextureArrayLayers",
            "maxColorAttachments", "maxColorAttachmentBytesPerSample"];
          const requiredLimits = {};
          for (const k of wantLimits) {
            const v = adapter.limits[k];
            if (v !== undefined) requiredLimits[k] = v;
          }
          if ($0 > 0) requiredLimits.maxBufferSize = Math.min(requiredLimits.maxBufferSize, $0);
          if ($1 > 0) {
            requiredLimits.maxStorageBufferBindingSize =
                Math.min(requiredLimits.maxStorageBufferBindingSize, $1);
          }
          if ($2 > 0) {
            requiredLimits.maxStorageBuffersPerShaderStage =
                Math.min(requiredLimits.maxStorageBuffersPerShaderStage, $2);
          }
          console.log("WGPU_LIMITS maxBufferSize=" + requiredLimits.maxBufferSize +
                      " maxStorageBufferBindingSize=" + requiredLimits.maxStorageBufferBindingSize +
                      " maxStorageBuffersPerShaderStage=" + requiredLimits.maxStorageBuffersPerShaderStage);
          Module["preinitializedWebGPUDevice"] =
              await adapter.requestDevice({ requiredFeatures, requiredLimits });
          /* Device loss (iPad memory pressure / app switch) freezes the canvas silently; the page
           * (demo/src/main.js scan) turns this line into a "GPU lost, reload" banner. */
          const gdev = Module["preinitializedWebGPUDevice"];
          /* Dawn validation errors are otherwise only a (rate limited) browser warning. */
          let gerrs = 0;
          const gerr_ch = new BroadcastChannel("blender-gpuerr");
          gdev.addEventListener("uncapturederror", (e) => {
            if (gerrs++ < 30) {
              console.log("WGPU_UNCAPTURED " + String(e.error.message).slice(0, 400));
              gerr_ch.postMessage(String(e.error.message).slice(0, 400));
            }
          });
          /* Sprint "iPad vydrží" M0/M2: live GPU buffer/texture counters for the telemetry
           * (demo/src/telemetry.js, channel "blender-gpumem"). A resource counts as live until
           * destroy() was called or the JS object was garbage collected (gc_* = freed only by GC,
           * which Safari does much later than Chrome). */
          const gmem = new BroadcastChannel("blender-gpumem");
          const gst = {buf: 0, buf_mb: 0, tex: 0, tex_mb: 0, buf_new: 0, buf_destroyed: 0, buf_gc: 0,
                       tex_new: 0, tex_destroyed: 0, tex_gc: 0, bg_new: 0, view_new: 0, lag_ms: 0, lag_max: 0,
                       /* T2 (A/B backpressure vs JS leak): */
                       submits: 0, completed: 0, q_depth: 0, q_depth_max: 0, write_mb: 0, write_mb_s: 0,
                       map_n: 0, map_mb: 0, pipelines_created: 0, wgpu_live: 0, pending_free: 0};
          let gwbytes = 0, gmbytes = 0, gwlast = 0, gwprev = 0;
          let glast = 0, gbytes = {buf: 0, tex: 0};
          const gpost = (force, extra) => {
            const n = performance.now();
            if (!force && n - glast < 1000) return;
            glast = n;
            gst.buf_mb = Math.round(gbytes.buf / 1048576); gst.tex_mb = Math.round(gbytes.tex / 1048576);
            gst.q_depth = gst.submits - gst.completed; if (gst.q_depth > gst.q_depth_max) gst.q_depth_max = gst.q_depth;
            gst.write_mb = Math.round(gwbytes / 1048576); gst.map_mb = Math.round(gmbytes / 1048576);
            if (gwlast) gst.write_mb_s = +((gwbytes - gwprev) / 1048576 / ((n - gwlast) / 1000)).toFixed(2);
            gwlast = n; gwprev = gwbytes;
            try { gst.pending_free = _wgpu_pending_free_n(); } catch (e) {}
            try { if (typeof WebGPU !== "undefined") gst.wgpu_live = Object.keys(WebGPU.Internals.jsObjects).length; } catch (e) {}
            gmem.postMessage(Object.assign({}, gst, extra || {}));
          };
          const grecs = new WeakMap();
          const greg = new FinalizationRegistry((r) => {
            if (!r.dead) { r.dead = true; gst[r.k]--; gbytes[r.k] -= r.size; gst[r.k + "_gc"]++; }
          });
          const gtrack = (k, obj, size) => {
            const r = {k, size, dead: false};
            grecs.set(obj, r); greg.register(obj, r);
            gst[k]++; gst[k + "_new"]++; gbytes[k] += size;
            gpost(false);
            return obj;
          };
          const gdestroy = (k, proto) => {
            const orig = proto.destroy;
            proto.destroy = function () {
              const r = grecs.get(this);
              if (r && !r.dead) { r.dead = true; gst[k]--; gbytes[k] -= r.size; gst[k + "_destroyed"]++; }
              gpost(false);
              return orig.apply(this, arguments);
            };
          };
          const gcb = gdev.createBuffer.bind(gdev), gct = gdev.createTexture.bind(gdev);
          /* IPAD_CRASH H1-A: creations per label/usage ("hist": count, hbytes: bytes). */
          gst.hist = {}; gst.hbytes = {}; gst.thist = {};
          gdev.createBuffer = (d) => {
            const hk = (d.label || "") + "/" + d.usage.toString(16);
            gst.hist[hk] = (gst.hist[hk] || 0) + 1; gst.hbytes[hk] = (gst.hbytes[hk] || 0) + d.size;
            return gtrack("buf", gcb(d), d.size);
          };
          gdev.createTexture = (d) => {
            const sz = d.size, w = sz.width !== undefined ? sz.width : sz[0], h = (sz.height !== undefined ? sz.height : sz[1]) || 1,
                  l = (sz.depthOrArrayLayers !== undefined ? sz.depthOrArrayLayers : sz[2]) || 1;
            const tk = (d.label || "") + "/" + d.format + "/" + w + "x" + h;
            gst.thist[tk] = (gst.thist[tk] || 0) + 1;
            return gtrack("tex", gct(d), w * h * l * 4 * (d.mipLevelCount > 1 ? 1.34 : 1));
          };
          /* IPAD_CRASH H1-0: bind group / view creation counters and GPU queue lag (every 2 s). */
          const gcbg = gdev.createBindGroup.bind(gdev);
          gdev.createBindGroup = (d) => { gst.bg_new++; return gcbg(d); };
          if (typeof GPUTexture !== "undefined") {
            const gcv = GPUTexture.prototype.createView;
            GPUTexture.prototype.createView = function () { gst.view_new++; return gcv.apply(this, arguments); };
          }
          /* T2: queue depth (submits - completed), CPU->GPU copy volume, mapAsync volume, pipelines. */
          {
            const qp = GPUQueue.prototype, qsub = qp.submit, qwb = qp.writeBuffer, qwt = qp.writeTexture;
            qp.submit = function () {
              gst.submits++;
              const r = qsub.apply(this, arguments);
              this.onSubmittedWorkDone().then(() => { gst.completed++; });
              return r;
            };
            qp.writeBuffer = function (b, o, d, dofs, sz) {
              gwbytes += d.byteLength === undefined ? 0 : (sz !== undefined ? sz * (d.BYTES_PER_ELEMENT || 1) : d.byteLength);
              return qwb.apply(this, arguments);
            };
            qp.writeTexture = function (dst, d) { gwbytes += d.byteLength || 0; return qwt.apply(this, arguments); };
            const bmap = GPUBuffer.prototype.mapAsync;
            GPUBuffer.prototype.mapAsync = function (m, o, sz) { gst.map_n++; gmbytes += sz !== undefined ? sz : this.size; return bmap.apply(this, arguments); };
            for (const fn of ["createRenderPipeline", "createComputePipeline", "createRenderPipelineAsync", "createComputePipelineAsync"]) {
              const o = gdev[fn].bind(gdev); gdev[fn] = (d) => { gst.pipelines_created++; return o(d); };
            }
          }
          setInterval(() => {
            const t = performance.now();
            gdev.queue.onSubmittedWorkDone().then(() => {
              gst.lag_ms = Math.round(performance.now() - t);
              if (gst.lag_ms > gst.lag_max) gst.lag_max = gst.lag_ms;
              gpost(false);
            });
          }, 2000);
          if ($3) {
            /* WGPU_DESTROYLOG=1 (debug): report every bind group / vertex / index buffer use of a
             * buffer that was already destroy()ed (it would invalidate the whole submit). */
            const gdead = new WeakSet(), gseen = {};
            const gdestroy0 = GPUBuffer.prototype.destroy;
            GPUBuffer.prototype.destroy = function () { gdead.add(this); return gdestroy0.apply(this, arguments); };
            const gwhine = (where, b) => {
              if (!gdead.has(b)) return;
              const k = where + " " + b.usage + " " + b.size;
              gseen[k] = (gseen[k] || 0) + 1;
              if (gseen[k] <= 3) console.log("DEADBUF " + where + " usage=0x" + b.usage.toString(16) + " size=" + b.size + " label='" + b.label + "' n=" + gseen[k]);
            };
            const gbg = gdev.createBindGroup.bind(gdev);
            const gbgs = new WeakMap();
            gdev.createBindGroup = (d) => {
              const bs = [];
              for (const e of d.entries) if (e.resource && e.resource.buffer) { gwhine("bindgroup", e.resource.buffer); bs.push(e.resource.buffer); }
              const g = gbg(d); gbgs.set(g, bs); return g;
            };
            const gsbg = GPURenderPassEncoder.prototype.setBindGroup;
            GPURenderPassEncoder.prototype.setBindGroup = function (i, g) { const bs = gbgs.get(g); if (bs) for (const b of bs) gwhine("setBindGroup" + i, b); return gsbg.apply(this, arguments); };
            for (const [P, fn, idx] of [[GPURenderPassEncoder.prototype, "setVertexBuffer", 1], [GPURenderPassEncoder.prototype, "setIndexBuffer", 0], [GPURenderPassEncoder.prototype, "drawIndirect", 0], [GPURenderPassEncoder.prototype, "drawIndexedIndirect", 0]]) {
              const o = P[fn]; P[fn] = function () { if (arguments[idx]) gwhine(fn, arguments[idx]); return o.apply(this, arguments); };
            }
          }
          if (typeof GPUBuffer !== "undefined") gdestroy("buf", GPUBuffer.prototype);
          if (typeof GPUTexture !== "undefined") gdestroy("tex", GPUTexture.prototype);
          gdev.lost.then(
              (i) => { gpost(true, {lost_reason: i.reason, lost_message: String(i.message).slice(0, 300)});
                       console.log("WEBGPU_LOST " + i.reason + " " + i.message); _blender_web_device_lost(); });
          _blender_web_device_ready();
        }
        catch (e) {
          /* _blender_web_device_ready re-enters main(), which establishes the
           * emscripten main loop by throwing 'unwind' — expected, not an error. */
          if (e !== "unwind") console.error("pthread WebGPU device acquire failed:", e);
        }
      })();
    }, cap_buf, cap_ssbo, cap_stage, getenv("WGPU_DESTROYLOG") != nullptr);
    emscripten_exit_with_live_runtime();
    return 0;
  }
#endif

  bContext *C;
#ifndef WITH_PYTHON_MODULE
  bArgs *ba;
#endif

  /* Ensure we free data on early-exit. */
  CreatorAtExitData app_init_data = {nullptr};
  BKE_blender_atexit_register(callback_main_atexit, &app_init_data);

  CreatorAtExitData_EarlyExit app_init_data_early_exit = {nullptr};
  app_init_data.early_exit = &app_init_data_early_exit;

/* Un-buffered `stdout` makes `stdout` and `stderr` better synchronized, and helps
 * when stepping through code in a debugger (prints are immediately
 * visible). However disabling buffering causes lock contention on windows
 * see #76767 for details, since this is a debugging aid, we do not enable
 * the un-buffered behavior for release builds. */
#ifndef NDEBUG
  setvbuf(stdout, nullptr, _IONBF, 0);
#endif

  restore_ld_preload();

#ifdef __EMSCRIPTEN__
  /* Release/demo builds (BLENDER_WEB_OPFS env, set by the demo page): mount
   * the browser's Origin Private File System at /opfs via the wasmfs OPFS
   * backend. The demo keeps ASSETS (extracted once by its setup screen) at
   * /opfs/assets and points HOME at /opfs/home for real persistence. The dev
   * fast-path (web/blender-gui.html) never sets the env and keeps using the
   * --preload-file MEMFS tree. */
  if (getenv("BLENDER_WEB_OPFS")) {
    /* wasmfs asserts if the OPFS backend is created on the main browser
     * thread (needs an async JS handshake) — do it on a pthread. The pool is
     * preloaded (PTHREAD_POOL_SIZE), so the thread starts while main blocks
     * in join. Subsequent main-thread file access proxies to the dedicated
     * OPFS worker. */
    auto mount_fn = [](void *) -> void * {
      backend_t opfs = wasmfs_create_opfs_backend();
      if (opfs == nullptr || wasmfs_create_directory("/opfs", 0777, opfs) != 0) {
        fprintf(stderr, "BLENDER_WEB_OPFS: mounting /opfs FAILED\n");
      }
      else {
        mkdir("/opfs/home", 0777); /* EEXIST is fine. */
      }
      return nullptr;
    };
    pthread_t mount_thread;
    if (pthread_create(&mount_thread, nullptr, +mount_fn, nullptr) == 0) {
      pthread_join(mount_thread, nullptr);
    }
    else {
      fprintf(stderr, "BLENDER_WEB_OPFS: mount thread creation FAILED\n");
    }
  }
  /* Zero-copy assets: mount the in-memory tar FsProvider at /assets before any
   * asset is read (BLENDER_SYSTEM_RESOURCES/PYTHON point there). The provider
   * (Module.geckoProviders[0], the decompressed tar indexed in demo/src/main.js)
   * is served by the WasmFS ProviderBackend (demo/provider_backend.cpp +
   * provider-fs.js, copied from gecko-wasm). Mirrors gecko's embed-init.cpp. */
  if (getenv("BLENDER_WEB_ASSET_PROVIDER")) {
    backend_t ab = wasmfs_create_provider_backend(0);
    if (ab == nullptr || wasmfs_create_directory("/assets", 0555, ab) != 0) {
      fprintf(stderr, "BLENDER_WEB_ASSET_PROVIDER: mounting /assets FAILED\n");
    }
  }
  /* wasmfs creates every --preload-file PARENT directory with a hardcoded
   * 0555 mode (S_IRUGO|S_IXUGO in wasmfs.cpp's preload flush, ignoring the
   * packager's canWrite flag). The python sysroot preload path runs through
   * /home/..., which leaves $HOME read-only: blend save (rename of the "@"
   * temp file) and config writes fail with EACCES. Re-open the HOME chain. */
  {
    chmod("/home", 0777);
    const char *home_dir = getenv("HOME");
    if (home_dir && home_dir[0] == '/') {
      char path_buf[1024];
      const size_t home_len = strlen(home_dir);
      if (home_len < sizeof(path_buf)) {
        memcpy(path_buf, home_dir, home_len + 1);
        for (char *p = path_buf + 1; *p; p++) {
          if (*p == '/') {
            *p = '\0';
            chmod(path_buf, 0777);
            *p = '/';
          }
        }
        mkdir(path_buf, 0777); /* Ensure HOME itself exists. */
        chmod(path_buf, 0777);
      }
    }
  }
#endif

#ifdef WIN32
#  ifdef USE_WIN32_UNICODE_ARGS
  /* Win32 Unicode Arguments. */
  {
    /* NOTE: Can't use `guardedalloc` allocation here, as it's not yet initialized
     * (it depends on the arguments passed in, which is what we're getting here!). */
    wchar_t **argv_16 = CommandLineToArgvW(GetCommandLineW(), &argc);
    app_init_data.argv = static_cast<char **>(malloc(argc * sizeof(char *)));
    for (int i = 0; i < argc; i++) {
      app_init_data.argv[i] = alloc_utf_8_from_16(argv_16[i], 0);
    }
    LocalFree(argv_16);

    /* Free on early-exit. */
    app_init_data.argv_num = argc;
  }
  const char **argv = const_cast<const char **>(app_init_data.argv);
#  endif /* USE_WIN32_UNICODE_ARGS */
#endif   /* WIN32 */

#if defined(WITH_OPENGL_BACKEND) && BLI_SUBPROCESS_SUPPORT
  if (STREQ(argv[0], "--compilation-subprocess")) {
    BLI_assert(argc == 2);
    GPU_compilation_subprocess_run(argv[1]);
    return 0;
  }
#endif

#if defined(WITH_TBB_MALLOC) && defined(__linux__)
  /* Enable huge pages for performance. */
  scalable_allocation_mode(TBBMALLOC_USE_HUGE_PAGES, 1);
#endif

  /* NOTE: Special exception for guarded allocator type switch:
   *       we need to perform switch from lock-free to fully
   *       guarded allocator before any allocation happened.
   */
  {
    int i;
    for (i = 0; i < argc; i++) {
      if (STR_ELEM(argv[i], "-d", "--debug", "--debug-memory", "--debug-all")) {
        printf("Switching to fully guarded memory allocator.\n");
        MEM_use_guarded_allocator();
        break;
      }
      if (STR_ELEM(argv[i], "--", "-c", "--command")) {
        break;
      }
    }
    MEM_init_memleak_detection();
  }

#ifdef BUILD_DATE
  {
    const time_t temp_time = build_commit_timestamp;
    const tm *tm = gmtime(&temp_time);
    if (tm) [[likely]] {
      strftime(build_commit_date, sizeof(build_commit_date), "%Y-%m-%d", tm);
      strftime(build_commit_time, sizeof(build_commit_time), "%H:%M", tm);
    }
    else {
      const char *unknown = "date-unknown";
      STRNCPY(build_commit_date, unknown);
      STRNCPY(build_commit_time, unknown);
    }
  }
#endif

  /* Initialize logging. */
  CLG_init();
  CLG_output_use_timestamp_set(true);
  CLG_output_use_memory_set(false);
  CLG_output_use_source_set(false);
  CLG_output_use_basename_set(false);
  CLG_fatal_fn_set(callback_clg_fatal);

  C = CTX_create();

  app_init_data_early_exit.C = C;

#ifdef WITH_PYTHON_MODULE
#  ifdef __APPLE__
  environ = *_NSGetEnviron();
#  endif

#  undef main
  main_python_evil_C = C;
#endif

#ifdef WITH_BINRELOC
  br_init(nullptr);
#endif

#ifdef WITH_LIBMV
  libmv_initLogging(argv[0]);
#endif

#if defined(WITH_TBB_MALLOC) && defined(_MSC_VER) && defined(NDEBUG) && defined(WITH_GMP)
  gmp_blender_init_allocator();
#endif

  main_callback_setup();

#if defined(__APPLE__) && !defined(WITH_PYTHON_MODULE) && !defined(WITH_HEADLESS)
  /* Patch to ignore argument finder gives us (PID?). */
  if (argc == 2 && STRPREFIX(argv[1], "-psn_")) {
    static char firstfilebuf[512];

    argc = 1;

    if (GHOST_HACK_getFirstFile(firstfilebuf)) {
      argc = 2;
      argv[1] = firstfilebuf;
    }
  }
#endif

#ifdef __FreeBSD__
  fpsetmask(0);
#endif

  /* Initialize path to executable. */
  BKE_appdir_program_path_init(argv[0]);

  BLI_threadapi_init();

  DNA_sdna_current_init();

  BKE_blender_globals_init(); /* `blender.cc` */

  BKE_cpp_types_init();
  fn::multi_function::register_common_functions();
  BKE_idtype_init();
  BKE_modifier_init();
  seq::modifiers_init();
  BKE_shaderfx_init();
  BKE_volumes_init();
  DEG_register_node_types();

  BKE_callback_global_init();

/* First test for background-mode (#Global.background). */
#ifndef WITH_PYTHON_MODULE
  ba = BLI_args_create(argc, argv); /* Skip binary path. */

  /* Ensure we free on early exit. */
  app_init_data.ba = ba;

  main_args_setup(C, ba, false);

  /* Parse environment handling arguments. */
  BLI_args_parse(ba, ARG_PASS_ENVIRONMENT, nullptr, nullptr);

#else
  /* Using preferences or user startup makes no sense for #WITH_PYTHON_MODULE. */
  G.factory_startup = true;
#endif

  /* After parsing #ARG_PASS_ENVIRONMENT such as `--env-*`,
   * since they impact `BKE_appdir` behavior. */
  BKE_appdir_init();

  /* After parsing number of threads argument. */
  BLI_task_scheduler_init();

  /* Initialize FFTW threading support. */
  fftw::initialize_float();

#ifndef WITH_PYTHON_MODULE
  /* The settings pass includes:
   * - Background-mode assignment (#Global.background), checked by other subsystems
   *   which may be skipped in background mode.
   * - The animation player may be launched which takes over argument passing,
   *   initializes the sub-systems it needs which have not yet been started.
   *   The animation player will call `exit(..)` too, so code after this call
   *   never runs when it's invoked.
   * - All the `--debug-*` flags.
   */
  BLI_args_parse(ba, ARG_PASS_SETTINGS, nullptr, nullptr);

  main_signal_setup();
#endif

  /* Continue with regular initialization, no need to use "early" exit. */
  app_init_data.early_exit = nullptr;

#ifdef WITH_CYCLES
  CCL_log_init();
  CCL_implicit_sharing_init();
#endif

  /* Set max open files to better handle production files that may use many
   * open geometry or texture cache file handles. After logging since it's used .*/
  BLI_system_max_open_files_ensure();

  /* Must be initialized after #BKE_appdir_init to account for color-management paths. */
  IMB_init();
  /* Keep after #ARG_PASS_SETTINGS since debug flags are checked. */
  MOV_init();

  /* After #ARG_PASS_SETTINGS arguments, this is so #WM_main_playanim skips #RNA_init. */
  RNA_init();

  RE_texture_rng_init();
  RE_engines_init();
  bke::node_system_init();

  BKE_brush_system_init();
  BKE_particle_init_rng();
  /* End second initialization. */

#if defined(WITH_PYTHON_MODULE) || defined(WITH_HEADLESS)
  /* Python module mode ALWAYS runs in background-mode (for now). */
  G.background = true;
  /* Manually using `--background` also forces the audio device. */
  BKE_sound_force_device("None");
#else
  if (G.background) {
    main_signal_setup_background();
  }
#endif

  /* Background render uses this font too. */
  BKE_vfont_builtin_register(datatoc_bfont_pfb, datatoc_bfont_pfb_size);

  /* Initialize FFMPEG if built in, also needed for background-mode if videos are
   * rendered via FFMPEG. */
  BKE_sound_init_once();

  BKE_materials_init();

#ifndef WITH_PYTHON_MODULE
  if (G.background == 0) {
    BLI_args_parse(ba, ARG_PASS_SETTINGS_GUI, nullptr, nullptr);
  }
  BLI_args_parse(ba, ARG_PASS_SETTINGS_FORCE, nullptr, nullptr);
#endif

  WM_init(C, argc, argv);

#ifndef WITH_PYTHON
  fprintf(stderr,
          "\n"
          "WARNING: Blender compiled without Python!\n"
          "This is not intended for typical usage.\n"
          "\n");
#endif

#ifdef WITH_FREESTYLE
  /* Initialize Freestyle. */
  FRS_init();
  FRS_set_context(C);
#endif

/* OK we are ready for it. */
#ifndef WITH_PYTHON_MODULE
  /* Handles #ARG_PASS_FINAL. */
  BLI_args_parse(ba, ARG_PASS_FINAL, main_args_handle_load_file, C);
#endif

  /* Explicitly free data allocated for argument parsing:
   * - `ba`
   * - `argv` on WIN32.
   */
  callback_main_atexit(&app_init_data);
  BKE_blender_atexit_unregister(callback_main_atexit, &app_init_data);

/* Paranoid, avoid accidental re-use. */
#ifndef WITH_PYTHON_MODULE
  ba = nullptr;
  (void)ba;
#endif

#ifdef USE_WIN32_UNICODE_ARGS
  argv = nullptr;
  (void)argv;
#endif

#ifndef WITH_PYTHON_MODULE
  if (G.background) {
    int exit_code;
    if (app_state.main_arg_deferred != nullptr) {
      exit_code = main_arg_deferred_handle();
      main_arg_deferred_free();
    }
    else {
      exit_code = G.is_break ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    /* Using window-manager API in background-mode is a bit odd, but works fine. */
    WM_exit(C, exit_code);
  }
  else {
    /* Not supported, although it could be made to work if needed. */
    BLI_assert(app_state.main_arg_deferred == nullptr);

    /* Shows the splash as needed. */
    WM_init_splash_on_startup(C);

    WM_main(C);
  }
  /* Neither #WM_exit, #WM_main return, this quiets CLANG's `unreachable-code-return` warning. */
  BLI_assert_unreachable();

#endif /* !WITH_PYTHON_MODULE */

  return 0;

} /* End of `int main(...)` function. */

/** \} */

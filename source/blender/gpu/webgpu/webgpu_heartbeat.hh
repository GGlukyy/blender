/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * D1: render-thread heartbeat in shared wasm memory. The render pthread stores the phase it is in
 * (shader module / pipeline / sync map / submit / present); the main thread reads it from JS
 * (`_wgpu_hb_ptr()`), so a frozen render thread can still be diagnosed from telemetry.
 */

#pragma once

#include <atomic>
#include <cstdint>

namespace blender::gpu {

enum WebGPUPhase : uint32_t {
  WGPU_PH_DRAW = 1,
  WGPU_PH_SHADER_MODULE = 2,
  WGPU_PH_PIPELINE = 3,
  WGPU_PH_MAP_READ = 4,
  WGPU_PH_SUBMIT = 5,
  WGPU_PH_PRESENT = 6,
};

/* [0] seq (bumped on every phase change), [1] frame (presents), [2] phase, [3] id,
 * [4] malloc bytes in use (updated per present). */
extern std::atomic<uint32_t> g_wgpu_hb[8];

inline void wgpu_hb_phase(WebGPUPhase phase, uint32_t id = 0)
{
  g_wgpu_hb[2].store(uint32_t(phase), std::memory_order_relaxed);
  g_wgpu_hb[3].store(id, std::memory_order_relaxed);
  g_wgpu_hb[0].fetch_add(1, std::memory_order_relaxed);
}

}  // namespace blender::gpu

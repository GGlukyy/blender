/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup draw_engine
 */

#pragma once

#include "BKE_object_types.hh"

#ifdef __EMSCRIPTEN__
#  include "BKE_object.hh"
#  include "BLI_math_geom_c.hh"
#endif

#include "DRW_gpu_wrapper.hh"

#include "GPU_select.hh"

#include "../intern/gpu_select_private.hh"

#include "draw_manager.hh"
#include "draw_pass.hh"

#include "gpu_shader_create_info.hh"

#include "select_defines.hh"
#include "select_shader_shared.hh"

namespace blender::draw::select {

// #define DEBUG_PRINT

enum class SelectionType { DISABLED = 0, ENABLED = 1 };

class ID {
 private:
  uint32_t value;

  /* Add type safety to selection ID. Only the select types should provide them. */
  ID(uint32_t value) : value(value) {};

  friend struct SelectBuf;
  friend struct SelectMap;

 public:
  uint32_t get() const
  {
    return value;
  }
};

/**
 * Add a dedicated selection id buffer to a pass.
 * To be used when not using a #PassMain which can pass the select ID via CustomID.
 */
struct SelectBuf {
  const SelectionType selection_type;

  StorageVectorBuffer<uint32_t> select_buf = {"select_buf"};

  SelectBuf(const SelectionType selection_type) : selection_type(selection_type) {};

  void select_clear()
  {
    if (selection_type != SelectionType::DISABLED) {
      select_buf.clear();
    }
  }

  void select_append(ID select_id)
  {
    if (selection_type != SelectionType::DISABLED) {
      select_buf.append(select_id.get());
    }
  }

  void select_bind(PassSimple::Sub &pass)
  {
    if (selection_type != SelectionType::DISABLED) {
      select_buf.push_update();
      pass.bind_ssbo(SELECT_ID_IN, &select_buf);
    }
  }
};

/**
 * Generate selection IDs from objects and keep record of the mapping between them.
 * The id's are contiguous so that we can create a destination buffer.
 */
struct SelectMap {
  const SelectionType selection_type;

  /** Mapping between internal IDs and `object->runtime->select_id`. */
  Vector<uint> select_id_map;
  /** Track objects with OB_DRAW_IN_FRONT. */
  Vector<bool> in_front_map;
#ifdef __EMSCRIPTEN__
  /** World-space AABB per internal id, for the CPU pick fallback (the WebGPU
   * backend cannot read the GPU result back synchronously — no JSPI). */
  Vector<std::pair<float3, float3>> bounds_map;
  /** World AABB of the bone being drawn; replaces the whole-object AABB in `select_id` so each
   * bone is its own pick target (otherwise every bone of an armature is the same box and a
   * click always picks the first one). Set by the overlay armature code. */
  std::optional<std::pair<float3, float3>> bone_bounds_override;
  /** World head/tail of the bone being drawn. Bones are picked by their projected line segment
   * (a box of a bone overlaps its neighbours in dense rigs). */
  std::optional<std::pair<float3, float3>> bone_segment_override;
  /** Per internal id: pick segment (head, tail) in world space, only valid when `has_seg_map`. */
  Vector<std::pair<float3, float3>> seg_map;
  Vector<bool> has_seg_map;
#endif
#ifndef NDEBUG
  /** Debug map containing a copy of the object name. */
  Vector<std::string> map_names;
#endif
  /** Stores the result of the whole selection drawing. Content depends on selection mode. */
  StorageArrayBuffer<uint> select_output_buf = {"select_output_buf"};
  /** Dummy buffer. Might be better to remove, but simplify the shader create info patching. */
  StorageArrayBuffer<uint, 4, true> dummy_select_buf = {"dummy_select_buf"};
  /** Uniform buffer to bind to all passes to pass information about the selection state. */
  UniformBuffer<SelectInfoData> info_buf;
  /** If clipping is enabled, this is the number of clip planes to enable. */
  int clipping_plane_count = 0;

  SelectMap(const SelectionType selection_type) : selection_type(selection_type) {};

  /* TODO(fclem): The sub_object_id id should eventually become some enum or take a sub-object
   * reference directly. This would isolate the selection logic to this class. */
  [[nodiscard]] const ID select_id(const ObjectRef &ob_ref, uint sub_object_id = 0)
  {
    if (selection_type == SelectionType::DISABLED) {
      return {0};
    }

    if (sub_object_id == uint(-1)) {
      /* WORKAROUND: Armature code set the sub_object_id to -1 when individual bones are not
       * selectable (i.e. in object mode). */
      sub_object_id = 0;
    }

#ifdef __EMSCRIPTEN__
    const uint raw_sub_object_id = sub_object_id;
#endif
    uint object_id = ob_ref.object->runtime->select_id;
    uint id = select_id_map.append_and_get_index(object_id | sub_object_id);
    in_front_map.append(ob_ref.object->dtx & OB_DRAW_IN_FRONT);

#ifdef __EMSCRIPTEN__
    {
      /* Record the world AABB for the CPU pick fallback (see read_result). */
      float3 bmin(-0.5f), bmax(0.5f);
      if (const std::optional<Bounds<float3>> b = BKE_object_boundbox_get(ob_ref.object)) {
        bmin = b->min;
        bmax = b->max;
      }
      else if (ob_ref.object->empty_drawsize > 0.0f) {
        bmin = float3(-ob_ref.object->empty_drawsize);
        bmax = float3(ob_ref.object->empty_drawsize);
      }
      const float4x4 &m = ob_ref.object->object_to_world();
      float3 wmin(FLT_MAX), wmax(-FLT_MAX);
      for (int c = 0; c < 8; c++) {
        float3 p((c & 1) ? bmax.x : bmin.x, (c & 2) ? bmax.y : bmin.y, (c & 4) ? bmax.z : bmin.z);
        p = math::transform_point(m, p);
        wmin = math::min(wmin, p);
        wmax = math::max(wmax, p);
      }
      if (bone_segment_override) {
        /* BONESEL_ROOT / BONESEL_TIP (edit-mode joints) = a point, otherwise the whole bone. */
        std::pair<float3, float3> seg = *bone_segment_override;
        if (raw_sub_object_id & (1u << 29)) {
          seg.second = seg.first;
        }
        else if (raw_sub_object_id & (1u << 30)) {
          seg.first = seg.second;
        }
        seg_map.append(seg);
        has_seg_map.append(true);
      }
      else {
        seg_map.append({float3(0), float3(0)});
        has_seg_map.append(false);
      }
      if (bone_bounds_override) {
        bounds_map.append(*bone_bounds_override);
      }
      else {
        bounds_map.append({wmin, wmax});
      }
    }
#endif

#ifdef DEBUG_PRINT
    /* Print mapping from object name, select id and the mapping to internal select id.
     * If something is wrong at this stage, it indicates an error in the caller code. */
    printf("%s : %u | %u = %u -> %u\n",
           ob_ref.object->id.name,
           object_id,
           sub_object_id,
           object_id | sub_object_id,
           id);
#endif

#ifndef NDEBUG
    map_names.append(ob_ref.object->id.name);
#endif
    return {id};
  }

  /* TODO: refactor this method to select::ID::invalid(). */
  /* Load an invalid index that will not write to the output (not selectable). */
  [[nodiscard]] static const ID select_invalid_id()
  {
    return {uint32_t(-1)};
  }

  void begin_sync(int clipping_plane_count)
  {
    if (selection_type == SelectionType::DISABLED) {
      return;
    }

    this->clipping_plane_count = clipping_plane_count;

    select_id_map.clear();
    in_front_map.clear();
#ifdef __EMSCRIPTEN__
    bounds_map.clear();
    seg_map.clear();
    has_seg_map.clear();
#endif
#ifndef NDEBUG
    map_names.clear();
#endif
  }

  /** IMPORTANT: Changes the draw state. Need to be called after the pass's own state_set. */
  void select_bind(PassSimple &pass)
  {
    if (selection_type == SelectionType::DISABLED) {
      return;
    }

    pass.state_set(DRW_STATE_WRITE_COLOR, clipping_plane_count);
    pass.bind_ubo(SELECT_DATA, &info_buf);
    pass.bind_ssbo(SELECT_ID_OUT, &select_output_buf);
  }

  /** IMPORTANT: Changes the draw state. Need to be called after the pass's own state_set. */
  void select_bind(PassMain &pass)
  {
    if (selection_type == SelectionType::DISABLED) {
      return;
    }

    pass.use_custom_ids = true;
    pass.state_set(DRW_STATE_WRITE_COLOR, clipping_plane_count);
    pass.bind_ubo(SELECT_DATA, &info_buf);
    /* IMPORTANT: This binds a dummy buffer `in_select_buf` but it is not supposed to be used. */
    pass.bind_ssbo(SELECT_ID_IN, &dummy_select_buf);
    pass.bind_ssbo(SELECT_ID_OUT, &select_output_buf);
  }

  /* TODO: Deduplicate. */
  /** IMPORTANT: Changes the draw state. Need to be called after the pass's own state_set. */
  void select_bind(PassMain &pass, PassMain::Sub &sub)
  {
    if (selection_type == SelectionType::DISABLED) {
      return;
    }

    pass.use_custom_ids = true;
    sub.state_set(DRW_STATE_WRITE_COLOR, clipping_plane_count);
    sub.bind_ubo(SELECT_DATA, &info_buf);
    /* IMPORTANT: This binds a dummy buffer `in_select_buf` but it is not supposed to be used. */
    sub.bind_ssbo(SELECT_ID_IN, &dummy_select_buf);
    sub.bind_ssbo(SELECT_ID_OUT, &select_output_buf);
  }

  void end_sync()
  {
    if (selection_type == SelectionType::DISABLED) {
      return;
    }

    BLI_assert(select_id_map.size() == in_front_map.size());

    select_output_buf.resize(max_uu(ceil_to_multiple_u(select_id_map.size(), 4), 4));
    select_output_buf.push_update();
  }

  void pre_draw()
  {
    if (selection_type == SelectionType::DISABLED) {
      return;
    }

    info_buf.radius = gpu_select_next_get_radius();
    switch (gpu_select_next_get_mode()) {
      /* Should not be used anymore for viewport selection. */
      case GPU_SELECT_NEAREST_FIRST_PASS:
      case GPU_SELECT_NEAREST_SECOND_PASS:
      case GPU_SELECT_INVALID:
        BLI_assert_unreachable();
        break;
      case GPU_SELECT_ALL:
        info_buf.mode = SelectType::SELECT_ALL;
        info_buf.cursor = gpu_select_next_get_pick_area_center();
        /* This mode uses atomicOr and store result as a bitmap. Clear to 0 (no selection). */
        GPU_storagebuf_clear(select_output_buf, 0);
        break;
      case GPU_SELECT_PICK_ALL:
        info_buf.mode = SelectType::SELECT_PICK_ALL;
        info_buf.cursor = gpu_select_next_get_pick_area_center();
        /* Mode uses atomicMin. Clear to UINT_MAX. */
        GPU_storagebuf_clear(select_output_buf, 0xFFFFFFFFu);
        break;
      case GPU_SELECT_PICK_NEAREST:
        info_buf.mode = SelectType::SELECT_PICK_NEAREST;
        info_buf.cursor = gpu_select_next_get_pick_area_center();
        /* Mode uses atomicMin. Clear to UINT_MAX. */
        GPU_storagebuf_clear(select_output_buf, 0xFFFFFFFFu);
        break;
    }

    info_buf.push_update();
  }

#ifdef __EMSCRIPTEN__
  void cpu_pick_fallback()
  {
    View &view = View::default_get();
    const float4x4 pv = view.winmat() * view.viewmat();
    const float4x4 pv_inv = math::invert(pv);

    /* Objects whose bones are individually selectable (edit/pose mode): their object-level AABB
     * entry must not take part, it is nearer than the bone hits and Blender then picks no bone. */
    constexpr uint kBoneSelAny = (1u << 29) | (1u << 30) | (1u << 31); /* BONESEL_ANY (ED_armature.hh) */
    Vector<uint> bone_pick_objects;
    for (const int i : bounds_map.index_range()) {
      if (has_seg_map[i] && (select_id_map[i] & kBoneSelAny)) {
        bone_pick_objects.append_non_duplicates(select_id_map[i] & 0xFFFFu);
      }
    }

    if (info_buf.mode == SelectType::SELECT_ALL) {
      /* Box/lasso "select everything in rect": the pick winmat maps the rect to
       * the full NDC cube, so an object is inside iff its projected AABB
       * overlaps [-1,1]². Conservative for corners behind the eye. */
      for (const int i : bounds_map.index_range()) {
        if (has_seg_map[i]) {
          /* Bone: its projected head-tail segment must touch the rect (Liang-Barsky on [-1,1]²). */
          const float4 ha = pv * float4(seg_map[i].first, 1.0f);
          const float4 hb = pv * float4(seg_map[i].second, 1.0f);
          bool hit = false;
          if (ha.w > 1e-8f && hb.w > 1e-8f) {
            const float2 a = float2(ha.x, ha.y) / ha.w;
            const float2 d = float2(hb.x, hb.y) / hb.w - a;
            float t0 = 0.0f, t1 = 1.0f;
            hit = true;
            for (int k = 0; k < 4 && hit; k++) {
              const float p = (k & 1) ? d[k >> 1] : -d[k >> 1];
              const float q = (k & 1) ? 1.0f - a[k >> 1] : a[k >> 1] + 1.0f;
              if (std::abs(p) < 1e-12f) {
                hit = q >= 0.0f;
              }
              else {
                const float r = q / p;
                if (p < 0.0f) {
                  t0 = math::max(t0, r);
                }
                else {
                  t1 = math::min(t1, r);
                }
                hit = t0 <= t1;
              }
            }
          }
          if (hit) {
            select_output_buf[i / 32] |= 1u << (i % 32);
          }
          continue;
        }
        const float3 &bmin = bounds_map[i].first;
        const float3 &bmax = bounds_map[i].second;
        float2 ndc_min(FLT_MAX), ndc_max(-FLT_MAX);
        bool any_behind = false;
        for (int c = 0; c < 8; c++) {
          float3 p((c & 1) ? bmax.x : bmin.x,
                   (c & 2) ? bmax.y : bmin.y,
                   (c & 4) ? bmax.z : bmin.z);
          const float4 hp = pv * float4(p, 1.0f);
          if (hp.w <= 1e-8f) {
            any_behind = true;
            continue;
          }
          const float2 ndc = float2(hp.x, hp.y) / hp.w;
          ndc_min = math::min(ndc_min, ndc);
          ndc_max = math::max(ndc_max, ndc);
        }
        const bool overlaps = any_behind || (ndc_min.x <= 1.0f && ndc_max.x >= -1.0f &&
                                             ndc_min.y <= 1.0f && ndc_max.y >= -1.0f);
        if (overlaps) {
          select_output_buf[i / 32] |= 1u << (i % 32);
        }
      }
      return;
    }

    /* Pick modes: ray through the rect center (the pick winmat centers the
     * cursor at NDC (0,0)). */
    const float3 p0 = math::project_point(pv_inv, float3(0.0f, 0.0f, -1.0f));
    const float3 p1 = math::project_point(pv_inv, float3(0.0f, 0.0f, 1.0f));
    const float3 dir = math::normalize(p1 - p0);
    for (const int i : bounds_map.index_range()) {
      if (!has_seg_map[i] && !(select_id_map[i] & kBoneSelAny) &&
          bone_pick_objects.contains(select_id_map[i] & 0xFFFFu))
      {
        continue;
      }
      if (has_seg_map[i]) {
        /* Bone: distance of the pick center (NDC 0,0; the pick rect spans [-1,1]²) to the
         * projected head-tail segment. Closest to the click wins. */
        const float4 ha = pv * float4(seg_map[i].first, 1.0f);
        const float4 hb = pv * float4(seg_map[i].second, 1.0f);
        if (ha.w <= 1e-8f || hb.w <= 1e-8f) {
          continue;
        }
        const float2 a = float2(ha.x, ha.y) / ha.w;
        const float2 b = float2(hb.x, hb.y) / hb.w;
        const float2 ab = b - a;
        const float len2 = math::dot(ab, ab);
        const float t = len2 > 1e-12f ? math::clamp(-math::dot(a, ab) / len2, 0.0f, 1.0f) : 0.0f;
        const float dist = math::length(a + ab * t);
        if (dist > 1.0f) {
          continue;
        }
        const float3 hit = math::interpolate(seg_map[i].first, seg_map[i].second, t);
        const float z01 = math::clamp(math::project_point(pv, hit).z * 0.5f + 0.5f, 0.0f, 1.0f);
        if (info_buf.mode == SelectType::SELECT_PICK_ALL) {
          uint32_t bits;
          const float zf = z01;
          memcpy(&bits, &zf, sizeof(bits));
          select_output_buf[i] = bits;
        }
        else {
          /* Distance to the click is the primary key for bones (depth is useless for a line
           * drawn on top of its neighbours). */
          select_output_buf[i] = (uint32_t(dist * float(0x00FFFFFFu)) << 8u) | 0u;
        }
        continue;
      }
      const float3 &bmin = bounds_map[i].first;
      const float3 &bmax = bounds_map[i].second;
      float tmin = 0.0f;
      if (!isect_ray_aabb_v3_simple(p0, dir, bmin, bmax, &tmin, nullptr)) {
        continue;
      }
      const float3 hit = p0 + dir * math::max(tmin, 0.0f);
      const float z01 = math::clamp(
          math::project_point(pv, hit).z * 0.5f + 0.5f, 0.0f, 1.0f);
      /* Match the shader encodings (select_lib.glsl). */
      if (info_buf.mode == SelectType::SELECT_PICK_ALL) {
        uint32_t bits;
        const float zf = z01;
        memcpy(&bits, &zf, sizeof(bits)); /* floatBitsToUint */
        select_output_buf[i] = bits;
      }
      else { /* SELECT_PICK_NEAREST: (depth24 << 8) | cursor_dist8. */
        select_output_buf[i] = (uint32_t(z01 * float(0x00FFFFFFu)) << 8u) | 0u;
      }
    }
  }
#endif

  void read_result()
  {
    if (selection_type == SelectionType::DISABLED) {
      return;
    }

    GPU_memory_barrier(GPU_BARRIER_BUFFER_UPDATE);
    /* This flush call should not be required. Still, on non-unified memory architecture
     * Apple devices this is needed for the result to be host visible.
     * This is likely to be a bug in the GPU backend.
     * So it should eventually be transformed into a backend
     * workaround instead of being fixed in user code. */
    select_output_buf.async_flush_to_host();
    select_output_buf.read();

#ifdef __EMSCRIPTEN__
    /* The WebGPU backend cannot read GPU buffers back synchronously (no JSPI):
     * select_output_buf still holds its clear pattern here. Synthesize the hits
     * on the CPU from the world AABBs recorded at sync time. Approximate (bbox,
     * not rasterized geometry), but it makes click/box select functional. */
    cpu_pick_fallback();
#endif

    Vector<GPUSelectResult> hit_results;

    /* Convert raw data from GPU to #GPUSelectResult. */
    switch (info_buf.mode) {
      case SelectType::SELECT_ALL:
        for (auto i : IndexRange(select_id_map.size())) {
          if (((select_output_buf[i / 32] >> (i % 32)) & 1) != 0) {
            GPUSelectResult hit_result{};
            hit_result.id = select_id_map[i];
            hit_result.depth = 0xFFFF;
            hit_results.append(hit_result);
          }
        }
        break;

      case SelectType::SELECT_PICK_ALL:
        for (auto i : IndexRange(select_id_map.size())) {
          if (select_output_buf[i] != 0xFFFFFFFFu) {
            GPUSelectResult hit_result{};
            hit_result.id = select_id_map[i];
            hit_result.depth = select_output_buf[i];
            if (in_front_map[i]) {
              /* Divide "In Front" objects depth so they go first. */
              /* TODO(Miguel Pozo): This reproduces the previous engine behavior, but it breaks
               * with code using depth for position reconstruction. Should we improve this? */
              float offset_depth = *reinterpret_cast<float *>(&hit_result.depth) / 100.0f;
              hit_result.depth = *reinterpret_cast<uint32_t *>(&offset_depth);
            }
            hit_results.append(hit_result);
          }
        }
        break;

      case SelectType::SELECT_PICK_NEAREST:
        for (auto i : IndexRange(select_id_map.size())) {
          if (select_output_buf[i] != 0xFFFFFFFFu) {
            /* NOTE: For `SELECT_PICK_NEAREST`, `select_output_buf` also contains the screen
             * distance to cursor in the lowest bits. */
            GPUSelectResult hit_result{};
            hit_result.id = select_id_map[i];
            hit_result.depth = select_output_buf[i];
            if (in_front_map[i]) {
              /* Divide "In Front" objects depth so they go first. */
              const uint32_t depth_mask = 0x00FFFFFFu << 8u;
              uint32_t offset_depth = ((hit_result.depth & depth_mask) >> 8u) / 100;
              hit_result.depth &= ~depth_mask;
              hit_result.depth |= offset_depth << 8u;
            }
            if (hit_results.is_empty() || hit_result.depth < hit_results[0].depth) {
              hit_results = {hit_result};
            }
          }
        }
        break;
    }
#ifdef DEBUG_PRINT
    for (auto &hit : hit_results) {
      /* Print hit results right out of the GPU selection buffer.
       * If something is wrong at this stage, it indicates an error in the selection shaders. */
      printf(" hit: %u: depth %u\n", hit.id, hit.depth);
    }
#endif

    gpu_select_next_set_result(hit_results.data(), hit_results.size());
  }
};

}  // namespace blender::draw::select

/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup edmesh
 *
 * WEB (WebGPU) build only: CPU occlusion test for Solid-mode edit-mesh picking/box/lasso select
 * (the GPU select-ID buffer cannot be read back synchronously there). Header-only.
 */

#pragma once

#ifdef __EMSCRIPTEN__

#  include "BLI_math_matrix.hh"
#  include "BLI_math_vector.hh"
#  include "BLI_vector.hh"

#  include "BKE_editmesh.hh"
#  include "BKE_editmesh_bvh.hh"

#  include "DNA_object_types.h"

#  include "ED_view3d.hh"

namespace blender {

/* -------------------------------------------------------------------- */
/** \name Web: CPU occlusion for Solid-mode picking
 * \{ */

/** Faces adjacent to the tested element never occlude it. */
struct WebOccExclude {
  const BMVert *vert;
  const BMEdge *edge;
  const BMFace *face;
};

inline bool web_occ_face_filter(BMFace *f, void *user_data)
{
  const WebOccExclude *ex = static_cast<const WebOccExclude *>(user_data);
  if (ex->vert && BM_vert_in_face(const_cast<BMVert *>(ex->vert), f)) {
    return false;
  }
  if (ex->edge && BM_edge_in_face(const_cast<BMEdge *>(ex->edge), f)) {
    return false;
  }
  if (ex->face == f) {
    return false;
  }
  return true;
}

/** One BVH per edit-mode base, built per pick (cheap for modeling-sized meshes). */
struct WebOccl {
  struct Tree {
    BMBVHTree *bvh;
    float4x4 obmat;
    float4x4 obinv;
  };
  const ViewContext *vc;
  Vector<Tree> trees;

  WebOccl(const ViewContext *vc_, const Span<Object *> objects) : vc(vc_)
  {
    for (Object *ob : objects) {
      BMEditMesh *em = BKE_editmesh_from_object(ob);
      Tree t;
      t.bvh = em ? BKE_bmbvh_new_from_editmesh(em, BKE_editmesh_bmesh_get_for_write(ob), BMBVH_RESPECT_HIDDEN, nullptr, false) : nullptr;
      t.obmat = ob->object_to_world();
      t.obinv = math::invert(t.obmat);
      trees.append(t);
    }
  }
  ~WebOccl()
  {
    for (Tree &t : trees) {
      if (t.bvh) {
        BKE_bmbvh_free(t.bvh);
      }
    }
  }

  /** Is the object-space point \a co_obj of tree \a tree_i, drawn at \a screen_co, visible? */
  bool point_visible(const int tree_i,
                     const float3 &co_obj,
                     const float2 &screen_co,
                     const BMVert *vert,
                     const BMEdge *edge,
                     const BMFace *face = nullptr) const
  {
    float start[3], end[3];
    if (!ED_view3d_win_to_segment_clipped(
            vc->depsgraph, vc->region, vc->v3d, screen_co, start, end, true))
    {
      return true;
    }
    const float3 co_world = math::transform_point(trees[tree_i].obmat, co_obj);
    const float3 to_eye = float3(start) - co_world;
    const float len = math::length(to_eye);
    if (len < 1e-9f) {
      return true;
    }
    /* Start a little off the surface so the element's own neighbourhood is not re-hit. */
    const float3 origin_world = co_world + to_eye * 1e-4f;
    const float3 eye_world = float3(start);

    for (const int i : trees.index_range()) {
      if (!trees[i].bvh) {
        continue;
      }
      const float3 o = math::transform_point(trees[i].obinv, origin_world);
      const float3 e = math::transform_point(trees[i].obinv, eye_world);
      float3 dir = e - o;
      float dist = math::length(dir);
      if (dist < 1e-9f) {
        continue;
      }
      dir /= dist;
      WebOccExclude ex = {i == tree_i ? vert : nullptr,
                          i == tree_i ? edge : nullptr,
                          i == tree_i ? face : nullptr};
      if (BKE_bmbvh_ray_cast_filter(
              trees[i].bvh, o, dir, 0.0f, &dist, nullptr, nullptr, web_occ_face_filter, &ex))
      {
        return false;
      }
    }
    return true;
  }

  /** The frontmost face under the cursor (all trees); sets \a r_tree. */
  BMFace *face_under_cursor(const float2 &mval, int &r_tree) const
  {
    float start[3], end[3];
    if (!ED_view3d_win_to_segment_clipped(
            vc->depsgraph, vc->region, vc->v3d, mval, start, end, true))
    {
      return nullptr;
    }
    BMFace *best = nullptr;
    float best_dist = FLT_MAX;
    for (const int i : trees.index_range()) {
      if (!trees[i].bvh) {
        continue;
      }
      const float3 o = math::transform_point(trees[i].obinv, float3(start));
      const float3 e = math::transform_point(trees[i].obinv, float3(end));
      float3 dir = e - o;
      const float len = math::length(dir);
      if (len < 1e-9f) {
        continue;
      }
      dir /= len;
      float dist = FLT_MAX;
      float3 hit;
      BMFace *f = BKE_bmbvh_ray_cast(trees[i].bvh, o, dir, 0.0f, &dist, hit, nullptr);
      if (f) {
        /* Compare in world space (object scale may differ between bases). */
        const float d = math::distance(math::transform_point(trees[i].obmat, hit),
                                       float3(start));
        if (d < best_dist) {
          best_dist = d;
          best = f;
          r_tree = i;
        }
      }
    }
    return best;
  }
};

/** \} */


}  // namespace blender

#endif /* __EMSCRIPTEN__ */

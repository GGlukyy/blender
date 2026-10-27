/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/* Based on Frostbite Unified Volumetric.
 * https://www.ea.com/frostbite/news/physically-based-unified-volumetric-rendering-in-frostbite */

/* Store volumetric properties into the froxel textures. */

#pragma once

#include "eevee_volume_lib.bsl.hh"

/* Needed includes for shader nodes. */
#include "draw_volume.bsl.hh"
#include "eevee_attributes_volume_lib.bsl.hh" /* IWYU pragma: export */
#include "eevee_nodetree_frag_lib.bsl.hh"
#include "eevee_occupancy_lib.bsl.hh"
#include "eevee_sampling_lib.bsl.hh"
#include "eevee_surf_common.bsl.hh"

namespace eevee {

struct VolumeProperties {
  float3 scattering;
  float3 absorption;
  float3 emission;
  float anisotropy;
};

struct SurfVolume {
  [[compilation_constant]] bool is_homogenous;
  [[compilation_constant]] bool is_world;

  [[resource_table]] draw::Volume volume;

  /* SSBO instead of R32UI image — see occupancy_buf_index (WGSL image atomics). */
  [[storage(OCCUPANCY_BUF_SLOT, read)]] const uint (&occupancy_buf)[];
  /* Accumulation mirror of the prop images: WebGPU forbids read_write storage
   * on RG11B10/R16F, so reads come from this buffer and the images stay
   * write-only (the last store per froxel holds the accumulated value).
   * Layout: VOLUME_PROP_BUF_STRIDE floats per froxel —
   * scatter.xyz, extinction.xyz, emission.xyz, phase, phase_weight, pad. */
  [[storage(VOLUME_PROP_BUF_SLOT, read_write)]] float (&volume_prop_buf)[];

  /* WRITE-only: reads go through volume_prop_buf (WebGPU forbids read_write
   * storage on RG11B10/R16F formats). */
  [[image(VOLUME_PROP_SCATTERING_IMG_SLOT, write, UFLOAT_11_11_10)]] image3D out_scattering_img;
  [[image(VOLUME_PROP_EXTINCTION_IMG_SLOT, write, UFLOAT_11_11_10)]] image3D out_extinction_img;
  [[image(VOLUME_PROP_EMISSION_IMG_SLOT, write, UFLOAT_11_11_10)]] image3D out_emissive_img;
  [[image(VOLUME_PROP_PHASE_IMG_SLOT, write, SFLOAT_16)]] image3D out_phase_img;
  [[image(VOLUME_PROP_PHASE_WEIGHT_IMG_SLOT, write, SFLOAT_16)]] image3D out_phase_weight_img;

  void write_froxel(int3 froxel, VolumeProperties prop)
  {
    float2 phase = float2(prop.anisotropy, 1.0f);

    /* Do not add phase weight if there's no scattering. */
    if (all(equal(prop.scattering, float3(0.0f)))) {
      phase = float2(0.0f);
    }

    float3 extinction = prop.scattering + prop.absorption;

    const int2 vts = imageSize(out_scattering_img).xy;
    const int accum = ((froxel.z * vts.y + froxel.y) * vts.x + froxel.x) *
                      VOLUME_PROP_BUF_STRIDE;
    if (!is_world) [[static_branch]] {
      /* Additive Blending. No race condition since we have a barrier between each conflicting
       * invocations. Reads go through the accumulation buffer: the images stay
       * write-only (WebGPU forbids read_write storage on these formats). */
      prop.scattering += float3(
          volume_prop_buf[accum + 0], volume_prop_buf[accum + 1], volume_prop_buf[accum + 2]);
      extinction += float3(
          volume_prop_buf[accum + 3], volume_prop_buf[accum + 4], volume_prop_buf[accum + 5]);
      prop.emission += float3(
          volume_prop_buf[accum + 6], volume_prop_buf[accum + 7], volume_prop_buf[accum + 8]);
      phase.x += volume_prop_buf[accum + 9];
      phase.y += volume_prop_buf[accum + 10];
    }

    volume_prop_buf[accum + 0] = prop.scattering.x;
    volume_prop_buf[accum + 1] = prop.scattering.y;
    volume_prop_buf[accum + 2] = prop.scattering.z;
    volume_prop_buf[accum + 3] = extinction.x;
    volume_prop_buf[accum + 4] = extinction.y;
    volume_prop_buf[accum + 5] = extinction.z;
    volume_prop_buf[accum + 6] = prop.emission.x;
    volume_prop_buf[accum + 7] = prop.emission.y;
    volume_prop_buf[accum + 8] = prop.emission.z;
    volume_prop_buf[accum + 9] = phase.x;
    volume_prop_buf[accum + 10] = phase.y;

    imageStoreFast(out_scattering_img, froxel, prop.scattering.xyzz);
    imageStoreFast(out_extinction_img, froxel, extinction.xyzz);
    imageStoreFast(out_emissive_img, froxel, prop.emission.xyzz);
    imageStoreFast(out_phase_img, froxel, phase.xxxx);
    imageStoreFast(out_phase_weight_img, froxel, phase.yyyy);
  }

  VolumeProperties eval_froxel(KernelGlobals &kg,
                               ShadingData sd,
                               const Uniform &uni,
                               const ViewMatrices view,
                               const ObjectMatrices obj,
                               const ObjectInfos ob_infos,
                               int3 froxel,
                               float jitter)
  {
    float3 uvw = (float3(froxel) + float3(0.5f, 0.5f, 0.5f - jitter)) *
                 uni.uniform_buf.volumes.inv_tex_size;

    float3 vP = volume_jitter_to_view(uni, view, uvw);
    float3 wP = view.point_view_to_world(vP);
    float3 lP = obj.point_world_to_object(wP);
    /* Compute Original Coordinate (ORCO). */
    float3 lP_orco = lP * ob_infos.orco_mul + ob_infos.orco_add;

    sd.P = wP;
    sd.N = float3(0.0f);
    sd.Ng = float3(0.0f);
    sd.ray_length = distance(sd.P, view.position());

    VolumePoint volume_pt;
    volume_pt.lP = lP;
    volume_pt.orco_default = lP_orco;
    for (int i = 0; i < 16 /* DRW_GRID_PER_VOLUME_MAX */; i++) [[unroll]] {
      volume_pt.grid_co[i] = transform_point(volume.drw_volume.grids_xform[i], lP);
    }

    attrib_load(volume_pt);

    nodetree_volume(kg, sd);

    sd.volume_scattering *= volume.drw_volume.density_scale;
    sd.volume_absorption *= volume.drw_volume.density_scale;
    sd.emission *= volume.drw_volume.density_scale;

    VolumeProperties prop;
    prop.scattering = sd.volume_scattering;
    prop.absorption = sd.volume_absorption;
    prop.emission = sd.emission;
    prop.anisotropy = sd.volume_anisotropy;
    return prop;
  }
};

/* Note: Only the front fragments have to be invoked. */
[[fragment]] [[early_fragment_tests]] [[texture_atomic]]
void surf_volume([[resource_table]] KernelGlobals &kg,
                 [[resource_table]] PipelineConstants &pipe,
                 [[resource_table]] SurfVolume &srt,
                 [[resource_table]] const Uniform &uni,
                 [[resource_table]] const draw::Model &models,
                 [[resource_table]] const draw::View &views,
                 [[resource_table]] const draw::Infos &infos,
                 [[resource_table]] const Sampling &sampling,
                 [[resource_table]] const UtilityTexture & /*util_tx*/,
                 [[in]] const VertOutCommon &interp,
                 [[in]] [[condition(is_curves)]] const VertOutCurves &curves_interp,
                 [[in]] [[condition(is_pointcloud)]] const VertOutPointcloud &ptcloud_interp,
                 [[frag_coord]] const float4 frag_co,
                 [[front_facing]] const bool front_face)
{
  int3 froxel = int3(int2(frag_co.xy), 0);
  float offset = sampling.rng_1D_get(SAMPLING_VOLUME_W);
  float jitter = volume_froxel_jitter(froxel.xy, offset);

  draw::ID id{interp.resource_id_raw};
  const uint resource_id = id.resource_id<1>();
  const ObjectMatrices obj = models.get(resource_id);
  const ObjectInfos ob_infos = infos.get(resource_id);
  const ViewMatrices view = views.get(0);

  VolumeProperties prop;

  ShadingData sd = init_globals(pipe, uni, interp, view, front_face, frag_co);
  if (pipe.is_mesh) [[static_branch]] {
    init_globals_mesh(interp, sd, float3(0.0));
  }
  else if (pipe.is_curves) [[static_branch]] {
    init_globals_curves(interp, curves_interp, sd, view);
  }
  else if (pipe.is_pointcloud) [[static_branch]] {
    init_globals_pointcloud(ptcloud_interp, sd);
  }

  if (srt.is_homogenous) [[static_branch]] {
    /* Homogenous volumes only evaluate properties at volume entrance and write the same values for
     * each froxel. */
    prop = srt.eval_froxel(kg, sd, uni, view, obj, ob_infos, froxel, jitter);
  }

  occupancy::Bits occupancy;

  if (!srt.is_world) [[static_branch]] {
    /* The buffer only holds ceil(tex_size.z / 32) words per froxel column —
     * unlike the old image, out-of-range reads are NOT defined-zero. */
    const int layer_len = (int(uni.uniform_buf.volumes.tex_size.z) + 31) / 32;
    for (int j = 0; j < 8; j++) {
      occupancy.bits[j] = (j < layer_len) ?
                              srt.occupancy_buf[::occupancy::occupancy_buf_index(
                                  froxel.xy, j, int2(uni.uniform_buf.volumes.tex_size.xy))] :
                              0u;
    }
  }

  /* Check all occupancy bits. */
  for (int j = 0; j < 8; j++) {
    for (int i = 0; i < 32; i++) {
      froxel.z = j * 32 + i;

      if (froxel.z >= imageSize(srt.out_scattering_img).z) {
        break;
      }

      if (!srt.is_world) [[static_branch]] {
        if (((occupancy.bits[j] >> i) & 1u) == 0) {
          continue;
        }
      }

      if (!srt.is_homogenous) [[static_branch]] {
        /* Heterogeneous volumes evaluate properties at every froxel position. */
        prop = srt.eval_froxel(kg, sd, uni, view, obj, ob_infos, froxel, jitter);
      }
      srt.write_froxel(froxel, prop);
    }
  }
}

}  // namespace eevee

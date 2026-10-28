/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "eevee_uniform.bsl.hh"

namespace eevee {

struct RenderPassOutput {
  /* Web (Safari): rgba16f/r16f read_write needs texture-formats-tier2 → write-only (load_color unused, removed). */
  [[image(RBUFS_COLOR_SLOT, write, SFLOAT_16_16_16_16)]] image2DArray rp_color_img;
  [[image(RBUFS_VALUE_SLOT, write, SFLOAT_16)]] image2DArray rp_value_img;

  void store_color(int2 texel, int id, float4 color)
  {
    if (id >= 0) {
      imageStoreFast(rp_color_img, int3(texel, id), color);
    }
  }

  void store_value(int2 texel, int id, float value)
  {
    if (id >= 0) {
      imageStoreFast(rp_value_img, int3(texel, id), float4(value));
    }
  }

  void clear_aovs([[resource_table]] const Uniform &uni, int2 texel)
  {
    for (int i = 0; i < AOV_MAX && i < uni.uniform_buf.render_pass.aovs.color_len; i++) {
      store_color(texel, uni.uniform_buf.render_pass.color_len + i, float4(0));
    }
    for (int i = 0; i < AOV_MAX && i < uni.uniform_buf.render_pass.aovs.value_len; i++) {
      store_value(texel, uni.uniform_buf.render_pass.value_len + i, 0.0f);
    }
  }
};

}  // namespace eevee

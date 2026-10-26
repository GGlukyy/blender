/* SPDX-FileCopyrightText: 2001-2002 NaN Holding BV. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup edscr
 */

#include "DNA_userdef_types.h"
#include "DNA_vec_types.h"

#include "BLI_utildefines.hh"

#include "BIF_glutil.hh"

#include "IMB_colormanagement.hh"
#include "IMB_imbuf_types.hh"

#include "GPU_immediate.hh"
#include "GPU_texture.hh"

namespace blender {

void PixelBitmapDrawer::init_vertex_attributes()
{
  GPUVertFormat *vert_format = immVertexFormat();
  this->pos = GPU_vertformat_attr_add(vert_format, "pos", gpu::VertAttrType::SFLOAT_32_32);
  this->texco = GPU_vertformat_attr_add(vert_format, "texCoord", gpu::VertAttrType::SFLOAT_32_32);
}

PixelBitmapDrawer::PixelBitmapDrawer() : shader(nullptr)
{
  init_vertex_attributes();
}

PixelBitmapDrawer::PixelBitmapDrawer(GPUBuiltinShader builtin_shader)
{
  init_vertex_attributes();

  this->shader = GPU_shader_get_builtin_shader(builtin_shader);
  /* Shader will be unbound in draw(). */
  immBindBuiltinProgram(builtin_shader);
}

void PixelBitmapDrawer::draw(const float x,
                             const float y,
                             const int img_w,
                             const int img_h,
                             const gpu::TextureFormat gpu_format,
                             const bool use_filter,
                             const void *rect,
                             const float scale_x,
                             const float scale_y,
                             const float color[4])
{
  const float draw_width = img_w * scale_x;
  const float draw_height = img_h * scale_y;

  /* When scaling down by more than 2x, create mipmaps for the texture and
   * use trilinear filtering. */
  const bool use_mipmap = use_filter && (scale_x < 0.5f || scale_y < 0.5f);
  const int mip_len = use_mipmap ? 9999 : 1;

  eGPUTextureUsage usage = GPU_TEXTURE_USAGE_SHADER_READ;
  if (use_mipmap) {
    usage |= GPU_TEXTURE_USAGE_SHADER_WRITE;
  }
  /* Small cache of textures (also mipmapped ones: Sculpt redraws a scaled-down brush image every frame): a redraw loop (e.g. dragging) would otherwise create and
   * free a texture per call. WebKit keeps freed GPU memory alive until the bind groups holding it are
   * garbage collected, which kills iPad Safari (docs/IPAD_CRASH.md H3). */
  struct CachedTex {
    gpu::Texture *tex = nullptr;
    int w = 0, h = 0;
    gpu::TextureFormat format = gpu::TextureFormat::Invalid;
    bool mip = false;
    uint64_t used = 0;
  };
  static CachedTex cache[4];
  static uint64_t cache_clock = 0;
  gpu::Texture *tex = nullptr;
  CachedTex *cached = nullptr;
  {
    for (CachedTex &c : cache) {
      if (c.tex && c.w == img_w && c.h == img_h && c.format == gpu_format && c.mip == use_mipmap) {
        cached = &c;
        break;
      }
    }
    if (cached == nullptr) {
      cached = &cache[0];
      for (CachedTex &c : cache) {
        if (c.tex == nullptr) {
          cached = &c;
          break;
        }
        if (c.used < cached->used) {
          cached = &c;
        }
      }
      if (cached->tex) {
        GPU_texture_free(cached->tex);
        cached->tex = nullptr;
      }
      cached->tex = GPU_texture_create_2d(
          "immDrawPixels", img_w, img_h, mip_len, gpu_format, usage, nullptr);
      cached->w = img_w;
      cached->h = img_h;
      cached->format = gpu_format;
      cached->mip = use_mipmap;
    }
    cached->used = ++cache_clock;
    tex = cached->tex;
  }
  if (tex == nullptr) {
    return;
  }

  const bool use_float_data = ELEM(gpu_format,
                                   gpu::TextureFormat::SFLOAT_16_16_16_16,
                                   gpu::TextureFormat::SFLOAT_16_16_16,
                                   gpu::TextureFormat::SFLOAT_16);
  eGPUDataFormat gpu_data_format = use_float_data ? GPU_DATA_FLOAT : GPU_DATA_UBYTE;
  GPU_texture_update(tex, gpu_data_format, rect);

  GPU_texture_filter_mode(tex, use_filter);
  if (use_mipmap) {
    GPU_texture_update_mipmap_chain(tex);
    GPU_texture_mipmap_mode(tex, true, true);
  }
  GPU_texture_extend_mode(tex, GPU_SAMPLER_EXTEND_MODE_EXTEND);

  GPU_texture_bind(tex, 0);

  /* NOTE: Shader could be null for GLSL OCIO drawing, it is fine, since
   * it does not need color. */
  static const float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  if (this->shader != nullptr && GPU_shader_get_uniform(this->shader, "color") != -1) {
    immUniformColor4fv((color) ? color : white);
  }

  const uint pos = this->pos, texco = this->texco;

  immBegin(GPU_PRIM_TRI_FAN, 4);
  immAttr2f(texco, 0.0f, 0.0f);
  immVertex2f(pos, x, y);

  immAttr2f(texco, 1.0f, 0.0f);
  immVertex2f(pos, x + draw_width, y);

  immAttr2f(texco, 1.0f, 1.0f);
  immVertex2f(pos, x + draw_width, y + draw_height);

  immAttr2f(texco, 0.0f, 1.0f);
  immVertex2f(pos, x, y + draw_height);
  immEnd();

  if (this->shader) {
    immUnbindProgram();
  }

  GPU_texture_unbind(tex);
  if (cached == nullptr) {
    GPU_texture_free(tex);
  }
}

void ED_draw_imbuf(const ImBuf *ibuf,
                   float x,
                   float y,
                   bool use_filter,
                   const ColorManagedViewSettings *view_settings,
                   const ColorManagedDisplaySettings *display_settings,
                   float zoom_x,
                   float zoom_y)
{
  using namespace blender::gpu;

  /* Early out */
  if (ibuf->byte_data() == nullptr && ibuf->float_data() == nullptr) {
    return;
  }

  PixelBitmapDrawer drawer;

  const ColorSpace *colorspace = ibuf->float_data() ? ibuf->float_buffer.colorspace :
                                                      ibuf->byte_buffer.colorspace;
  const bool predivide = ibuf->float_data() != nullptr;
  if (!IMB_colormanagement_setup_glsl_draw_from_space(
          view_settings, display_settings, colorspace, ibuf->dither, predivide, false))
  {
    return;
  }

  const void *texture_data = nullptr;
  TextureFormat format = TextureFormat::Invalid;
  if (ibuf->float_data()) {
    texture_data = ibuf->float_data();
    if (ibuf->channels == 1) {
      format = TextureFormat::SFLOAT_16;
    }
    else if (ibuf->channels == 3) {
      format = TextureFormat::SFLOAT_16_16_16;
    }
    else if (ibuf->channels == 4) {
      format = TextureFormat::SFLOAT_16_16_16_16;
    }
    else {
      BLI_assert_msg(0, "Incompatible number of channels for GLSL display");
    }
  }
  else {
    texture_data = ibuf->byte_data();
    format = TextureFormat::UNORM_8_8_8_8;
  }

  if (format != TextureFormat::Invalid) {
    drawer.draw(x, y, ibuf->x, ibuf->y, format, use_filter, texture_data, zoom_x, zoom_y, nullptr);
  }

  IMB_colormanagement_finish_glsl_draw();
}

void ED_draw_imbuf_ctx(const bContext *C,
                       const ImBuf *ibuf,
                       float x,
                       float y,
                       bool use_filter,
                       float zoom_x,
                       float zoom_y)
{
  ColorManagedViewSettings *view_settings;
  ColorManagedDisplaySettings *display_settings;
  IMB_colormanagement_display_settings_from_ctx(C, &view_settings, &display_settings);
  ED_draw_imbuf(ibuf, x, y, use_filter, view_settings, display_settings, zoom_x, zoom_y);
}

void immDrawBorderCorners(uint pos, const rcti *border, float zoomx, float zoomy)
{
  float delta_x = 4.0f * UI_SCALE_FAC / zoomx;
  float delta_y = 4.0f * UI_SCALE_FAC / zoomy;

  delta_x = min_ff(delta_x, border->xmax - border->xmin);
  delta_y = min_ff(delta_y, border->ymax - border->ymin);

  /* left bottom corner */
  immBegin(GPU_PRIM_LINE_STRIP, 3);
  immVertex2f(pos, border->xmin, border->ymin + delta_y);
  immVertex2f(pos, border->xmin, border->ymin);
  immVertex2f(pos, border->xmin + delta_x, border->ymin);
  immEnd();

  /* left top corner */
  immBegin(GPU_PRIM_LINE_STRIP, 3);
  immVertex2f(pos, border->xmin, border->ymax - delta_y);
  immVertex2f(pos, border->xmin, border->ymax);
  immVertex2f(pos, border->xmin + delta_x, border->ymax);
  immEnd();

  /* right bottom corner */
  immBegin(GPU_PRIM_LINE_STRIP, 3);
  immVertex2f(pos, border->xmax - delta_x, border->ymin);
  immVertex2f(pos, border->xmax, border->ymin);
  immVertex2f(pos, border->xmax, border->ymin + delta_y);
  immEnd();

  /* right top corner */
  immBegin(GPU_PRIM_LINE_STRIP, 3);
  immVertex2f(pos, border->xmax - delta_x, border->ymax);
  immVertex2f(pos, border->xmax, border->ymax);
  immVertex2f(pos, border->xmax, border->ymax - delta_y);
  immEnd();
}

}  // namespace blender

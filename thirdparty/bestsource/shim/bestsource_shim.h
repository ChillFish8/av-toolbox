#ifndef AV_TOOLBOX_BESTSOURCE_SHIM_H
#define AV_TOOLBOX_BESTSOURCE_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bs_video bs_video;
typedef struct bs_frame bs_frame;

typedef struct bs_format {
    int color_family;
    int alpha;
    int is_float;
    int bits;
    int sub_sampling_w;
    int sub_sampling_h;
} bs_format;

typedef struct bs_video_props {
    int64_t num_frames;
    int fps_num;
    int fps_den;
    int time_base_num;
    int time_base_den;
    int sar_num;
    int sar_den;
    int width;
    int height;
    int ss_mod_width;
    int ss_mod_height;
    bs_format format;
} bs_video_props;

typedef struct bs_frame_props {
    int64_t pts;
    int64_t duration;
    int width;
    int height;
    int ss_mod_width;
    int ss_mod_height;
    bs_format format;
    int matrix;
    int primaries;
    int transfer;
    int chroma_location;
    int color_range;
    int key_frame;
    int interlaced;
    int top_field_first;
} bs_frame_props;

/* Every function returning int gives 0 on success, or -1 with a NUL terminated message in error. */

int bs_video_open(const char *path, int track, int threads, int cache_mode, const char *cache_path,
                  bs_video **out_video, char *error, size_t error_len);
void bs_video_free(bs_video *video);
void bs_video_properties(const bs_video *video, bs_video_props *out_props);
int bs_video_frame(bs_video *video, int64_t n, bs_frame **out_frame, char *error, size_t error_len);

void bs_frame_info(const bs_frame *frame, bs_frame_props *out_props);
int bs_frame_export_planar(const bs_frame *frame, uint8_t *const planes[3], const ptrdiff_t strides[3],
                           uint8_t *alpha, ptrdiff_t alpha_stride, char *error, size_t error_len);
void bs_frame_free(bs_frame *frame);

#ifdef __cplusplus
}
#endif

#endif

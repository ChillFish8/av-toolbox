#include "bestsource_shim.h"

#include <cstring>
#include <exception>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

#include "videosource.h"

namespace {

void write_error(char *error, size_t error_len, const char *message) {
    if (error == nullptr || error_len == 0) {
        return;
    }

    size_t length = std::strlen(message);
    if (length >= error_len) {
        length = error_len - 1;
    }

    std::memcpy(error, message, length);
    error[length] = '\0';
}

/* Runs body, turning any exception into -1 and a message, since none may cross into Rust. */
template <typename Body>
int guarded(char *error, size_t error_len, Body body) {
    try {
        body();
        return 0;
    } catch (const std::exception &exception) {
        write_error(error, error_len, exception.what());
    } catch (...) {
        write_error(error, error_len, "unknown C++ exception");
    }

    return -1;
}

void fill_format(const BSVideoFormat &format, bs_format *out_format) {
    out_format->color_family = format.ColorFamily;
    out_format->alpha = format.Alpha ? 1 : 0;
    out_format->is_float = format.Float ? 1 : 0;
    out_format->bits = format.Bits;
    out_format->sub_sampling_w = format.SubSamplingW;
    out_format->sub_sampling_h = format.SubSamplingH;
}

BestVideoSource *as_source(bs_video *video) {
    return reinterpret_cast<BestVideoSource *>(video);
}

const BestVideoSource *as_source(const bs_video *video) {
    return reinterpret_cast<const BestVideoSource *>(video);
}

const BestVideoFrame *as_frame(const bs_frame *frame) {
    return reinterpret_cast<const BestVideoFrame *>(frame);
}

}  // namespace

extern "C" int bs_video_open(const char *path, int track, int threads, int cache_mode, const char *cache_path,
                             bs_video **out_video, char *error, size_t error_len) {
    return guarded(error, error_len, [&] {
        std::filesystem::path source_path = std::filesystem::u8path(path);
        std::filesystem::path cache_directory;
        if (cache_path != nullptr) {
            cache_directory = std::filesystem::u8path(cache_path);
        }

        std::map<std::string, std::string> no_options;
        auto *source = new BestVideoSource(source_path, false, std::nullopt, track, 0, threads, cache_mode,
                                           cache_directory, &no_options);
        *out_video = reinterpret_cast<bs_video *>(source);
    });
}

extern "C" void bs_video_free(bs_video *video) {
    delete as_source(video);
}

extern "C" void bs_video_properties(const bs_video *video, bs_video_props *out_props) {
    const BSVideoProperties &properties = as_source(video)->GetVideoProperties();

    out_props->num_frames = properties.NumFrames;
    out_props->fps_num = properties.FPS.Num;
    out_props->fps_den = properties.FPS.Den;
    out_props->time_base_num = properties.TimeBase.Num;
    out_props->time_base_den = properties.TimeBase.Den;
    out_props->sar_num = properties.SAR.Num;
    out_props->sar_den = properties.SAR.Den;
    out_props->width = properties.Width;
    out_props->height = properties.Height;
    out_props->ss_mod_width = properties.SSModWidth;
    out_props->ss_mod_height = properties.SSModHeight;
    fill_format(properties.VF, &out_props->format);
}

extern "C" int bs_video_frame(bs_video *video, int64_t n, bs_frame **out_frame, char *error, size_t error_len) {
    return guarded(error, error_len, [&] {
        BestVideoSource *source = as_source(video);
        BestVideoFrame *frame = source->GetFrame(n);
        if (frame == nullptr) {
            std::string num_frames = std::to_string(source->GetVideoProperties().NumFrames);
            throw std::out_of_range("frame " + std::to_string(n) + " is not one of the source's " + num_frames +
                                    " frames");
        }

        *out_frame = reinterpret_cast<bs_frame *>(frame);
    });
}

extern "C" void bs_frame_info(const bs_frame *frame, bs_frame_props *out_props) {
    const BestVideoFrame *video_frame = as_frame(frame);

    out_props->pts = video_frame->PTS;
    out_props->duration = video_frame->Duration;
    out_props->width = video_frame->Width;
    out_props->height = video_frame->Height;
    out_props->ss_mod_width = video_frame->SSModWidth;
    out_props->ss_mod_height = video_frame->SSModHeight;
    fill_format(video_frame->VF, &out_props->format);
    out_props->matrix = video_frame->Matrix;
    out_props->primaries = video_frame->Primaries;
    out_props->transfer = video_frame->Transfer;
    out_props->chroma_location = video_frame->ChromaLocation;
    out_props->color_range = video_frame->ColorRange;
    out_props->key_frame = video_frame->KeyFrame ? 1 : 0;
    out_props->interlaced = video_frame->InterlacedFrame ? 1 : 0;
    out_props->top_field_first = video_frame->TopFieldFirst ? 1 : 0;
}

extern "C" int bs_frame_export_planar(const bs_frame *frame, uint8_t *const planes[3], const ptrdiff_t strides[3],
                                      uint8_t *alpha, ptrdiff_t alpha_stride, char *error, size_t error_len) {
    return guarded(error, error_len, [&] {
        bool exported = as_frame(frame)->ExportAsPlanar(planes, strides, alpha, alpha_stride);
        if (!exported) {
            throw std::runtime_error("the frame's format cannot be exported as planar");
        }
    });
}

extern "C" void bs_frame_free(bs_frame *frame) {
    delete reinterpret_cast<BestVideoFrame *>(frame);
}

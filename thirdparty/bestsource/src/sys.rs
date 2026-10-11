use std::ffi::{c_char, c_int};

#[repr(C)]
pub(crate) struct BsVideo {
    _private: [u8; 0],
}

#[repr(C)]
pub(crate) struct BsFrame {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct BsFormat {
    pub color_family: c_int,
    pub alpha: c_int,
    pub is_float: c_int,
    pub bits: c_int,
    pub sub_sampling_w: c_int,
    pub sub_sampling_h: c_int,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct BsVideoProps {
    pub num_frames: i64,
    pub fps_num: c_int,
    pub fps_den: c_int,
    pub time_base_num: c_int,
    pub time_base_den: c_int,
    pub sar_num: c_int,
    pub sar_den: c_int,
    pub width: c_int,
    pub height: c_int,
    pub ss_mod_width: c_int,
    pub ss_mod_height: c_int,
    pub format: BsFormat,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct BsFrameProps {
    pub pts: i64,
    pub duration: i64,
    pub width: c_int,
    pub height: c_int,
    pub ss_mod_width: c_int,
    pub ss_mod_height: c_int,
    pub format: BsFormat,
    pub matrix: c_int,
    pub primaries: c_int,
    pub transfer: c_int,
    pub chroma_location: c_int,
    pub color_range: c_int,
    pub key_frame: c_int,
    pub interlaced: c_int,
    pub top_field_first: c_int,
}

unsafe extern "C" {
    pub(crate) fn bs_video_open(
        path: *const c_char,
        track: c_int,
        threads: c_int,
        cache_mode: c_int,
        cache_path: *const c_char,
        out_video: *mut *mut BsVideo,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;
    pub(crate) fn bs_video_free(video: *mut BsVideo);
    pub(crate) fn bs_video_properties(video: *const BsVideo, out_props: *mut BsVideoProps);
    pub(crate) fn bs_video_frame(
        video: *mut BsVideo,
        n: i64,
        out_frame: *mut *mut BsFrame,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;

    pub(crate) fn bs_frame_info(frame: *const BsFrame, out_props: *mut BsFrameProps);
    pub(crate) fn bs_frame_export_planar(
        frame: *const BsFrame,
        planes: *const *mut u8,
        strides: *const isize,
        alpha: *mut u8,
        alpha_stride: isize,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;
    pub(crate) fn bs_frame_free(frame: *mut BsFrame);
}

use std::cell::RefCell;

use av_toolbox_cubek::DevicePlane;
use av_toolbox_gpu::{FrameContext, testing_context};
use cubecl::server::Handle;

use crate::aa::direction::FieldPass;
use crate::scratch::Scratch;

/// A field pass that only records the offset and output buffer of each run.
#[derive(Default)]
struct RecordingPass {
    runs: RefCell<Vec<(u32, *const Handle)>>,
}

impl FieldPass for RecordingPass {
    fn run(&self, _context: &FrameContext, _input: DevicePlane<'_>, output: DevicePlane<'_>, offset: u32) {
        let buffer: *const Handle = output.handle();

        let mut runs = self.runs.borrow_mut();
        runs.push((offset, buffer));
    }
}

#[test]
fn the_default_run_both_runs_each_parity_in_turn() {
    let context = testing_context();
    let input = Scratch::new::<f32>(&context, 8, 8);
    let even = Scratch::new::<f32>(&context, 8, 8);
    let odd = Scratch::new::<f32>(&context, 8, 8);
    let input_plane = input.plane();
    let even_plane = even.plane();
    let odd_plane = odd.plane();

    let pass = RecordingPass::default();
    pass.run_both(&context, input_plane, even_plane, odd_plane);

    let even_handle: *const Handle = even_plane.handle();
    let odd_handle: *const Handle = odd_plane.handle();
    let runs = pass.runs.into_inner();
    assert_eq!(runs, [(0, even_handle), (1, odd_handle)]);
}

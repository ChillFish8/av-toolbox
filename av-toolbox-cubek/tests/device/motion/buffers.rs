use av_toolbox_cubek::motion::{BlockGrid, FieldBuffers, MotionSettings, SceneFlags};
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::support::backend;

/// What a field of `FieldBuffers` reports about its planes.
#[derive(Debug, PartialEq, Eq)]
struct FieldShape {
    vectors: (u32, u32),
    sads: (u32, u32),
    confidence: Option<(u32, u32)>,
    pel: u32,
}

/// Asks for a field of a 100x60 frame's grid, and for a flag.
struct BufferRun {
    grid: BlockGrid,
    pel: u32,
    scored: bool,
}

impl BuildTask for BufferRun {
    type Output = (FieldShape, u32, u32);

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let buffers = FieldBuffers::new(client, &self.grid);
        let field = buffers.field(self.pel, self.scored);
        let shape = FieldShape {
            vectors: (field.planes.vectors.width(), field.planes.vectors.height()),
            sads: (field.planes.sads.width(), field.planes.sads.height()),
            confidence: field
                .planes
                .confidence
                .map(|confidence| (confidence.width(), confidence.height())),
            pel: field.pel,
        };

        let flags = SceneFlags::new(client, 3);
        let flag = flags.flag(2);

        (shape, flags.plane().width(), flag.index)
    }
}

#[test]
fn field_buffers_give_a_field_on_their_grid() {
    let grid = BlockGrid::new(&MotionSettings::default(), 100, 60);

    for (pel, scored) in [(1, false), (2, true)] {
        let task = BufferRun { grid, pel, scored };
        let (shape, flag_count, flag_index) = backend().build(task);

        let expected = FieldShape {
            vectors: grid.vector_dims(),
            sads: grid.sad_dims(),
            confidence: scored.then(|| grid.sad_dims()),
            pel,
        };
        assert_eq!(shape, expected, "pel {pel} scored {scored}");
        assert_eq!((flag_count, flag_index), (3, 2));
    }
}





/// The registry tracks all stages available for use within the pipeline.
pub struct PipelineRegistry {
    sinks: Vec<()>,
    masks: Vec<()>,
    filters: Vec<()>,
    transforms: Vec<()>,
    sources: Vec<()>,
}

impl PipelineRegistry {

}


struct Entry {
    full_identifier: String,
    package: &'static str,
    stage_identifier: &'static str,
}


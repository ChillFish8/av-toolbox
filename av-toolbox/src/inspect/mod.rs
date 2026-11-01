mod export;
mod hash;

use std::path::Path;
use std::process::ExitCode;

use av_toolbox_core::PipelineRegistry;
use av_toolbox_graph::Diagnostic;
use av_toolbox_pipeline::{DiskFiles, Pipeline};

use self::export::{Export, FORMAT_VERSION};
use crate::cli::{Args, InspectArgs};
use crate::load;
use crate::report::{self, PipelineFiles};

pub(crate) fn run(args: &Args, inspect: &InspectArgs) -> ExitCode {
    let Some(registry) = load::registry() else {
        return ExitCode::FAILURE;
    };

    let inspected = inspect_pipeline(&registry, &inspect.pipeline, &inspect.vars);

    let mut stderr = report::stderr(args.color);
    report::diagnostics(&mut stderr, &inspected.files, &inspected.diagnostics);

    if inspect.hash {
        if let Some(hash) = &inspected.export.hash {
            println!("{hash}");
        }
    } else {
        let written = serde_json::to_string_pretty(&inspected.export).expect("the export is plain JSON");
        println!("{written}");
    }

    if inspected.export.valid {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// A pipeline's export, and the files and diagnostics behind it for printing to stderr.
pub(crate) struct Inspected {
    pub(crate) files: PipelineFiles,
    pub(crate) diagnostics: Vec<Diagnostic>,
    pub(crate) export: Export,
}

/// Loads and builds the pipeline at `path` without a device, describing it whether or not it is valid.
pub(crate) fn inspect_pipeline(
    registry: &PipelineRegistry,
    path: &Path,
    vars: &[(String, String)],
) -> Inspected {
    let supplied = load::supplied(vars);
    let read = av_toolbox_pipeline::load(path, &DiskFiles, &supplied);
    let pipeline = path.display().to_string();

    let loaded = match read {
        Ok(loaded) => loaded,
        Err(failure) => {
            let files = PipelineFiles::new(failure.sources);
            let diagnostics = failure.diagnostics.into_inner();
            let exported = export::diagnostics(&files, &diagnostics);
            let export = Export {
                format_version: FORMAT_VERSION,
                valid: false,
                pipeline,
                hash: None,
                variables: None,
                stages: None,
                diagnostics: exported,
            };

            return Inspected {
                files,
                diagnostics,
                export,
            };
        },
    };

    let files = PipelineFiles::new(loaded.sources);
    let variables = export::variables(&loaded.variables);
    let stages = export::stages(&loaded.specs);

    let built = Pipeline::build(registry, &loaded.specs);
    let (valid, diagnostics) = match built {
        Ok(pipeline) => (true, pipeline.warnings().to_vec()),
        Err(error) => (false, error.into_diagnostics().into_inner()),
    };

    let mut hash = None;
    if valid {
        let digest = hash::pipeline_hash(&stages);
        hash = Some(digest);
    }

    let exported = export::diagnostics(&files, &diagnostics);
    let export = Export {
        format_version: FORMAT_VERSION,
        valid,
        pipeline,
        hash,
        variables: Some(variables),
        stages: Some(stages),
        diagnostics: exported,
    };

    Inspected {
        files,
        diagnostics,
        export,
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use serde_json::json;
    use tempfile::TempDir;

    use super::*;

    const PIPELINE: &str = r#"
[variable.width]
type = "integer"
default = 1280

[variable.output]
type = "path"
default = "out.y4m"

[source.input]
type = "y4m/stdin"

[filter.resize]
type = "resize/spline36"
inputs = ["input"]
args.width = "{{ width }}"
args.height = 720

[sink.output]
type = "y4m/pipe"
inputs = ["resize"]
args.path = "{{ output }}"
"#;

    const REORDERED: &str = r#"
# The same pipeline with its sections moved and comments added.

[sink.output]
type = "y4m/pipe"
inputs = ["resize"]
args.path = "{{ output }}"

[variable.output]
type = "path"
default = "out.y4m"

[filter.resize]
type = "resize/spline36"
inputs = ["input"]
# Height first this time.
args.height = 720
args.width = "{{ width }}"

[variable.width]
type = "integer"
default = 1280

[source.input]
type = "y4m/stdin"
"#;

    const UNUSED_FILTER: &str = r#"
[source.input]
type = "y4m/stdin"

[filter.spare]
type = "resize/spline36"
inputs = ["input"]
args.width = 640
args.height = 360

[sink.output]
type = "y4m/stdout"
inputs = ["input"]
"#;

    const CALLED_PART: &str = r#"
[variable.width]
type = "integer"
default = 640

[input.clip]

[filter.shrink]
type = "resize/spline36"
inputs = ["clip"]
args.width = "{{ width }}"
args.height = 360

[output.small]
inputs = ["shrink"]
"#;

    const CALLING: &str = r#"
[source.input]
type = "y4m/stdin"

[call.parts]
path = "part.toml"
inputs = { clip = "input" }

[sink.output]
type = "y4m/stdout"
inputs = ["parts"]
"#;

    fn written(folder: &TempDir, name: &str, content: &str) -> PathBuf {
        let path = folder.path().join(name);
        std::fs::write(&path, content).expect("writes the pipeline");

        path
    }

    fn inspected(path: &Path, vars: &[(String, String)]) -> Export {
        let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
        let inspected = inspect_pipeline(&registry, path, vars);

        inspected.export
    }

    fn keys(export: &Export) -> Vec<String> {
        let stages = export.stages.as_ref().expect("stages");
        stages.iter().map(|stage| stage.key.clone()).collect()
    }

    #[test]
    fn a_valid_pipeline_exports_its_variables_stages_and_hash() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let path = written(&folder, "pipeline.toml", PIPELINE);

        let export = inspected(&path, &[]);
        let variables = export.variables.as_ref().expect("variables");
        let hash = export.hash.as_deref().expect("a hash");

        assert!(export.valid, "{:?}", export.diagnostics);
        assert_eq!(export.format_version, 2);
        assert_eq!(variables["width"].variable_type, "integer");
        assert_eq!(variables["width"].value, json!(1280));
        assert_eq!(keys(&export), ["source.input", "filter.resize", "sink.output"]);
        assert!(hash.starts_with("sha256:"), "{hash}");
        assert_eq!(hash.len(), "sha256:".len() + 64);
    }

    #[test]
    fn comments_and_section_order_do_not_change_the_hash() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let original = written(&folder, "pipeline.toml", PIPELINE);
        let reordered = written(&folder, "reordered.toml", REORDERED);

        let original_export = inspected(&original, &[]);
        let reordered_export = inspected(&reordered, &[]);

        assert!(original_export.hash.is_some());
        assert_eq!(original_export.hash, reordered_export.hash);
    }

    #[test]
    fn a_var_reaching_an_arg_changes_the_hash() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let path = written(&folder, "pipeline.toml", PIPELINE);
        let wider = [("width".to_string(), "1920".to_string())];

        let default_export = inspected(&path, &[]);
        let wider_export = inspected(&path, &wider);

        assert!(wider_export.valid);
        assert_ne!(default_export.hash, wider_export.hash);
    }

    #[test]
    fn a_pipeline_with_only_warnings_is_valid_and_hashed() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let path = written(&folder, "pipeline.toml", UNUSED_FILTER);

        let export = inspected(&path, &[]);

        assert!(export.valid);
        assert!(export.hash.is_some());
        assert_eq!(export.diagnostics[0].severity, "warning");
        assert_eq!(export.diagnostics[0].code, "unused-stage");
    }

    #[test]
    fn a_called_file_exports_its_prefixed_stages_but_not_its_variables() {
        let folder = tempfile::tempdir().expect("a temp dir");
        written(&folder, "part.toml", CALLED_PART);
        let path = written(&folder, "pipeline.toml", CALLING);

        let export = inspected(&path, &[]);
        let variables = export.variables.as_ref().expect("variables");

        assert!(export.valid, "{:?}", export.diagnostics);
        assert!(keys(&export).contains(&"filter.parts.shrink".to_string()));
        assert!(variables.is_empty());
    }

    #[test]
    fn a_load_failure_exports_null_stages_and_a_located_error() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let path = written(
            &folder,
            "pipeline.toml",
            "[sources.input]\ntype = \"y4m/stdin\"\n",
        );

        let export = inspected(&path, &[]);

        assert!(!export.valid);
        assert_eq!(export.hash, None);
        assert!(export.variables.is_none());
        assert!(export.stages.is_none());
        assert_eq!(export.diagnostics[0].code, "invalid-definition");
        assert_eq!(export.diagnostics[0].location.line, Some(1));
        assert_eq!(export.diagnostics[0].location.column, Some(2));
    }

    #[test]
    fn a_missing_file_exports_an_error_without_a_location() {
        let path = Path::new("/nonexistent-directory/pipeline.toml");

        let export = inspected(path, &[]);

        assert!(!export.valid);
        assert_eq!(export.diagnostics[0].code, "unreadable-file");
        assert_eq!(export.diagnostics[0].location.file, None);
        assert_eq!(export.diagnostics[0].location.line, None);
    }

    #[test]
    fn a_build_failure_exports_its_stages_and_the_error() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let misspelt = PIPELINE.replace("resize/spline36", "resize/splin36");
        let path = written(&folder, "pipeline.toml", &misspelt);

        let export = inspected(&path, &[]);

        assert!(!export.valid);
        assert_eq!(export.hash, None);
        assert_eq!(keys(&export).len(), 3);
        assert_eq!(export.diagnostics[0].code, "unknown-stage");
    }
}

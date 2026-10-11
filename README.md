# av-toolbox

GPU accelerated video filtering pipelines. Built on the learnings of [av-denoise](https://github.com/ChillFish8/av-denoise)


## Installing

Each release has a prebuilt binary per platform, with ffms2 and FFmpeg linked in.

| Archive                           | Backends     |
|-----------------------------------|--------------|
| `av-toolbox-linux-x86.tar.gz`     | vulkan, cuda |
| `av-toolbox-win64-x86.zip`        | vulkan, cuda |
| `av-toolbox-macos-aarch64.tar.gz` | metal        |

Optional models are not bundled. Fetch one with `av-toolbox pull <name>`.

### Docker

Images are published to `ghcr.io/chillfish8/av-toolbox` as `vulkan-<version>`, `cuda-<version>` and
`rocm-<version>`. Models are stored in `/models`, so mount a volume there to keep them between runs.

```bash
docker run --rm -v avtb-models:/models ghcr.io/chillfish8/av-toolbox:vulkan-latest pull artcnn
```

## Building

Clone with `--recursive`, or run `just submodules` after cloning, to fetch the vendored BestSource sources.

Pick a backend when building. `vulkan` works on most machines, `metal` on macOS.

```bash
cargo install --path av-toolbox --features vulkan
```

To link ffms2 and FFmpeg statically, as the release binaries do, build them from vcpkg first.
`just static-deps` needs vcpkg's host tools: `git`, `curl`, `zip`, `unzip`, `tar`, `pkg-config`, `ninja`, `nasm` and a C++ compiler.

```bash
just static-deps
just build-static
```

## Running a pipeline

```bash
av-toolbox execute examples/passthrough.toml < input.y4m > output.y4m
```

## Variables

A pipeline declares its settings under `[variable.<name>]` and uses them with `{{ name }}`.

```toml
[variable.input]
type = "path"
description = "The video to decode"
default = "input.mkv"

[source.clip]
type = "ffms2/file"
args.path = "{{ input }}"
```

The type is one of `string`, `path`, `integer`, `float` or `bool`. A variable may also set `min` and `max` for
numbers, or `choices` for strings and integers. A variable with no `default` must be given.

```bash
av-toolbox execute examples/scenes.toml --var input=clip.mkv
AVTB_VAR_INPUT=clip.mkv av-toolbox execute examples/scenes.toml
```

`--var` wins over `AVTB_VAR_<NAME>`, which wins over the default. A path is relative to the file it is written in,
or to the working directory when given on the command line.

## Splitting pipelines

A pipeline can call another pipeline file. The called file declares the streams it takes as `[input.<name>]` and
the streams it gives back as `[output.<name>]`.

```toml
# parts/shrink.toml
[variable.width]
type = "integer"
default = 640

[input.clip]

[filter.shrink]
type = "resize/bilinear"
inputs = ["clip"]
args.width = "{{ width }}"
args.height = 360

[output.shrunk]
inputs = ["shrink"]
```

```toml
# pipeline.toml
[call.parts]
path = "parts/shrink.toml"
inputs = ["input"]
variables.width = 1280

[sink.output]
type = "y4m/pipe"
inputs = ["parts"]
args.path = "{{ output }}"
```

Read a call's output as `<call>.<output>`, or as `<call>` when the file has one output. Stages from a called file
show up as `<call>.<stage>`, such as `parts.shrink`.

## Scene breaks

A stage that works one scene at a time asks for scene breaks itself, and the pipeline finds them once for every
stage asking. `chunked.toml` runs a denoise on two workers, each handed whole scenes. `metadata/ndjson` writes each
frame's scene when given `args.scenes`.

```bash
av-toolbox execute examples/chunked.toml --var input=clip.mkv --var output=output.y4m
av-toolbox execute examples/scenes.toml --var input=clip.mkv --var scenes=scenes.ndjson
```

## Resizing and generated frames

The `resize/*` filters resize a clip with a point, bilinear, bicubic, lanczos or spline kernel. `solid/fill` and
`image/file` make a clip from a colour or a PNG or JPEG image, and `solid/like` and `image/like` make one in the
size, format and length of another clip.

```bash
av-toolbox execute examples/resize.toml --var width=1280 --var height=720 --var input=clip.mkv --var output=output.y4m
av-toolbox execute examples/solid.toml --var colour=#202020 --var output=solid.y4m
av-toolbox execute examples/image.toml --var image=logo.png --var output=logo.y4m
av-toolbox execute examples/recolour-lines.toml --var colour=#ff0000 --var input=clip.mkv --var output=output.y4m
```

## Cropping

`crop/manual` removes a given number of samples from each edge. `crop/auto` finds the black bars around the picture
and removes them, measuring 13 frames spread across the clip before the run starts. It needs a file source, and has
to come before any stage that changes the frame size.

```toml
[filter.crop]
type = "crop/manual"
inputs = ["input"]
args.top = 140
args.bottom = 140
```

`examples/crop.toml` uses `crop/auto` instead, finding the bars for itself.

```bash
av-toolbox execute examples/crop.toml --var input=clip.mkv --var output=output.y4m
```

## Drawing a pipeline

```bash
av-toolbox explain examples/scenes.toml
av-toolbox explain examples/scenes.toml --html
av-toolbox explain examples/scenes.toml --html --path graph.html
av-toolbox explain examples/scenes.toml --structure-only
av-toolbox explain examples/scenes.toml --profile scenes.profile.toml
```

## Profiling a pipeline

`profile` runs a pipeline three times over the same frames, with its sinks replaced so nothing is written. It
prints how much of each frame every stage takes, and writes the worker counts it found to a profile file.

```bash
av-toolbox profile examples/scenes.toml --var input=clip.mkv
av-toolbox profile examples/scenes.toml --var input=clip.mkv --frames 600 --output scenes.profile.toml
av-toolbox execute examples/scenes.toml --var input=clip.mkv --var output=out.y4m --profile scenes.profile.toml
```

A `workers` written in the pipeline always wins over the profile. Profiling needs a file source, since a stream
such as standard input can only be read once.

## Finding stages and devices

```bash
av-toolbox list-stages
av-toolbox list-stages y4m/pipe
av-toolbox list-devices
```

## Licence

Licensed under the European Union Public Licence v1.2 only (EUPL-1.2). Later versions of the EUPL do not apply.
See [LICENSE](LICENSE).

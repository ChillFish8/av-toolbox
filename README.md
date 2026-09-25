# av-toolbox

GPU accelerated video filtering pipelines. Built on the learnings of [av-denoise](https://github.com/ChillFish8/av-denoise)


## Building

Pick a backend when building. `vulkan` works on most machines, `metal` on macOS.

```bash
cargo install --path av-toolbox --features vulkan
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
# parts/scenes.toml
[variable.speed]
type = "string"
default = "standard"
choices = ["standard", "fast"]

[input.clip]

[transform.detect]
type = "scene/detect"
inputs = ["clip"]
args.mode = "natural"
args.speed = "{{ speed }}"

[output.scenes]
inputs = ["detect"]
```

```toml
# chunked.toml
[call.scenes]
path = "parts/scenes.toml"
inputs = ["input"]
variables.speed = "fast"

[sink.output]
type = "y4m/stdout"
inputs = ["scenes"]
```

Read a call's output as `<call>.<output>`, or as `<call>` when the file has one output. Stages from a called file
show up as `<call>.<stage>`, such as `scenes.detect`.

```bash
av-toolbox execute examples/chunked.toml < input.y4m > output.y4m
```

## Drawing a pipeline

```bash
av-toolbox explain examples/scenes.toml
av-toolbox explain examples/scenes.toml --html
av-toolbox explain examples/scenes.toml --html --path graph.html
av-toolbox explain examples/scenes.toml --structure-only
```

## Finding stages and devices

```bash
av-toolbox list-stages
av-toolbox list-stages y4m/pipe
av-toolbox list-devices
```
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
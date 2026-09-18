# av-toolbox

GPU accelerated video filtering pipelines. Built on the learnings of [av-denoise](https://github.com/ChillFish8/av-denoise)


## Rendering pipeline dependency graph

```bash
av-toolbox render-graph --pipeline ./example/pipeline.toml 
```

## Running pipeline

```bash
av-toolbox execute --pipeline ./example/pipeline.toml --arg 'var_a'
```
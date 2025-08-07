# Meshtryoshka

This is the code for Meshtryoshka, a differentiable rendering framework that operates via mesh rendering.

## How does this work?

The underlying scene representation is a sparse cubic voxel grid. Each corner stores a signed distance and color (as spherical harmonics). During every training step, we extract several level sets of the corresponding signed distance field. We then non-differentiably render the level sets and differentiably composite the resulting images using a NeuS-like rendering formulation. This yields a final image on which a loss can be computed.

We use the [slang.D](https://shader-slang.org/) language to implement our kernels for differentiable marching cubes and triangle rasterization.

## Environment Setup

```
git clone git@github.com:dcharatan/meshtryoshka.git
cd meshtryoshka
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data
Our code assumes the datasets are stored under the following relative directories:
```
datasets/mipnerf360
datasets/nerf_synthetic
```

The datasets can be downloaded from the respective project pages:
```
https://jonbarron.info/mipnerf360/
https://www.matthewtancik.com/nerf
```

## Running the Code

The code can be run as follows:
`WORKSPACE=workspace python3 -m meshtryoshka +experiment=mip360 dataset.scene=bicycle enable_viewer=true`

**Compiling the Slang kernels will take a few minutes the first time you run the code. It will be faster after that.**

We have the following experiments:
```
mip360
mip360_24gb
mip360_140gb
nerf_synthetic
```

We provide a script to run the MIP360 scenes sequentially:
`python -m scripts.train_mip360`. Due to differences in how the scenes are scaled, for the flowers scene, we hardcode a different min-depth from the other scenes (see script).

## Fused Rasterization vs. Explicit Rasterization

There are two rasterization modes that can be used with Meshtryoshka:

- **Explicit:** In this mode, non-differentiable rasterization is completely separate from differentiable compositing. Rasterization produces an image where each pixel contains a triangle index and barycentric coordinates that represent a ray-triangle intersection. This mode is conceptually easiest to understand. However, it requires activations of `12*B*S*H*W` bytes to be stored during optimization, where `B` is the number of rendered cameras, `S` is the number of shells, and `H` and `W` are the image dimensions. This makes it less practical for achieving high-quality results given a limited VRAM budget.
- **Fused:** In this mode, triangle hit detection (the final rasterization step) is combined with interpolation and deferred shading. This both avoids having to pay the memory cost of storing per-pixel barycentric coordinates and triangle indices in global memory and results in faster runtime. This formulation is equivalent to the above one, but slightly more complicated to implement. This is the default in our code.

Use `model.render_mode=explicit` to switch from the default `fused` mode to the `explicit` mode.

## Results
The max resolution of our method is limited by GPU memory. Below, we report PSNR metrics on the main (`mip360`) and 24GB (`mip360_24gb`) configurations.

A100 (80GB) results using `mip360` configuration:
| Evaluation | mean | bicycle | bonsai | counter | flowers | garden | kitchen | room | stump | treehill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All layers | 24.32 | 22.02 | 27.79 | 25.37 | 19.10 | 24.63 | 26.86 | 27.71 | 24.26 | 21.15 |
| Zero Level Set | 24.27 | 22.00 | 27.72 | 25.32 | 19.09 | 24.59 | 26.77 | 27.63 | 24.22 | 21.10 |

4090 (24GB) results using `mip360_24gb` configuration:
| Evaluation | mean | bicycle | bonsai | counter | flowers | garden | kitchen | room | stump | treehill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All layers | 23.66 | 21.42 | 26.72 | 24.78 | 18.94 | 23.90 | 25.64 | 26.61 | 23.75 | 21.17 |
| Zero Level Set | 23.62 | 21.40 | 26.65 | 24.75 | 18.92 | 23.87 | 25.56 | 26.54 | 23.71 | 21.13 |

## Common Issues
- **Slang compilation error** The slangtorch version and CUDA Toolkit version must be compatible. We have tested Slangtorch 1.3.7, CUDA 12.4, and Slangtorch 1.3.11, CUDA 12.6. In requirements.txt, we have pinned slangtorch 1.3.11, but please check the CUDA version you are using. Additionally, ensure the environment variables `CUDA_HOME`, `TORCH_CUDA_ARCH_LIST` are set to be compatible with your system.

- **Why does the code hang?** This may be because Slang compilation is hanging. Deleting the following can help: `<your python>/site-packages/triangle_rasterization/slang/.slangtorch_cache`, `triangle_rasterization/slang/.slangtorch_cache`, `triangle_extraction/slang/.slangtorch_cache`.

## Workflow

If you're building on top of this code, make sure you install the following software:

- **`clang-format`:** `sudo apt install clang-format`
- **The VS Code Slang Extension:** `shader-slang.slang-language-extension`
- **Ninja:** For some reason, in order to discover tests in VS Code, this has to be installed via `sudo apt install ninja-build` rather than via Pip.

### Why is my Slang function not differentiable?

- Confirm that your function and its sub-functions are marked with `[Differentiable]`.
- Confirm that any structs returned by sub-functions inherit from `IDifferentiable`.
- Note: `const` vs. non-`const` (or alternatively, `let` vs. `var`) makes no difference.

## Acknowledgements
We thank the contributors of slang.D for their outstanding work. The language proved invaluable in enabling the efficient implementation of our kernels. We are also deeply grateful to Saipraveen Bangaru who answered many of our questions about the language.

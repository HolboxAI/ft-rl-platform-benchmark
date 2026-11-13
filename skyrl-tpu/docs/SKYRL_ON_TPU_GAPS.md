# SkyRL and Miles on TPU: what was missing, why, and the patches

## Part 1: SkyRL

### 1.1 What the original SkyRL did not have

- SkyRL had no TPU support. It supported only NVIDIA GPUs.
- SkyRL had no communication method for TPU chips. It used only NCCL, which works only on NVIDIA GPUs.
- SkyRL had no text engine for TPU. It used vLLM on GPUs.
- SkyRL had no method to send new weights to an engine without GPU links.
- SkyRL could not read image data without the GPU version of vLLM.

### 1.2 Why SkyRL did not work on TPU

- The training code called GPU functions directly at many points. For example: select the device, wait for the device, read the memory, clear the memory. On a TPU VM there is no GPU, so the first call stopped the program.
- The training processes started their connection with NCCL. A TPU VM has no NCCL, so the processes could not connect.
- After each step, SkyRL sent the weights to vLLM through GPU links. A TPU VM has none of these links, so the engine never got the new weights.
- The TPU library for PyTorch is new. Some PyTorch and model operations did not work with it:
  - The tool that divides the model across chips moved weights in a way that the TPU library rejects.
  - The position code of the model wrote into a part of a table in place. The TPU library stops on this.
  - One small table of the image position code stayed on the CPU. The calculation then used data from two devices and failed.
- Each process that loaded PyTorch also opened a TPU chip. Helper processes took chips from the training processes.

### 1.3 The patches for SkyRL

| Patch | What it does |
|---|---|
| Device layer | A new layer receives all device calls. One switch selects GPU or TPU. |
| TPU communication | The processes connect with the TPU method, with a CPU method for the CPU parts. |
| Chip control | Each training process opens only its own chip. Other processes open no chip. |
| Weight move fix | The weight move uses a method that the TPU library accepts. |
| Position code fixes | The text position code does not write in place. The image position table moves to the correct device. |
| New weight transfer | The trainer writes the weights to memory. A helper starts the engine again with these weights. |
| Image data reader | The trainer reads image data in a simple format. It does not need vLLM. |
| Own text engine | A new engine server uses the Hugging Face generate function on the TPU. It accepts the requests that SkyRL sends. |

## Multimodal Datasets for Training Python Copilots from Source Code Analysis

<img src="https://raw.githubusercontent.com/matlok-ai/python-copilot-image-and-audio-examples/main/static/matlok-multimodal-python-copilot-training-datasets-intro-1.jpg" alt="Multimodal Datasets for Training Python Copilots from Source Code Analysis" width="500" style="display: block; margin: auto;"/>

<p align="center">
  <audio controls>
    <source src="https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/transformers/examples/pytorch/contrastive-image-text/audio.func.alp.question.run_clip.mp3" type="audio/mp3">
    Play question.run_clip.mp3
  </audio>
</p>

Below are image and audio (narrated mp3) samples extracted from the [matlok datasets](https://huggingface.co/datasets/matlok/multimodal-python-copilot-training-overview). These samples provide an overview for how the images look, and how the mp3s are structured with an answer and a question from the image's knowledge graph text box.

Welcome to the matlok multimodal python copilot training datasets. This is an overview for our training and fine-tuning datasets found below:

- ~2.35M unique source code rows
- ~1.7M instruct alpaca yaml text rows
- ~923K png knowledge graph images with alpaca text description
- ~410K mp3s for ~2 years of continuous audio playtime
- requires 1.2 TB storage on disk

Please reach out if you find an issue or want help with a similar dataset. We want to make it easier to create and share large datasets:
hello@matlok.ai

## Python Copilot Training using Knowledge Graph Images

These are knowledge graphs created for training generative ai models on how writing python CLIP transformer code by understanding an overview on:

- classes
- base classes for inheritance and polymorphism
- global functions
- imports

### Class - Knowledge Graph Images

Here are samples from the [python copilot class image knowledge graph dataset (304 GB)](https://huggingface.co/datasets/matlok/python-image-copilot-training-using-class-knowledge-graphs). These images attempt to teach how to use software with a networkx graph saved as a png with an alpaca text box:

#### How to use the transformers/src/transformers/models/clip/configuration_clip.py CLIPConfig class

<img src="https://raw.githubusercontent.com/matlok-ai/python-copilot-image-and-audio-examples/main/png/transformers/src/transformers/models/clip/image.class.configuration_clip.CLIPConfig.png" alt="How to use the transformers/src/transformers/models/clip/configuration_clip.py CLIPConfig class" width="500" style="display: block; margin: auto;"/>

#### How to use the transformers/src/transformers/models/clip/configuration_clip.py CLIPOnnxConfig class

<img src="https://raw.githubusercontent.com/matlok-ai/python-copilot-image-and-audio-examples/main/png/transformers/src/transformers/models/clip/image.class.configuration_clip.CLIPOnnxConfig.png" alt="How to use the transformers/src/transformers/models/clip/configuration_clip.py CLIPOnnxConfig class" width="500" style="display: block; margin: auto;"/>

#### How to use the transformers/src/transformers/models/clip/tokenization_clip.py CLIPTokenizer class

<img src="https://raw.githubusercontent.com/matlok-ai/python-copilot-image-and-audio-examples/main/png/transformers/src/transformers/models/clip/image.class.tokenization_clip.CLIPTokenizer.png" alt="How to use the transformers/src/transformers/models/clip/tokenization_clip.py CLIPTokenizer class" width="500" style="display: block; margin: auto;"/>

### Base Class - Inheritance and Polymorphism Knowledge Graph Images

Here are samples from the [python copilot base class inheritance and polymorphism image knowledge graph dataset (135 GB)](https://huggingface.co/datasets/matlok/python-image-copilot-training-using-inheritance-knowledge-graphs). These images attempt to teach how to use software with a networkx graph saved as a png with an alpaca text box:

#### How to use the transformers/src/transformers/models/clip/configuration_clip.py CLIPConfig inherited base class(es)

<img src="https://raw.githubusercontent.com/matlok-ai/python-copilot-image-and-audio-examples/main/png/transformers/src/transformers/models/clip/image.base.configuration_clip.CLIPConfig.png" alt="How to use the transformers/src/transformers/models/clip/configuration_clip.py CLIPConfig inherited base class" width="500" style="display: block; margin: auto;"/>

#### How to use the transformers/src/transformers/models/clip/tokenization_clip_fast.py CLIPTokenizerFast inherited base class(es)

<img src="https://raw.githubusercontent.com/matlok-ai/python-copilot-image-and-audio-examples/main/png/transformers/src/transformers/models/clip/image.base.tokenization_clip_fast.CLIPTokenizerFast.png" alt="How to use the transformers/src/transformers/models/clip/tokenization_clip_fast.py CLIPTokenizerFast inherited base class" width="500" style="display: block; margin: auto;"/>

### Global Functions - Knowledge Graph Images

Here are samples from the [python copilot global functions image knowledge graph dataset (130 GB)](https://huggingface.co/datasets/matlok/python-image-copilot-training-using-functions-knowledge-graphs). These images attempt to teach how to use software with a networkx graph saved as a png with an alpaca text box:

#### How to use the transformers/src/transformers/models/clip/convert_clip_original_pytorch_to_hf.py global functions

<img src="https://raw.githubusercontent.com/matlok-ai/python-copilot-image-and-audio-examples/main/png/transformers/src/transformers/models/clip/image.func.convert_clip_original_pytorch_to_hf.png" alt="How to use the transformers/src/transformers/models/clip/convert_clip_original_pytorch_to_hf.py global functions" width="500" style="display: block; margin: auto;"/>

#### How to use the transformers/src/transformers/models/clip/tokenization_clip.py global functions

<img src="https://raw.githubusercontent.com/matlok-ai/python-copilot-image-and-audio-examples/main/png/transformers/src/transformers/models/clip/image.func.tokenization_clip.png" alt="How to use the transformers/src/transformers/models/clip/tokenization_clip.py global functions" width="500" style="display: block; margin: auto;"/>

### Imports - Knowledge Graph Images

Here are samples from the [python copilot imports image knowledge graph dataset (211 GB)](https://huggingface.co/datasets/matlok/python-image-copilot-training-using-import-knowledge-graphs). These images attempt to teach how to use software with a networkx graph saved as a png with an alpaca text box:

#### How to use the transformers/src/transformers/models/clip/configuration_clip.py imports like the CLIPConfig class

<img src="https://raw.githubusercontent.com/matlok-ai/python-copilot-image-and-audio-examples/main/png/transformers/src/transformers/models/clip/image.import.configuration_clip.CLIPConfig.png" alt="How to use the transformers/src/transformers/models/clip/configuration_clip.py imports like the CLIPConfig class" width="500" style="display: block; margin: auto;"/>

#### How to use the transformers/src/transformers/models/clip/configuration_clip.py imports like the CLIPTextConfig class

<img src="https://raw.githubusercontent.com/matlok-ai/python-copilot-image-and-audio-examples/main/png/transformers/src/transformers/models/clip/image.import.configuration_clip.CLIPTextConfig.png" alt="How to use the transformers/src/transformers/models/clip/configuration_clip.py imports like the CLIPTextConfig class" width="500" style="display: block; margin: auto;"/>

#### How to use the transformers/src/transformers/models/clip/configuration_clip.py imports like the CLIPVisionConfig class

<img src="https://raw.githubusercontent.com/matlok-ai/python-copilot-image-and-audio-examples/main/png/transformers/src/transformers/models/clip/image.import.configuration_clip.CLIPVisionConfig.png" alt="How to use the transformers/src/transformers/models/clip/configuration_clip.py imports like the CLIPVisionConfig class" width="500" style="display: block; margin: auto;"/>

#### How to use the transformers/src/transformers/models/clip/tokenization_clip_fast.py imports like the CLIPTokenizerFast class

<img src="https://raw.githubusercontent.com/matlok-ai/python-copilot-image-and-audio-examples/main/png/transformers/src/transformers/models/clip/image.import.tokenization_clip_fast.CLIPTokenizerFast.png" alt="How to use the transformers/src/transformers/models/clip/tokenization_clip_fast.py imports like the CLIPTokenizerFast class" width="500" style="display: block; margin: auto;"/>

## Audio Training Examples - Question and Answering in Alpaca

Below are extracted question and answer mp3 samples. Each mp3 is either a recording of the alpaca question or answer. Question mp3s use a different speaker than the answer mp3 voice.

Question | Answer
--- | ---
![Play question.run_clip.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/transformers/examples/pytorch/contrastive-image-text/audio.func.alp.question.run_clip.mp3) | ![Play answer.run_clip.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/transformers/examples/pytorch/contrastive-image-text/audio.func.alp.answer.run_clip.mp3)
![Play question.run_clip.Transform.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/transformers/examples/pytorch/contrastive-image-text/audio.base.alp.question.run_clip.Transform.mp3) | ![Play answer.run_clip.Transform.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/transformers/examples/pytorch/contrastive-image-text/audio.base.alp.answer.run_clip.Transform.mp3)
![Play question.run_generation_contrastive_search.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/transformers/examples/pytorch/text-generation/audio.func.alp.question.run_generation_contrastive_search.mp3) | ![Play answer.run_generation_contrastive_search.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/transformers/examples/pytorch/text-generation/audio.func.alp.answer.run_generation_contrastive_search.mp3)
![Play question.run_generation.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/transformers/examples/pytorch/text-generation/audio.func.alp.question.run_generation.mp3) | ![Play answer.run_generation.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/transformers/examples/pytorch/text-generation/audio.func.alp.answer.run_generation.mp3)
![Play question.checkpointing.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/accelerate/examples/by_feature/audio.func.alp.question.checkpointing.mp3) | ![Play answer.checkpointing.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/accelerate/examples/by_feature/audio.func.alp.answer.checkpointing.mp3)
![Play question.fully_sharded_data_parallel.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/pytorch/torch/distributed/fsdp/audio.func.alp.question.fully_sharded_data_parallel.mp3) | ![Play answer.fully_sharded_data_parallel.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/pytorch/torch/distributed/fsdp/audio.func.alp.answer.fully_sharded_data_parallel.mp3)
![Play question.fully_sharded_data_parallel.FullyShardedDataParallel.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/pytorch/torch/distributed/fsdp/audio.base.alp.question.fully_sharded_data_parallel.FullyShardedDataParallel.mp3) | ![Play answer.fully_sharded_data_parallel.FullyShardedDataParallel.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/pytorch/torch/distributed/fsdp/audio.base.alp.answer.fully_sharded_data_parallel.FullyShardedDataParallel.mp3)
![Play question.convert-hf-to-gguf.QwenModel.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/llama.cpp/audio.base.alp.question.convert-hf-to-gguf.QwenModel.mp3) | ![Play answer.convert-hf-to-gguf.QwenModel.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/llama.cpp/audio.base.alp.answer.convert-hf-to-gguf.QwenModel.mp3)
![Play question.engine.DeepSpeedEngine.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/deepspeed/deepspeed/runtime/audio.base.alp.question.engine.DeepSpeedEngine.mp3) | ![Play answer.engine.DeepSpeedEngine.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/deepspeed/deepspeed/runtime/audio.base.alp.answer.engine.DeepSpeedEngine.mp3)
![Play question.flash_mixtral_modeling.MixtralModel.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/text-generation-inference/server/text_generation_server/models/custom_modeling/audio.base.alp.question.flash_mixtral_modeling.MixtralModel.mp3) | ![Play answer.flash_mixtral_modeling.MixtralModel.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/text-generation-inference/server/text_generation_server/models/custom_modeling/audio.base.alp.answer.flash_mixtral_modeling.MixtralModel.mp3)
![Play question.flash_mixtral_modeling.MixtralLayer.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/text-generation-inference/server/text_generation_server/models/custom_modeling/audio.base.alp.question.flash_mixtral_modeling.MixtralLayer.mp3) | ![Play answer.flash_mixtral_modeling.MixtralLayer.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/text-generation-inference/server/text_generation_server/models/custom_modeling/audio.base.alp.answer.flash_mixtral_modeling.MixtralLayer.mp3)
![Play question.flash_mixtral_modeling.MixtralAttention.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/text-generation-inference/server/text_generation_server/models/custom_modeling/audio.base.alp.question.flash_mixtral_modeling.MixtralAttention.mp3) | ![Play answer.flash_mixtral_modeling.MixtralAttention.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/text-generation-inference/server/text_generation_server/models/custom_modeling/audio.base.alp.answer.flash_mixtral_modeling.MixtralAttention.mp3)
![Play question.flash_mixtral_modeling.BlockSparseMoE.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/text-generation-inference/server/text_generation_server/models/custom_modeling/audio.import.alp.question.flash_mixtral_modeling.BlockSparseMoE.mp3) | ![Play answer.flash_mixtral_modeling.BlockSparseMoE.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/text-generation-inference/server/text_generation_server/models/custom_modeling/audio.import.alp.answer.flash_mixtral_modeling.BlockSparseMoE.mp3)
![Play question.flash_mixtral_modeling.MixtralModel.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/text-generation-inference/server/text_generation_server/models/custom_modeling/audio.import.alp.question.flash_mixtral_modeling.MixtralModel.mp3) | ![Play answer.flash_mixtral_modeling.MixtralModel.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/text-generation-inference/server/text_generation_server/models/custom_modeling/audio.import.alp.answer.flash_mixtral_modeling.MixtralModel.mp3)
![Play question.flash_llama_modeling.FlashLlamaAttention.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/text-generation-inference/server/text_generation_server/models/custom_modeling/audio.base.alp.question.flash_llama_modeling.FlashLlamaAttention.mp3) | ![Play answer.flash_llama_modeling.FlashLlamaAttention.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/text-generation-inference/server/text_generation_server/models/custom_modeling/audio.base.alp.answer.flash_llama_modeling.FlashLlamaAttention.mp3)
![Play question.flash_llama_modeling.FlashLlamaLayer.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/text-generation-inference/server/text_generation_server/models/custom_modeling/audio.base.alp.question.flash_llama_modeling.FlashLlamaLayer.mp3) | ![Play answer.flash_llama_modeling.FlashLlamaLayer.mp3](https://github.com/matlok-ai/python-copilot-image-and-audio-examples/raw/main/mp3/text-generation-inference/server/text_generation_server/models/custom_modeling/audio.base.alp.answer.flash_llama_modeling.FlashLlamaLayer.mp3)

## Thanks for reading and your time

<img src="https://raw.githubusercontent.com/matlok-ai/python-copilot-image-and-audio-examples/main/static/lok-1-python.jpg" alt="Thanks for reading" width="500" style="display: block; margin: auto;"/>

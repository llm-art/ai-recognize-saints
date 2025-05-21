

## Models

| Name             | Model Type   | Architecture            | Vision Component       | Text Component         | Training Objective                    | Loss Function          | Parameter Count | Token Limit | Image Resolution | Training Data Size     | License         |
|------------------|--------------|--------------------------|------------------------|------------------------|----------------------------------------|-------------------------|------------------|--------------|-------------------|------------------------|------------------|
| [BLIP](https://huggingface.co/docs/transformers/model_doc/blip)             | VLM          | Dual Encoder + Decoder   | ViT-B/L                | BERT-style or T5       | Captioning + VQA                      | Contrastive + CE       | ~250M–500M       | 512           | 224×224           | COCO + web data         | Salesforce (open)|
| [BLIP-2](https://huggingface.co/docs/transformers/model_doc/blip-2)           | VLM          | Vision-Language Bridge   | ViT-G/14 or ViT-L      | Frozen LLM (OPT/T5)    | Pretrain + Q-former + Instruction     | Contrastive + CE       | ~1.2B            | 512–1024      | 224×224+           | Public mixes            | Salesforce (open)|
| [CLIP (ViT-B/32)](https://huggingface.co/openai/clip-vit-base-patch32)  | VLM          | Dual Encoder             | ViT-B/32               | Transformer (GPT-like) | Contrastive Learning                   | InfoNCE (Softmax)       | ~125M            | 77           | 224×224           | 400M pairs              | OpenAI (open)    |
| [Gemini](https://huggingface.co/describeai/gemini)           | LMM          | Unified Transformer      | Internal vision encoder| Gemini Text Decoder    | Multimodal Instruction Following      | Cross-Entropy          | 770M | ~128k         | Variable           | Proprietary             | Google (closed)  |
| [GIT](https://huggingface.co/docs/transformers/model_doc/git) | VLM          | Unified Transformer      | CNN or ViT             | T5-style decoder       | Captioning, VQA, Dense prediction     | Cross-Entropy          | ~750M            | 512           | 224×224           | Internal + COCO         | Google (open)    |
| [GPT-4o](https://platform.openai.com/docs/models/gpt-4o) | LMM          | Unified Transformer      | Internal vision module | GPT-4o Text Decoder    | Instruction Following + Multimodal    | Cross-Entropy          | Not disclosed    | ~128k        | Variable           | Proprietary             | OpenAI (closed)  |
| [GPT-4o Mini](https://platform.openai.com/docs/models/gpt-4o-mini)      | LMM          | Unified Transformer      | Internal vision module | GPT-4o Mini Text       | Instruction Following + Multimodal    | Cross-Entropy          | Not disclosed    | ~32k         | Variable           | Proprietary             | OpenAI (closed)  |
| [ImageBind](https://imagebind.metademolab.com/) | Multimodal   | Cross-modal Embedding    | ViT-L                  | BERT-style             | Joint embedding across modalities     | Contrastive            | ~500M+           | 512           | 224×224           | Audio+Text+Image        | Meta (open)      |
| [Kosmos-2](https://huggingface.co/docs/transformers/model_doc/kosmos-2) | VLM          | Unified Transformer      | Internal visual encoder| Transformer Decoder    | Multimodal Pretraining + VQA          | Cross-Entropy          | ~1.6B–2B         | 2048          | 224×224+           | Multimodal web data     | Microsoft (open) |
| [LLaVA (13B)](https://huggingface.co/docs/transformers/model_doc/llava) | VLM          | Vision-Language Bridge   | CLIP-ViT-L/336px       | Vicuna 13B (LLaMA)     | Instruction Tuning (VQA, GPT4Gen)     | Cross-Entropy          | ~13B             | 2048          | 336×336           | ~1.2M                  | Academic (open)  |
| [MiniGPT-4](https://minigpt-4.github.io/) | VLM          | Vision-Language Bridge   | CLIP-ViT-L/14          | Vicuna 7B (LLaMA)      | Instruction Tuning + Captioning       | Cross-Entropy          | ~7B              | 2048          | 224–336px         | ~3M + GPT4 responses    | Academic (open)  |
| [SEED](https://ailab-cvc.github.io/seed/index.html) | Multimodal   | Cross-modal Embedding    | ViT-variant            | BERT-style             | Semantic Alignment                    | Contrastive            | Experimental     | 512           | 224×224           | Multisensory            | Meta (open)      |
| [SigLIP (ViT-L/14)]([https://huggingface.co/google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-large-patch16-256))| VLM          | Dual Encoder             | ViT-L/14               | BERT-style Transformer | Contrastive Learning                   | Sigmoid                | ~478M            | 128–256       | 224–384px         | 1B+ pairs               | Google (open)    |


### Model Type

* **[VLM (Vision-Language Model)](https://huggingface.co/blog/vlms)**: Models that process both visual and textual data for tasks like captioning, VQA, and classification.
* **[LMM (Large Multimodal Model)](https://research.aimultiple.com/large-multimodal-models/)**: General-purpose models that combine vision, language, and often other modalities (audio, video) for instruction following and generation.
* **Multimodal**: Broader models trained across multiple sensory modalities (e.g., image, audio, text) and meant for embedding or alignment.

### Architecture

* **Dual Encoder:** Independent vision and text encoders projected into a shared embedding space (e.g., CLIP, SigLIP).
* **Unified Transformer**: A single transformer model that processes both visual and textual inputs (e.g., Kosmos-2, Gemini).
* **Vision-Language Bridge:** A connector module (MLP, Q-former) bridges vision encoder to a language model (e.g., BLIP-2, LLaVA).
* **Cross-modal Embedding:** Aligns multimodal embeddings into a shared semantic space (e.g., ImageBind, SEED).

### Vision Component

* **Backbone used to encode image data:** ViT-B/L, CLIP-ViT, CNN, or internal modules in closed-source models.
* **LLM or encoder used for language:** Transformer (GPT-like), BERT-style, T5-style, or frozen models.

### Training Objective

* **Contrastive Learning:** Match correct image-text pairs.
* **Captioning:** Generate descriptive text for images.
* **Instruction Following:** Respond to prompts in few-shot style.
* **VQA:** Answer questions based on image context.
* **Joint Embedding:** Learn aligned representations across modalities.

### Loss Function

* **InfoNCE (Softmax):** Contrastive loss using softmax over similarities.
* **Sigmoid:** Binary prediction for image-text match.
* **Cross-Entropy (CE):** Standard loss for classification/generation.
* **Contrastive + CE:** Hybrid training for matching and generation.

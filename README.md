# DRL Final- Fine Tuned Part

## Requirements

To install requirements:

```bash=
pip install -r requirements.txt
```

## Training

### fine tunning

To train the model(s) in the report, choose one directory (trl_DPO, trl_SFT):

```bash=
python fine_tune_generate.py # generate the sampled data
python fine_tune_llm.py # fine tuned the llm with the sampled data
```

This operation will generate the merged model under `fine_tuned_smac_adapter\merged`.

### merge and push model to local ollama

clone and compile llama.cpp

    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    cmake .

in llama.cpp

    mkdir gguf

    python3 convert_hf_to_gguf.py --outtype f16 --outfile ./gguf/gemma3-smac.gguf [path to merged_model]

in `gguf`, create a new file `Modelfile`:

```bash=
FROM gemma3

# Optional: custom metadata or tags
PARAMETER num_ctx 4096

# Load your custom model
MODEL ./gemma3-smac.gguf
```

last, create the model in ollama

    ollama create smac-gemma3-1b -f Modelfile

## Evaluation

To evaluate fine tuned model, direct run the `test_llm.py`:

```bash=
python test_llm.py
```

## Pre-trained Models

You can download pretrained models here:

- [smac-dpo-gemma3](https://ollama.com/remijang/smac-dpo-gemma3)

- [smac-sft-gemma3](https://ollama.com/remijang/smac-sft-gemma3)

## Results

Our model achieves the following performance on '3m' map from SMAC:

| Model/Method    | Avg Total Rewards | Avg Response Time |
| --------------- | ----------------- | ----------------- |
| Gemma3:4b (Raw) | about 3.21        | about 3 second    |
| Gemma3:1b (Raw) | about 1.24        | about 1.5 second  |
| SFT             | about 1.67        | about 1.5 second  |
| DPO             | about 1.7         | about 1.4 second  |

# README

## next instruction

compile llama.cpp

    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    cmake .

in llama.cpp

    mkdir gguf

    python3 convert_hf_to_gguf.py --outtype f16 --outfile ./gguf/llama3-smac.gguf [path to merged_model]

in `gguf`, create a new file `Modelfile`:
```bash=
FROM llama3

# Optional: custom metadata or tags
PARAMETER num_ctx 4096

# Load your custom model
MODEL ./llama3-smac.gguf
```

last, create the model in ollama

    ollama create smac-llama3 -f Modelfile

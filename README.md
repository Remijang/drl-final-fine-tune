# README

## next instruction

    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make

    mkdir gguf

    python3 convert_hf_to_gguf.py --outtype f16 --outfile ./gguf/llama3-smac.gguf ../fine_tuned_smac_llama3_adapter/merged_model

    cd gguf

    vim Modelfile

    ```
        FROM llama3

        # Optional: custom metadata or tags
        PARAMETER num_ctx 4096

        # Load your custom model
        MODEL ./llama3-smac.gguf
    ```

    ollama create smac-llama3 -f Modelfile

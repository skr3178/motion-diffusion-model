
1. Generate from test set prompts:
```
python -m sample.generate --model_path ./save/humanml_enc_512_50steps/model000750000.pt --num_samples 10 --num_repetitions 3
```


2. Generate from your own text file:
```
python -m sample.generate --model_path ./save/humanml_enc_512_50steps/model000750000.pt --input_text ./assets/example_text_prompts.txt
```


3. Generate from a single text prompt:
```
python -m sample.generate --model_path ./save/humanml_enc_512_50steps/model000750000.pt --text_prompt "the person walked forward and is picking up his toolbox."
```
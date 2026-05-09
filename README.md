# MARS

MARS is an efficient generation and evaluation framework for Large Language Models.

## Configuration

To set up the environment, we recommend creating a new virtual environment (e.g., using conda):

```bash
conda create -n mars python=3.10 -y
conda activate mars
```

Then, install the necessary dependencies:

```bash
cd MARS
pip install -r requirements.txt
```

## Preparation

Before running inference or evaluation, please ensure you have downloaded the required models and updated the placeholders in the scripts:

1. **Repository Root:** Update `YOUR_REPO_ROOT` to the absolute path of your current `MARS` project directory in the `.sh` scripts.
2. **Base Model:** Update `YOUR_MODEL_PATH` to the path of your downloaded base model (e.g., Llama-3, Qwen, DeepSeek).
3. **EA Model:** Update `YOUR_EA_MODEL_PATH` to the local path of your target EA model checkpoints.

## Inference

Scripts for model answer generation are located inside `mars/eval/` and `mars/evaluation/`. To generate answers for a given task, modify the target script with your specific model paths and run it.

For example, to run generation for Llama-3-Chat:

```bash
cd mars/eval
bash gen_ea_answer_llama3chat.sh
```

> **Note:** Inside the scripts, you can customize various generation parameters such as `--temperature`, `--top-k`, `--depth`, and generation strategies explicitly. The results will be automatically saved as `.jsonl` files in the defined `OUTPUT_PATH`.

## Evaluation

Once answers are successfully generated, you can evaluate the models across supported benchmarks (e.g., HumanEval, GSM8K, wmt19). The evaluation scripts are available in their respective directories within `mars/eval/`.

For example, to evaluate performance on HumanEval:

```bash
python -m mars.eval.humaneval.eval_humaneval --answer_file <path-to-your-generated-answers.jsonl>
```

Replace `<path-to-your-generated-answers.jsonl>` with the actual JSONL file generated during the inference phase.

## Acknowledgements

This research and codebase are built upon the excellent foundation of **EAGLE3**. We sincerely thank the authors of EAGLE3 for their pioneering work and their open-source contributions which made MARS possible.

## Reference

If you find this repository useful in your research, please consider citing our work:

```bibtex
@misc{song2026mars,
      title={MARS: Unleashing the Power of Speculative Decoding via Margin-Aware Verification}, 
      author={Jingwei Song and Xinyu Wang and Hanbin Wang and Xiaoxuan Lei and Bill Shi and Shixin Han and Eric Yang and Xiao-Wen Chang and Lynn Ai},
      year={2026},
      eprint={2601.15498},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.15498}, 
}
```
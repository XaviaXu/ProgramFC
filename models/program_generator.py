import argparse
import os
import json
import logging
from tqdm import tqdm

import re
from prompts import Prompt_Loader
from utils import OpenAIModel

from llama_cpp import Llama

from typing import List


class Reasoning_Program_Generator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        # self.model_name = args.model_name
        self.save_path = args.save_path
        self.num_programs_per_example = args.num_programs_per_example
        self.parse_type = args.parse_type

        # self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        self.prompt_loader = Prompt_Loader(args.parse_type,args.dataset_name)

        self.max_seq_len = args.max_seq_len
        self.max_batch_size = args.max_batch_size
        self.model = Llama(
            model_path="/xinyuxu/llama.cpp/models/llama-2-70b/ggml-model-f16.gguf",
            #max_seq_len=self.max_seq_len,
            n_batch = self.max_batch_size,
            n_ctx = self.max_seq_len,
            n_gpu_layers=-1
        )
        self.num_hops = args.num_hops

    def update_results(self, sample, generated_text):
        #program_list = [operation.strip() for operation in generated_text.split('\n')]
        # programs = [program_list]

        print(generated_text)
        match=re.search(r'([\s\S]*?label\s*=\s*Predict\(.*?\))',generated_text['choices'][0]['text'])
        text = match.group(1)
        program_list = [operation.strip() for operation in text.split('\n')][1:]

        #program_list = generated_text['choices'][0]['text']
        #print(generated_text)
        self.result_dict[sample['id']]['predicted_programs'].append(program_list)

    def batch_generate_programs(self, batch_size=10):
        # create output_dir
        self.result_dict = []
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # load dataset
        with open(os.path.join(self.data_path, self.dataset_name, 'claims', 'dev.json'), 'r') as f:
            raw_dataset = json.load(f)
        
        if self.dataset_name=='HOVER':
            raw_dataset = [d for d in raw_dataset if d['num_hops']==self.num_hops]

        raw_dataset = raw_dataset if self.args.num_eval_samples < 0 else raw_dataset[:self.args.num_eval_samples]
        print(f"Loaded {len(raw_dataset)} examples from {self.dataset_name} dev set.")

        # generate programs
        temperature = 0.8
        top_p = 0.95
        max_gen_len = int(-1)


        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]

        # initialize empty results
        result_dict = {}
        for idx, sample in enumerate(raw_dataset):
            result = {'idx': idx,
                      'id': sample['id'],
                      'claim': sample['claim'],
                      'gold': sample['label'],
                      'predicted_programs': []}
            result_dict[sample['id']] = result
        self.result_dict = result_dict

        # for each iteration
        for iteration in range(self.num_programs_per_example):
            print(f"Generating programs for iteration {iteration + 1}...")
            # for each chunk
            for chunk in tqdm(dataset_chunks):
                # create prompt
                full_prompts = [self.prompt_loader.prompt_construction(example['claim']) for example
                                in chunk]
                for sample, full_prompt in zip(chunk, full_prompts):
                    try:
                        output = self.model.create_completion(
                            prompt=full_prompt,
                            max_tokens=max_gen_len,
                            temperature=temperature,
                            top_p=top_p,
                         )
                        self.update_results(sample, output)
                    except Exception as e:
                        logging.exception(e)
                        print('Error in generating reasoning programs for example: ', sample['id'])

        print(f"Generated {len(result_dict)} examples.")
        # create outputs
        for key in result_dict:
            outputs.append(result_dict[key])
        sorted_outputs = sorted(outputs, key=lambda x: x['idx'])

        # save outputs
        with open(os.path.join(self.save_path,
                               f'{self.parse_type}_{self.dataset_name}_N={self.num_programs_per_example}_Hops={self.num_hops}_programs.json'),
                  'w') as f:
            json.dump(sorted_outputs, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset_name', default='HOVER', type=str)
    parser.add_argument('--data_path',default='/xinyuxu/ProgramFC/datasets', type=str)
    parser.add_argument('--num_eval_samples', default=-1, type=int)
    parser.add_argument('--num_programs_per_example', default=1, type=int)
    parser.add_argument('--save_path', default='/xinyuxu/ProgramFC/results/programs', type=str)
    parser.add_argument('--stop_words', type=str, default='# The claim is')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--parse_type',type=str,default="CONSTITUENCY")

    #parser.add_argument('--ckpt_dir',type=str,default='/xinyuxu/llama/llama-2-13b/')
    parser.add_argument('--tokenizer_path',type=str,default='/xinyuxu/llama/tokenizer.model')
    parser.add_argument('--max_seq_len',type=int,default=8192)
    parser.add_argument('--max_batch_size',type=int,default=12)
    parser.add_argument('--num_hops',type=int,default=2)

    args = parser.parse_args()
    return args

def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_gen_len: int = 64,
        max_batch_size: int = 6,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,

        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    args = parse_args()
    generator = Reasoning_Program_Generator(args)
    print("initialized")
    generator.batch_generate_programs()
    # fire.Fire(main)

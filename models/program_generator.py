import argparse
import os
import json
import logging
from tqdm import tqdm

import re
from prompts import Prompt_Loader
from transformers import AutoModelForCausalLM,AutoTokenizer

from llama_cpp import Llama

from typing import List

model_id = "mistralai/Mixtral-8x7B-v0.1"

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.num_hops = args.num_hops

    def update_results(self, sample, generated_text):
        #program_list = [operation.strip() for operation in generated_text.split('\n')]
        # programs = [program_list]

        print(generated_text)
        match=re.search(r'([\s\S]*?label\s*=\s*Predict\(.*?\))',generated_text)
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
        max_gen_len = int(128)


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
                #print(full_prompts)
                for sample, full_prompt in zip(chunk, full_prompts):
                    try:
                        inputs = self.tokenizer(full_prompt,return_tensors="pt")
                        outputs = self.model.generate(**inputs,max_new_tokens=max_gen_len)
                        self.update_results(sample, self.tokenizer.decode(outputs[0], skip_special_tokens=True))
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
                               f'Mixtral_{self.parse_type}_{self.dataset_name}_N={self.num_programs_per_example}_Hops={self.num_hops}_programs.json'),
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
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--parse_type',type=str,default="CONSTITUENCY")

    #parser.add_argument('--ckpt_dir',type=str,default='/xinyuxu/llama/llama-2-13b/')
    parser.add_argument('--tokenizer_path',type=str,default='/xinyuxu/llama/tokenizer.model')
    parser.add_argument('--max_seq_len',type=int,default=8192)
    parser.add_argument('--max_batch_size',type=int,default=512)
    parser.add_argument('--num_hops',type=int,default=2)

    args = parser.parse_args()
    return args


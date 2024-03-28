import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
from tqdm import tqdm
import re
import os
import json
from clocq.CLOCQ import CLOCQ
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient
from clocq.interface.CLOCQTaskHandler import CLOCQTaskHandler


from question_answering import T5_Question_Answering
from retriever import PyseriniRetriever
from evaluate import print_evaluation_results

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--FV_data_path', type=str)
    parser.add_argument('--setting', help='[gold | open-book | close-book]', type=str)
    parser.add_argument('--num_eval_samples', default=2000, type=int)
    parser.add_argument('--program_dir', type=str)
    parser.add_argument('--program_file_name', type=str)
    parser.add_argument('--output_dir', type=str)
    # fact checker args
    parser.add_argument("--model_name", default = 'google/flan-t5-xl', type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument('--corpus_index_path', default=None, type=str)
    parser.add_argument('--num_retrieved', default=5, type=int)
    parser.add_argument('--max_evidence_length', default=3000, help = 'to avoid exceeding GPU memory', type=int)
    args = parser.parse_args()
    return args

class Program_Linking:
    def __init__(self, args) -> None:
        # load model
        self.args = args
        self.clocq = CLOCQInterfaceClient(port="7778")
       
    def map_direct_answer_to_label(self, predict):
        predict = predict.lower().strip()
        label_map = {'true': True, 'false': False, 'yes': True, 'no': False, "it's impossible to say": False}
        if predict in label_map:
            return label_map[predict]
        else:
            print(f"Alert!!! wrong answer mapping: {predict}")
            return random.sample([True, False], 1)[0]

    def parse_verify_command(self, command, variable_map):
        return_var, tmp = command.split('= Verify')
        return_var = return_var.strip()
        # claim = tmp.replace("\")", "").strip()

        p1 = re.compile(f'Verify\([f]?\"(.*)\"\)', re.S)
        matching = re.findall(p1, command)
        claim = matching[0] if len(matching)>0 else tmp

        # replace variable
        for variable_name, variable_value in variable_map.items():
            replace_var = "{" + str(variable_name) + "}"
            if claim.find(replace_var) >=0:
                claim = claim.replace(replace_var, variable_value)

        return return_var, claim

    def parse_question_command(self, command, variable_map):
        return_var, tmp = command.split('= Question')
        return_var = return_var.strip()
        # question = tmp.replace("\")", "").strip()

        p1 = re.compile(f'Question\([f]?\"(.*)\"\)', re.S)
        matching = re.findall(p1, command)
        question = matching[0] if len(matching)>0 else tmp

        # replace variable
        for variable_name, variable_value in variable_map.items():
            replace_var = "{" + str(variable_name) + "}"
            if question.find(replace_var) >=0:
                question = question.replace(replace_var, variable_value)

        return return_var, question

    def get_command_type(self, command):
        if command.find("label = ")>=0:
            return "FINAL"
        elif command.find('= Verify')>=0:
            return "VERIFY"
        elif command.find('= Question')>=0:
            return "QUESTION"
        else:
            return "UNKNOWN"

    def derive_final_answer(self, command, variable_map):
        final_label = True
        command = command.replace('label =', '').strip()
        p1 = re.compile(r'Predict[(](.*?)[)]', re.S)
        command_arg = re.findall(p1, command)[0]
        verify_subs = command_arg.split(" and ")
        arguments = [arg.strip() for arg in verify_subs]
        for argument in arguments:
            if argument in variable_map:
                final_label = variable_map[argument] and final_label
            else:
                print(f"Alert!!! wrong argument: {argument}")
        return final_label
    
    def parse_program(self, ID, program, evidence):
        #print(program)
        variable_map = {}
        # for each command
        for command in program:
            c_type = self.get_command_type(command)
            final_answer = None
            # verify a claim
            if c_type == "VERIFY":
                return_var, claim = self.parse_verify_command(command, variable_map)
                link_dict = self.clocq.entity_linking(claim)
                if len(link_dict)==0:
                    return 1
            elif c_type == "QUESTION":
                return_var, question = self.parse_question_command(command, variable_map)
                link_dict = self.clocq.entity_linking(question)
                if len(link_dict)==0:
                    return 1
              
            elif c_type == 'FINAL':
                    return 0
        
        return 0

    def execute_on_dataset(self):
        # load generated program
        with open(os.path.join(self.args.program_dir, self.args.program_file_name), 'r') as f:
            dataset = json.load(f)
        

        gt_labels, predictions = [], []
        results = []
        negative_linking = 0
        for sample in tqdm(dataset):
            program = sample['predicted_programs']
            #program = [s.replace("\\","") for s in program]
            #print(program)
            gt_labels.append(sample['gold'])

            # get evidence
            evidence = self.gold_evidence_map[sample['id']] if self.args.setting == 'gold' else None
            
            
            for sample_program in program:
                try:
                    negative_linking += self.parse_program(sample['id'], [s.replace("\\","") for s in sample_program], evidence)
                except Exception as e:
                    print(f"Alert!!! execution error: {sample['id']}")
                    negative_linking +=1
                
            print(f"Linking failed:{negative_linking}, program count:{len(dataset)},rate:{1-negative_linking/len(dataset)}")
            



if __name__ == "__main__":
    args = parse_args()
    program_executors = Program_Linking(args)
    program_executor.execute_on_dataset()

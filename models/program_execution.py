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

params = {"h_match": 0.4,
                  "h_rel": 0.2,
                  "h_conn": 0.3,
                  "h_coh": 0.1,
                  "d": 20,
                  "k": 5,
                  "p_setting": 1000,  # setting for search_space function
                  "bm25_limit": False}

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
    parser.add_argument("--model_name", default='google/flan-t5-xl', type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument('--corpus_index_path', default=None, type=str)
    parser.add_argument('--num_retrieved', default=5, type=int)
    parser.add_argument('--max_evidence_length', default=3000, help='to avoid exceeding GPU memory', type=int)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--dump_path',type=str)
    args = parser.parse_args()
    return args


class Program_Execution:
    def __init__(self, args) -> None:
        # load model
        self.args = args
        CACHE_DIR = args.cache_dir
        self.model_name = args.model_name
        self.dataset_name = args.dataset_name
        print(f"Loading model {self.model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
        self.model.parallelize()
        print(f"Model {self.model_name} loaded.")
        self.clocq = CLOCQInterfaceClient(port="7778")

        self.QA_module = T5_Question_Answering(self.model, self.tokenizer)

        # load retriever
        if self.args.setting == 'open-book':
            self.searcher = PyseriniRetriever(self.args.corpus_index_path, use_bm25=True, k1=0.9, b=0.4)
        else:
            self.searcher = None

        # load dataset
        with open(os.path.join(args.FV_data_path, args.dataset_name, 'claims', f'dev.json'), 'r') as f:
            dataset = json.load(f)
        self.gold_evidence_map = {sample['id']: sample['evidence'] for sample in dataset}

        
        try:
            with open(args.dump_path, 'r') as w:
                self.dump = json.load(w)
                print(f"Dump file founded, successfully load {len(self.dump)} dump files.")
        except FileNotFoundError:
            print("Dump file not found.")
            self.dump = []

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
        claim = matching[0] if len(matching) > 0 else tmp

        # replace variable
        for variable_name, variable_value in variable_map.items():
            replace_var = "{" + str(variable_name) + "}"
            if claim.find(replace_var) >= 0:
                claim = claim.replace(replace_var, variable_value)

        return return_var, claim

    def parse_question_command(self, command, variable_map):
        return_var, tmp = command.split('= Question')
        return_var = return_var.strip()
        # question = tmp.replace("\")", "").strip()

        p1 = re.compile(f'Question\([f]?\"(.*)\"\)', re.S)
        matching = re.findall(p1, command)
        question = matching[0] if len(matching) > 0 else tmp

        # replace variable
        for variable_name, variable_value in variable_map.items():
            replace_var = "{" + str(variable_name) + "}"
            if question.find(replace_var) >= 0:
                question = question.replace(replace_var, variable_value)

        return return_var, question

    def get_command_type(self, command):
        if command.find("label = ") >= 0:
            return "FINAL"
        elif command.find('= Verify') >= 0:
            return "VERIFY"
        elif command.find('= Question') >= 0:
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

    def retrieve_evidence(self, query):
        hits = self.searcher.retrieve(query, self.args.num_retrieved)
        evidence = '\n'.join([hit['text'].strip() for hit in hits])
        # cut overlong evidence
        if len(evidence.split()) > self.args.max_evidence_length:
            print('evidence is too long, cut it to max_evidence_length')
            evidence = ' '.join(evidence.split()[:self.args.max_evidence_length])

        # save retrieval results (can comment out if not needed)
        retrieved_results = []
        for hit in hits:
            retrieved_results.append({'id': hit['doc_id'], 'score': hit['score'], 'query': query})

        return evidence, retrieved_results

    def parse_program(self, ID, program, evidence):
        # print(program)
        variable_map = {}
        claim_only = True if self.args.setting == 'close-book' else False
        retrieved_evidence = []
        # for each command
        for command in program:
            c_type = self.get_command_type(command)
            final_answer = None
            # verify a claim
            if c_type == "VERIFY":
                return_var, claim = self.parse_verify_command(command, variable_map)
                # if open-book setting, then retrieve evidence from the corpus
                if self.args.setting == 'open-book':
                    evidence, retrieved_results = self.retrieve_evidence(claim)
                    retrieved_evidence += retrieved_results
                evidence += f"a claim-relevant KG subset are provided:{self.kb_evidence(claim, self.args.top_k,ID)}"

                answer = self.QA_module.answer_verify_question(claim, evidence, claim_only)['answer_text']
                variable_map[return_var] = self.map_direct_answer_to_label(answer)
            # ask a question
            elif c_type == "QUESTION":
                return_var, question = self.parse_question_command(command, variable_map)
                # if open-book setting, then retrieve evidence from the corpus
                if self.args.setting == 'open-book':
                    evidence, retrieved_results = self.retrieve_evidence(question)
                    retrieved_evidence += retrieved_results
                evidence += f"a question-relevant KG subset are provided:{self.kb_evidence(question, self.args.top_k,ID)}"

                answer = self.QA_module.answer_question_directly(question, evidence, claim_only)['answer_text']
                variable_map[return_var] = answer
            elif c_type == 'FINAL':
                try:
                    final_answer = self.derive_final_answer(command, variable_map)
                except:
                    print(f"Alert!!! parsing error: {ID}")
                    final_answer = random.sample([True, False], 1)[0]
        
        return final_answer, retrieved_evidence

    def kb_evidence(self, claim, top_k,ID):
        if ID in self.dump and claim in self.dump[ID]:
            res = self.dump[ID][claim]
        else:
            res = self.clocq.get_search_space(question=claim, parameters=params)
            if ID not in self.dump:
                self.dump[ID] = {}
            self.dump[ID][claim] = res
        return self.search_space_filter(claim, res["search_space"], top_k)

    def search_space_filter(self, claim, rdfs, top_k):
        evidence = []
        processed_claim = claim.replace(' ', '').lower()
        for rdf in rdfs:
            iterator = iter(rdf)
            sub = next(iterator)['label']
            score = self.score_rdf(processed_claim, sub)
            try:
                while True:
                    pred = next(iterator)['label']
                    obj = next(iterator)['label']
                    total_score = (score + self.score_rdf(processed_claim, pred) + self.score_rdf(processed_claim,
                                                                                                  obj)) / 3
                    if total_score > 0.7:
                        evidence.append(dict(triple=f"<{sub},{pred},{obj}>", score=total_score))
            except:
                continue

        evidence = [dict(t) for t in {tuple(d.items()) for d in evidence}]
        evidence.sort(key=lambda x: (x['score']), reverse=True)
        return [item['triple'] for item in evidence][:top_k]

    def score_rdf(self, claim, word):
        word = word.replace(' ', '').lower()
        return self.longestCommonSubsequence(claim, word) / len(word)

    def longestCommonSubsequence(self, text1, text2):
        m, n = len(text1), len(text2)
        dp0, dp1 = [0 for _ in range(n + 1)], [0 for _ in range(n + 1)]
        for t1 in text1:
            for i, t2 in enumerate(text2):
                if t1 == t2:
                    dp1[i + 1] = dp0[i] + 1
                else:
                    dp1[i + 1] = max(dp0[i + 1], dp1[i])
            dp0, dp1 = dp1, dp0
        return dp0[-1]

    def execute_on_dataset(self):
        # load generated program
        with open(os.path.join(self.args.program_dir, self.args.program_file_name), 'r') as f:
            dataset = json.load(f)
        dataset = dataset if self.args.num_eval_samples < 0 else dataset[:self.args.num_eval_samples]

        gt_labels, predictions = [], []
        results = []
        for sample in tqdm(dataset):
            program = sample['predicted_programs']
            # program = [s.replace("\\","") for s in program]
            # print(program)
            gt_labels.append(sample['gold'])

            # get evidence
            evidence = self.gold_evidence_map[sample['id']] if self.args.setting == 'gold' else None

            # execute program
            sample_predictions = []
            for sample_program in program:
                try:
                    single_prediction, retrieved_evidence = self.parse_program(sample['id'],
                                                                               [s.replace("\\", "") for s in
                                                                                sample_program], evidence)
                except Exception as e:
                    print(f"Alert!!! execution error: {sample['id']}")
                    single_prediction = random.sample([True, False], 1)[0]
                sample_predictions.append(single_prediction)

            true_count = len([pred for pred in sample_predictions if pred == True])
            false_count = len([pred for pred in sample_predictions if pred == False])
            final_prediction = True if true_count > false_count else False
            predictions.append('supports' if final_prediction == True else 'refutes')
            results.append({'id': sample['id'],
                            'claim': sample['claim'],
                            'gold': sample['gold'],
                            'prediction': 'supports' if final_prediction == True else 'refutes'})

        # evaluate
        self.evaluation(predictions, gt_labels)

        # save results to file
        output_path = os.path.join(self.args.output_dir,
                                   '{}_{}'.format(self.model_name.split('/')[-1], self.args.setting))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_file_name = f'{self.args.program_file_name}.program.json'
        with open(os.path.join(output_path, output_file_name), 'w') as f:
            f.write(json.dumps(results, indent=2))
        with open(args.dump_path,'w')as f:
            f.write(json.dumps(self.dump,indent=2))

    def evaluation(self, predictions, gt_labels):
        print_evaluation_results(predictions, gt_labels, num_of_classes=2)


if __name__ == "__main__":
    args = parse_args()
    program_executor = Program_Execution(args)
    program_executor.execute_on_dataset()

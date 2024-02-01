import re

import amrlib
from stanfordcorenlp import StanfordCoreNLP
import stanza

HOVER_DECOMPOSE = ''''Split the given claim by extracting clauses, verbs and other miscellaneous phrases to break it into multiple short sentences, and generate a Python program based on the generated clauses to verify the claim step-by-step. You can call three functions in the program: 1. Question() to answer a question; 2. Verify() to verify a simple claim; 3. Predict() to predict the veracity label. Several examples are given as follows.

# The claim is that Sally Menke was influenced by a cinematographer that wrote M3, M4 and M5 and influenced by M1's editor.
# Subclause identification: Sally Menke was influenced by (a cinematographer that wrote M3, M4 and M5) and influenced by M1's editor.
# Verb and other miscellaneous phrases identification: Sally Menke (was influenced by (a cinematographer that (wrote M3, M4 and M5))) and (influenced by M1's editor).
# Decompose into short sentences: A cinematographer wrote M3, M4 and M5. Sally Menke was influenced by the cinematographer. Sally Menke was influenced by M1's editor.
# Generate program:
def program():
    answer_1 = Question("Which cinematographer wrote M3,M4 and M5?")
    fact_1 = Verify(f"Sally Menke was influenced by {answer_1}.")
    fact_2 = Verify("Sally Menke was influenced by M1's editor.")
    label = Predict(fact_1 and fact_2)

# The claim is that Along with the New York Islanders and the New York Rangers, the New Jersey Devils NFL franchise is popular in the New York metropolitan area.
# Subclause identification: Along with (the New York Islanders and the New York Rangers), (the New Jersey Devils NFL franchise) is popular in the New York metropolitan area.
# Verb and other miscellaneous phrases identification: (Along with (the New York Islanders and the New York Rangers)), (the New Jersey Devils NFL franchise) (is popular in the New York metropolitan area).
# Decompose into short sentences: The New York Islanders is popular in the New York metropolitan area. The New York Rangers is popular in the New York metropolitan area. The New Jersey Devils NFL franchise is popular in the New York metropolitan area.
# Generate program: 
def program():
    fact_1 = Verify("The New York Islanders is popular in the New York metropolita n area.")
    fact_2 = Verify("The New York Rangers is popular in the New York metropolitan area.")
    fact_3 = Verify("The New Jersey Devils NFL franchise is popular in the New York metropolitan area.")
    label = Predict(fact_1 and fact_2 and fact_3)


# The claim is that Talking Heads, an American rock band that was "one of the most critically acclaimed bands of the 80's" is featured in KSPN's AAA format.
# Subclause identification: Talking Heads, an American rock band that (was "one of the most critically acclaimed bands of the 80's") is featured in KSPN's AAA format.
# Verb and other miscellaneous phrases identification: (Talking Heads), an American rock band that (was "one of the most critically acclaimed bands of the 80's") is featured in KSPN's AAA format.
# Decompose into short sentences: Talking Heads is an American rock band. Talking Heads was 'one of the most critically acclaimed bands of the 80's'. Talking Heads is featured in KSPN's AAA format.
# Generate program:
def program():
    fact_1 = Verify("Talking Heads is an American rock band")
    fact_2 = Verify("Talking Heads was 'one of the most critically acclaimed bands of the 80's'.")
    fact_3 = Verify("Talking Heads is featured in KSPN's AAA format.")
    label = Predict(fact_1 and fact_2 and fact_3)

# The claim is that The song recorded by Fergie that was produced by Polow da Don and was followed by Life Goes On was M.I.L.F.$.
# Subclause identification: (The song recorded by Fergie) (that was produced by Polow da Don) (and was followed by Life Goes On) (was M.I.L.F.$).
# Verb and other miscellaneous phrases identification: (The song recorded by Fergie) (that was produced by Polow da Don) (and was followed by Life Goes On) (was M.I.L.F.$).
# Decompose into short sentences: Fergie recorded M.I.L.F.$. M.I.L.F.$ was produced by Polow da Don. M.I.L.F.$ was followed by Life Goes On. 
# Generated program:
def program():
    fact_1 = Verify("M.I.L.F.$ was recorded by Fergie")
    fact_2 = Verify("M.I.L.F.$ was produced by Polow da Don.")
    fact_2 = Verify("M.I.L.F.$ was was followed by Life Goes On.")
    label = Predict(fact_1 and fact_2 and fact_3)

# The claim is that Eatza Pizza and Your Pie were not founded in the same state.
# Subclause identification: Eatza Pizza and Your Pie were not founded in the same state.
# Verb and other miscellaneous phrases identification: Eatza Pizza and Your Pie (were not founded in the same state).
# Decompose into short sentences: Eatza Pizza was founded in state A. Your Pie was founded in state B. A and B are not the same state.
# Generated program:
def program():
    answer_1 = Question("Which state was Eatza Pizza founded in?")
    answer_2 = Question("Which state was Your Pie founded in?")
    fact_1 = Verify(f"{answer_1} and {answer_2} are not the same state.")
    label = Predict(fact_1)

# The claim is that [[CLAIM]]
# Subclause identification:'''

HOVER_PROGRAM_FC = ''''Generate a python-like program that describes the reasoning steps required to verify the claim step-by-step. You can call three functions in the program: 1. Question() to answer a question; 2. Verify() to verify a simple claim; 3. Predict() to predict the veracity label. Several examples are given as follows.

# The claim is that WWE Super Tuesday took place at an arena that currently goes by the name TD Garden.
def program():
    answer_1 = Question("Which arena the WWE Super Tuesday took place?")
    fact_1 = Verify(f"{answer_1} currently goes by the name TD Garden.")
    label = Predict(fact_1)

# The claim is that Jack McFarland is the best known role of the host of the 64th Annual Tony Awards.
def program():
    answer_1 = Question("Who is the host of the 64th Annual Tony Awards?")
    fact_1 = Verify(f\"Jack McFarland is the best known role of {answer_1}.)
    label = Predict(fact_1)

# The claim is that Blackstar is the name of the album released by David Bowie that was recorded in secret.
def program():
    fact_1 = Verify("David Bowie released an album called Blackstar.")
    fact_2 = Verify("David Bowie recorded an album in secret.")
    label = Predict(fact_1 and fact_2)

# The claim is that Maria Esther Andion Bueno, not Jimmy Connors, is the player that is from Brazil.
def program():
    fact_1 = Verify("Maria Esther Andion Bueno is from Brazil.")
    fact_2 = Verify("Jimmy Connors is not from Brazil.")
    label = Predict(fact_1 and fact_2)
    
# The claim is that The song recorded by Fergie that was produced by Polow da Don and was followed by Life Goes On was M.I.L.F.$.
def program():
    fact_1 = Verify("M.I.L.F.$ was recorded by Fergie that was produced by Polow da Don.")
    fact_2 = Verify("M.I.L.F.$ was was followed by Life Goes On.")
    label = Predict(fact_1 and fact_2)

# The claim is that Talking Heads, an American rock band that was "one of the most critically acclaimed bands of the 80's" is featured in KSPN's AAA format.
def program():
    fact_1 = Verify("Talking Heads is an American rock band that was 'one of the most critically acclaimed bands of the 80's'.")
    fact_2 = Verify("Talking Heads is featured in KSPN's AAA format.")
    label = Predict(fact_1 and fact_2)

# The claim is that An IndyCar race driver drove a Formula 1 car designed by Peter McCool during the 2007 Formula One season.
def program():
    answer_1 = Question("Which Formula 1 car was designed by Peter McCool during the 2007 Formula One season?")
    fact_1 = Verify(f"An IndyCar race driver drove the car {answer_1}.")
    label = Predict(fact_1)
    
# The claim is that Don Ashley Turlington graduated from Saint Joseph's College, a private Catholic liberal arts college in Standish.
def program():
    fact_1 = Verify("Saint Joseph's College is a private Catholic liberal arts college is located in Standish.")
    fact_2 = Verify(f"Don Ashley Turlington graduated from Saint Joseph's College.")
    label = Predict(fact_1 and fact_2)
    
# The claim is that Gael and Fitness are not published in the same country.
def program():
    answer_1 = Question("Which country was Gael published in?")
    answer_2 = Question("Which country was Fitness published in?")
    fact_1 = Verify(f"{answer_1} and {answer_2} are not the same country.")
    label = Predict(fact_1)

# The claim is that Gina Bramhill was born in a village. The 2011 population of the area that includes this village was 167,446.
def program():
    answer_1 = Question("Which village was Gina Bramhill born in?")
    fact_1 = Verify(f"The 2011 population of the area that includes {answer_1} was 167,446.")
    label = Predict(fact_1)
    
# The claim is that In the 2004 Hockey film produced by a former major league baseball pitcher Kurt Russell played the USA coach.
def program():
    answer_1 = Question("Which 2004 Hockey film was produced a former major league baseball pitcher?")
    fact_1 = Verify("Kurt Russell played the USA coach in the film {answer_1}.")
    label = Predict(fact_1)
    
# The claim is that Along with the New York Islanders and the New York Rangers, the New Jersey Devils NFL franchise is popular in the New York metropolitan area.
def program():
    fact_1 = Verify("The New York Islanders and the New York Rangers are popular in the New York metropolitan area.")
    fact_2 = Verify("The New Jersey Devils NFL franchise is popular in the New York metropolitan area.")
    label = Predict(fact_1 and fact_2)
    
# The claim is that Thomas Loren Friedman has won more Pulitzer Prizes than Colson Whitehead.
def program():
    answer_1 = Question("How many Pulitzer Prizes has Thomas Loren Friedman won?")
    answer_2 = Question("How many Pulitzer Prizes has Colson Whitehead won?")
    fact_1 = Verify(f"{answer_1} is more than {answer_2}.")
    label = Predict(fact_1)
    
# The claim is that The model of car Trevor Bayne drives was introduced for model year 2006. The Rookie of The Year in the 1997 CART season drives it in the NASCAR Sprint Cup.
def program():
    answer_1 = Question("Which model of car is drived by Trevor Bayne?")
    fact_1 = Verify(f"{answer_1} was introduced for model year 2006.")
    answer_2 = Question("Who is the Rookie of The Year in the 1997 CART season?")
    fact_2 = Verify(f"{answer_2} drives the model of car Trevor Bayne drives in the NASCAR Sprint Cup.")
    label = predict(fact_1 and fact_2)
    
# The claim is that Barton Mine was halted by a natural disaster not Camlaren Mine.
def program():
    fact_1 = Verify("Barton Mine was halted by a natural disaster.")
    fact_2 = Verify("Camlaren Mine was not halted by a natural disaster.")
    label = Predict(fact_1 and fact_2)

# The claim is that Howard University Hospital and Providence Hospital are both located in Washington, D.C.
def program():
    fact_1 = Verify("Howard University Hospital is located in Washington, D.C.")
    fact_2 = Verify("Providence Hospital is located in Washington, D.C.")
    label = Predict(fact_1 and fact_2)
    
# The claim is that Vladimir Igorevich Arnold died after Georg Cantor.
def program():
    answer_1 = Question("When did Vladimir Igorevich Arnold die?")
    answer_2 = Question("When did Georg Cantor die?")
    fact_1 = Verify(f"{answer_1} is after {answer_2}.")
    label = Predict(fact_1)
    
# The claim is that Eatza Pizza and Your Pie were not founded in the same state.
def program():
    answer_1 = Question("Which state was Eatza Pizza founded in?")
    answer_2 = Question("Which state was Your Pie founded in?")
    fact_1 = Verify(f"{answer_1} and {answer_2} are not the same state.")
    label = Predict(fact_1)

# The claim is that Gregg Rolie and Rob Tyner, are not a keyboardist.
def program():
    fact_1 = Verify("Gregg Rolie is not a keyboardist.")
    fact_2 = Verify("Rob Tyner is not a keyboardist.")
    label = Predict(fact_1 and fact_2)
    
# The claim is that John O'Hara and Rabindranath Tagore are not the same nationality.
def program():
    answer_1 = Question("What is the nationality of John O'Hara?")
    answer_2 = Question("What is the nationality of Rabindranath Tagore?")
    fact_1 = Verify(f"{answer_1} and {answer_2} are not the same nationality.")
    label = Predict(fact_1)
    
# The claim is that [[CLAIM]]
def program():'''

DELETED_HOVER = '''

'''

FEVEROUS_PROGRAM_FC = '''Generate a python-like program that describes the reasoning steps required to verify the claim step-by-step. You can call three functions in the program: 1. Question() to answer a question; 2. Verify() to verify a simple claim; 3. Predict() to predict the veracity label. Several examples are given as follows.

# The claim is that In 1959, former Chilean boxer Alfredo Cornejo Cuevas (born June 6, 1933) won the gold medal in the welterweight division at the Pan American Games (held in Chicago, United States, from August 27 to September 7) in Chicago, United States, and the world amateur welterweight title in Mexico City.
def program():
    fact_1 = Verify("Alfredo Cornejo Cuevas was born in June 6, 1933.")
    fact_2 = Verify("Alfredo Cornejo Cuevas won the gold medal in the welterweight division at the Pan American Games in 1959.")
    fact_3 = Verify("The Pan American Games in 1959 was held in Chicago, United States, from August 27 to September 7.")
    fact_4 = Verify("Alfredo Cornejo Cuevas won the world amateur welterweight title in Mexico City.")
    label = Predict(fact_1 and fact_2 and fact_3 and fact_4)

# The claim is that The Footwork FA12, which was intended to start the season, finally debuted at the San Marino Grand Prix, a Formula One motor race held at Imola on 28 April 1991.
def program():
    fact_1 = Verify("The Footwork FA12, which was intended to start the season.")
    fact_2 = Verify("The Footwork FA12 finally debuted at the San Marino Grand Prix.")
    fact_3 = Verify("The San Marino Grand Prix was a Formula One motor race held at Imola on 28 April 1991.")
    label = Predict(fact_1 and fact_2 and fact_3)

# The claim is that SkyHigh Mount Dandenong (formerly Mount Dandenong Observatory) is a restaurant located on top of Mount Dandenong, Victoria, Australia.
def program():
    fact_1 = Verify("SkyHigh Mount Dandenong is a restaurant located on top of Mount Dandenong, Victoria, Australia.")
    fact_2 = Verify("SkyHigh Mount Dandenong is formerly known as Mount Dandenong Observatory.")
    label = Predict(fact_1 and fact_2)

# The claim is that Before the first Europeans arrived or copra companies leased it, Maupihaa was home to Inca's in ancient times.
def program():
    fact_1 = Verify("Maupihaa was home to Inca's in ancient times.")
    fact_2 = Verify("Maupihaa was home to Inca's before the first Europeans arrived or copra companies leased it.")
    label = Predict(fact_1 and fact_2)
    
# The claim is that Shulin, a 33.1288 km (12.7911 sq mi) land located in New Taipei City, China, a country in East Asia, has a total population of 183,946 in December 2018.
def program():
    fact_1 = Verify("Shulin is a 33.1288 km (12.7911 sq mi) land located in New Taipei City, China.")
    fact_2 = Verify("Shulin has a total population of 183,946 in December 2018.")
    label = Predict(fact_1 and fact_2)
    
# The claim is that Sumo wrestler Toyozakura Toshiaki committed match-fixing, ending his career in 2011 that started in 1989.
def program():
    fact_1 = Verify("Toyozakura Toshiaki ended his career in 2011 that started in 1989.")
    fact_2 = Verify("Toyozakura Toshiaki is a Sumo wrestler.")
    fact_3 = Verify("Toyozakura Toshiaki committed match-fixing.")
    label = Predict(fact_1 and fact_2 and fact_3)

# The claim is that In 1959, former Chilean boxer Alfredo Cornejo Cuevas (born June 6, 1933) won the gold medal in the welterweight division at the Pan American Games (held in Chicago, United States, from August 27 to September 7) in Chicago, United States, and the world amateur welterweight title in Mexico City.
def program():
    fact_1 = Verify("Alfredo Cornejo Cuevas is a former Chilean boxer.")
    fact_2 = Verify("Alfredo Cornejo won the gold medal in the welterweight division at the Pan American Games.")
    fact_3 = Verify("The Pan American Games was held in Chicago, United States, from August 27 to September 7.")
    fact_4 = Verify("Alfredo Cornejo won the world amateur welterweight title in Mexico City.")
    label = Predict(fact_1 and fact_2 and fact_3 and fact_4)

# The claim is that Adductor hiatus is associated with nine structures, seven of which enter and leave through hiatus.
def program():
    fact_1 = Verify("Adductor hiatus is associated with nine structures.")
    fact_2 = Verify("Seven of the nine structures associated with Adductor hiatus enter and leave through hiatus.")
    label = Predict(fact_1 and fact_2)
    
# The claim is that Ifor Bowen Lloyd was educated at Winchester (an independent boarding school for boys in the British public school tradition) and Exeter College, Oxford where he was a member of the Library Committee of the Oxford Union Society, as well as, received a BA in Modern History in 1924.
def program():
    fact_1 = Verify("Ifor Bowen Lloyd was educated at Winchester and Exeter College, Oxford.")
    fact_2 = Verify("Winchester is an independent boarding school for boys in the British public school tradition.")
    fact_3 = Verify("While at Oxford, Ifor Bowen Lloyd was a member of the Library Committee of the Oxford Union Society.")
    fact_4 = Verify("Ifor Bowen Lloyd received a BA in Modern History in 1924 at Oxford.")
    label = Predict(fact_1 and fact_2 and fact_3 and fact_4)
  
# The claim is that In the 2001 Stanley Cup playoffs Eastern Conference Semifinals Devils' Elias scored and Maple Leafs' left Devils player Scott Neidermayer hurt.
def program():
    fact_1 = Verify("In the 2001 Stanley Cup playoffs Eastern Conference Semifinals Devils' Elias scored.")
    fact_2 = Verify("Maple Leafs' left Devils player Scott Neidermayer hurt.")
    label = Predict(fact_1 and fact_2)
    
# The claim is that Teldenia helena is a moth first described in 1967 by Wilkinson.
def program():
    fact_1 = Verify("Teldenia helena is a moth.")
    fact_2 = Verify("Teldenia helena was first described by Wilkinson in 1967.")
    label = Predict(fact_1 and fact_2)
    
# The claim is that Born December 30, 1974, William Frick was a dark horse candidate in the Maryland House of Delegates appointment process.
def program():
    fact_1 = Verify("William Frick was born in December 30, 1974.")
    fact_2 = Verify("William Frick was a dark horse candidate in the Maryland House of Delegates appointment process.")
    label = Predict(fact_1 and fact_2)

# The claim is that [[CLAIM]]
def program():'''

DEPENDENCY_TEMPLATE = '''
# The claim is that {claim}
Dependency parsing of the claim:
{parsing}def program():{program}'''

CONSTITUENCY_TEMPLATE = '''
# The claim is that {claim}
Constituency parsing of the claim:
{parsing}def program():{program}'''

AMR_TEMPLATE = '''
# The claim is that {claim}
Abstract meaning representation of the claim:
{parsing}
def program():{program}'''

NULL_TEMPLATE ='''
# The claim is that {claim}
{parsing}
def program():
{program}'''
template_map = {"DEPENDENCY": DEPENDENCY_TEMPLATE, "CONSTITUENCY": CONSTITUENCY_TEMPLATE, "AMR": AMR_TEMPLATE, "NULL":NULL_TEMPLATE}


def mixtral_format(sys_message,query):
    return f'<s> [INST] {sys_message} [/INST]\n{query}'
class Prompt_Loader:
    def __init__(self, parse_type, dataset_name) -> None:
        self.hover_program_fc = HOVER_PROGRAM_FC
        self.feverous_program_fc = FEVEROUS_PROGRAM_FC
        self.nlp = StanfordCoreNLP(r'/xinyuxu/stanford-corenlp-4.5.5')
        # self.nlp = StanfordCoreNLP(r'E:/stanford-corenlp-4.5.5')
        self.prompt = ""
        self.template = template_map[parse_type]
        self.parse_type = parse_type
        self.tokenizer = stanza.Pipeline(lang='en', processors='tokenize')
        self.amr = amrlib.load_stog_model()

        self.prompt_init(dataset_name)

    def parsing(self, claim):
        doc = self.tokenizer(claim)
        analysis = ""
        if self.parse_type == "DEPENDENCY":
            for sentence in doc.sentences:
                analysis += f"{self.nlp.dependency_parse(sentence.text)}\n"
        elif self.parse_type == "CONSTITUENCY":
            for sentence in doc.sentences:
                result = self.nlp.parse(sentence.text).replace(' ' * 2, '\t')
                analysis += f"{result}\n"
        elif self.parse_type == "AMR":
            graphs = self.amr.parse_sents([sentence.text for sentence in doc.sentences])
            for graph in graphs:
                amr_lines = graph.replace(' ' * 6, '\t').split('\n')
                amr_structure = '\n'.join(
                    [re.sub(r'(?<!\t) ', '', line) for line in amr_lines if not line.startswith('# ::snt')])
                analysis += amr_structure
            # print(analysis)
        return analysis

    def prompt_init(self, dataset_name):
        template = None
        if dataset_name == 'HOVER':
            template = self.hover_program_fc
        elif dataset_name == 'FEVEROUS':
            template = self.feverous_program_fc
        else:
            raise NotImplementedError

        self.prompt = "Generate a python-like program that describes the reasoning steps required to verify the claim step-by-step. " \
                      "The parsing tree of the claim is provided to assist in breaking down the claim and generating programs. " \
                      "You can and only can call three functions in the program: 1. Question() to answer a question; 2. Verify() to verify a simple claim; 3. Predict() to predict the veracity label. " \
                      "The only logic expression you can use is 'and'. 'not' and all relational operators are strictly prohibited. " \
                      "You are not allowed to have any loop structure in your generated program. " \
                      "Several examples are given as follows. You should strictly follow the template that shown in the examples.\n"
        claims = re.findall(r'# The claim is that(.*?)\ndef program\(\):(.*?)\n\s*\n', template, re.DOTALL)
        for claim, program_code in claims:
            parse_tree = self.parsing(claim)
            self.prompt += self.template.format(claim=claim.strip(), parsing=parse_tree, program=program_code + "\n")
        self.prompt += "Let's get started. The claim and parsing information is given as follows."
        print("prompt initialized. Total length of prompt:"+str(len(self.prompt)))

    def prompt_construction(self, claim):
        parse_tree = self.parsing(claim)
        temp = self.template.format(claim=claim.strip(), parsing=parse_tree, program="")
        return mixtral_format(self.prompt, temp)


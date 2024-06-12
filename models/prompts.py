import json
import hanlp
from phrasetree.tree import Tree
import re

HOVER_PROGRAM_FC = ''''Generate a python-like program that describes the reasoning steps required to verify the claim step-by-step. You can call three functions in the program: 1. Question() to answer a question; 2. Verify() to verify a simple claim; 3. Predict() to predict the veracity label. Several examples are given as follows.

# The claim is that Howard University Hospital and Providence Hospital are both located in Washington, D.C.
def program():
    fact_1 = Verify("Howard University Hospital is located in Washington, D.C.")
    fact_2 = Verify("Providence Hospital is located in Washington, D.C.")
    label = Predict(fact_1 and fact_2)

# The claim is that WWE Super Tuesday took place at an arena that currently goes by the name TD Garden.
def program():
    answer_1 = Question("Which arena the WWE Super Tuesday took place?")
    fact_1 = Verify(f"{answer_1} currently goes by the name TD Garden.")
    label = Predict(fact_1)

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

# The claim is that Gina Bramhill was born in a village. The 2011 population of the area that includes this village was 167,446.
def program():
    answer_1 = Question("Which village was Gina Bramhill born in?")
    fact_1 = Verify(f"The 2011 population of the area that includes {answer_1} was 167,446.")
    label = Predict(fact_1)

# The claim is that Don Ashley Turlington graduated from Saint Joseph's College, a private Catholic liberal arts college in Standish.
def program():
    fact_1 = Verify("Saint Joseph's College is a private Catholic liberal arts college in Standish.")
    fact_2 = Verify(f"Don Ashley Turlington graduated from Saint Joseph's College.")
    label = Predict(fact_1 and fact_2)

# The claim is that Gael and Fitness are not published in the same country.
def program():
    answer_1 = Question("Which country was Gael published in?")
    answer_2 = Question("Which country was Fitness published in?")
    fact_1 = Verify(f"{answer_1} and {answer_2} are not the same country.")
    label = Predict(fact_1)

# The claim is that Blackstar is the name of the album released by David Bowie that was recorded in secret.
def program():
    fact_1 = Verify("David Bowie released an album called Blackstar.")
    fact_2 = Verify("The album Blackstar is recorded in secret.")
    label = Predict(fact_1 and fact_2)

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

# The claim is that Jack McFarland is the best known role of the host of the 64th Annual Tony Awards.
def program():
    answer_1 = Question("Who are the hosts of the 64th Annual Tony Awards?")
    fact_1 = Verify(f\"Jack McFarland is the best known role of {answer_1}.)
    label = Predict(fact_1)

# The claim is that The song recorded by Fergie that was produced by Polow da Don and was followed by Life Goes On was M.I.L.F.$.
def program():
    fact_1 = Verify("M.I.L.F.$ was recorded by Fergie.")
    fact_2 = Verify("M.I.L.F.$ was produced by Polow da Don.")
    fact_3 = Verify("M.I.L.F.$ was was followed by Life Goes On.")
    label = Predict(fact_1 and fact_2 and fact_3)

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

# The claim is that Maria Esther Andion Bueno, not Jimmy Connors, is the player that is from Brazil.
def program():
    fact_1 = Verify("Maria Esther Andion Bueno is from Brazil.")
    fact_2 = Verify("Jimmy Connors is not from Brazil.")
    label = Predict(fact_1 and fact_2)

# The claim is that Vladimir Igorevich Arnold died after Georg Cantor.
def program():
    answer_1 = Question("When did Vladimir Igorevich Arnold die?")
    answer_2 = Question("When did Georg Cantor die?")
    fact_1 = Verify(f"{answer_1} is after {answer_2}.")
    label = Predict(fact_1)

# The claim is that Barton Mine was halted by a natural disaster not Camlaren Mine.
def program():
    fact_1 = Verify("Barton Mine was halted by a natural disaster.")
    fact_2 = Verify("Camlaren Mine was not halted by a natural disaster.")
    label = Predict(fact_1 and fact_2)

# The claim is that John O'Hara and Rabindranath Tagore are not the same nationality.
def program():
    answer_1 = Question("What is the nationality of John O'Hara?")
    answer_2 = Question("What is the nationality of Rabindranath Tagore?")
    fact_1 = Verify(f"{answer_1} and {answer_2} are not the same nationality.")
    label = Predict(fact_1)

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
    fact_2 = Verify(f"{answer_2} drives {answer_1} in the NASCAR Sprint Cup.")
    label = predict(fact_1 and fact_2)

# The claim is that [[CLAIM]]
def program():'''

HOVER_REDUCED = '''Generate a python-like program that describes the reasoning steps required to verify the claim step-by-step. You can call three functions in the program: 1. Question() to answer a question; 2. Verify() to verify a simple claim; 3. Predict() to predict the veracity label. Several examples are given as follows.

# The claim is that Howard University Hospital and Providence Hospital are both located in Washington, D.C.
def program():
    fact_1 = Verify("Howard University Hospital is located in Washington, D.C.")
    fact_2 = Verify("Providence Hospital is located in Washington, D.C.")
    label = Predict(fact_1 and fact_2)
    
# The claim is that An IndyCar race driver drove a Formula 1 car designed by Peter McCool during the 2007 Formula One season.
def program():
    answer_1 = Question("Which Formula 1 car was designed by Peter McCool during the 2007 Formula One season?")
    fact_1 = Verify(f"An IndyCar race driver drove the car {answer_1}.")
    label = Predict(fact_1)

# The claim is that WWE Super Tuesday took place at an arena that currently goes by the name TD Garden.
def program():
    answer_1 = Question("Which arena the WWE Super Tuesday took place?")
    fact_1 = Verify(f"{answer_1} currently goes by the name TD Garden.")
    label = Predict(fact_1)

# The claim is that Talking Heads, an American rock band that was "one of the most critically acclaimed bands of the 80's" is featured in KSPN's AAA format.
def program():
    fact_1 = Verify("Talking Heads is an American rock band that was 'one of the most critically acclaimed bands of the 80's'.")
    fact_2 = Verify("Talking Heads is featured in KSPN's AAA format.")
    label = Predict(fact_1 and fact_2)
    
# The claim is that The model of car Trevor Bayne drives was introduced for model year 2006. The Rookie of The Year in the 1997 CART season drives it in the NASCAR Sprint Cup.
def program():
    answer_1 = Question("Which model of car is drived by Trevor Bayne?")
    fact_1 = Verify(f"{answer_1} was introduced for model year 2006.")
    answer_2 = Question("Who is the Rookie of The Year in the 1997 CART season?")
    fact_2 = Verify(f"{answer_2} drives {answer_1} in the NASCAR Sprint Cup.")
    label = predict(fact_1 and fact_2)

# The claim is that The song recorded by Fergie that was produced by Polow da Don and was followed by Life Goes On was M.I.L.F.$.
def program():
    fact_1 = Verify("M.I.L.F.$ was recorded by Fergie.")
    fact_2 = Verify("M.I.L.F.$ was produced by Polow da Don.")
    fact_3 = Verify("M.I.L.F.$ was was followed by Life Goes On.")
    label = Predict(fact_1 and fact_2 and fact_3)    

# The claim is that Gina Bramhill was born in a village. The 2011 population of the area that includes this village was 167,446.
def program():
    answer_1 = Question("Which village was Gina Bramhill born in?")
    fact_1 = Verify(f"The 2011 population of the area that includes {answer_1} was 167,446.")
    label = Predict(fact_1)

# The claim is that Don Ashley Turlington graduated from Saint Joseph's College, a private Catholic liberal arts college in Standish.
def program():
    fact_1 = Verify("Saint Joseph's College is a private Catholic liberal arts college in Standish.")
    fact_2 = Verify(f"Don Ashley Turlington graduated from Saint Joseph's College.")
    label = Predict(fact_1 and fact_2)
    
# The claim is that In the 2004 Hockey film produced by a former major league baseball pitcher Kurt Russell played the USA coach.
def program():
    answer_1 = Question("Which 2004 Hockey film was produced a former major league baseball pitcher?")
    fact_1 = Verify("Kurt Russell played the USA coach in the film {answer_1}.")
    label = Predict(fact_1)

# The claim is that Jack McFarland is the best known role of the host of the 64th Annual Tony Awards.
def program():
    answer_1 = Question("Who are the hosts of the 64th Annual Tony Awards?")
    fact_1 = Verify(f\"Jack McFarland is the best known role of {answer_1}.)
    label = Predict(fact_1)

# The claim is that [[CLAIM]]
def program():
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


def render_span(begin, end, unidirectional=False):
    if end - begin == 1:
        return ['───►']
    elif end - begin == 2:
        return [
            '──┐',
            '──┴►',
        ] if unidirectional else [
            '◄─┐',
            '◄─┴►',
        ]

    rows = []
    for i in range(begin, end):
        if i == (end - begin) // 2 + begin:
            rows.append('  ├►')
        elif i == begin:
            rows.append('──┐' if unidirectional else '◄─┐')
        elif i == end - 1:
            rows.append('──┘' if unidirectional else '◄─┘')
        else:
            rows.append('  │')
    return rows


def make_table(rows, insert_header=False):
    col_widths = [max(len(s) for s in col) for col in zip(*rows[1:])]
    rows[0] = [x[:l] for x, l in zip(rows[0], col_widths)]
    fmt = '\t'.join('%%-%ds' % width for width in col_widths)
    if insert_header:
        rows.insert(1, ['─' * width for width in col_widths])
    return '\n'.join(fmt % tuple(row) for row in rows)


def render_labeled_span(b, e, spans, labels, label, offset, unidirectional=False):
    spans.extend([''] * (b - offset))
    spans.extend(render_span(b, e, unidirectional))
    center = b + (e - b) // 2
    labels.extend([''] * (center - offset))
    labels.append(label)
    labels.extend([''] * (e - center - 1))


def condense(block_, extras_=None):
    text_ = make_table(block_, insert_header=False)
    text_ = [x.split('\t', 1) for x in text_.split('\n')]
    text_ = [[x[0], x[1].replace('\t', '')] for x in text_]
    if extras_:
        for r, s in zip(extras_, text_):
            r.extend(s)
    return text_


def pretty_print(tree):
    length = len(tree.pos())
    block = [[] for _ in range(length + 1)]
    extras = [[] for j in range(length + 1)]
    block[0].extend(('Token', 'PoS'))
    for j, t in enumerate(tree.pos()):
        block[j + 1].extend(t)

    for height in range(2, tree.height() + (0 if len(tree) == 1 else 1)):
        offset = 0
        spans = []
        labels = []
        for k, subtree in enumerate(tree.subtrees(lambda x: x.height() == height)):
            subtree: Tree = subtree
            b, e = offset, offset + len(subtree.leaves())
            if height >= 3:
                b, e = subtree[0].center, subtree[-1].center + 1
            subtree.center = b + (e - b) // 2
            render_labeled_span(b, e, spans, labels, subtree.label(), offset, unidirectional=True)
            offset = e
        if len(spans) != length:
            spans.extend([''] * (length - len(spans)))
        if len(labels) != length:
            labels.extend([''] * (length - len(labels)))
        if height < 3:
            continue
        block[0].extend(['', f'{height}'])
        for j, (_s, _t) in enumerate(zip(spans, labels)):
            block[j + 1].extend((_s, _t))
        # check short arrows and increase their length
        for j, arrow in enumerate(spans):
            if not arrow:
                # -1 current tag ; -2 arrow to current tag ; -3 = prev tag ; -4 = arrow to prev tag
                if block[j + 1][-3] or block[j + 1][-4] == '───►':
                    if height > 3:
                        if block[j + 1][-3]:
                            block[j + 1][-1] = block[j + 1][-3]
                            block[j + 1][-2] = '───►'
                        else:
                            block[j + 1][-1] = '────'
                            block[j + 1][-2] = '────'
                        block[j + 1][-3] = '────'
                        if block[j + 1][-4] == '───►':
                            block[j + 1][-4] = '────'
                    else:
                        block[j + 1][-1] = '────'
                    if block[j + 1][-1] == '────':
                        block[j + 1][-2] = '────'
                    if not block[j + 1][-4]:
                        block[j + 1][-4] = '────'
    # If the root label is shorter than the level number, extend it to the same length
    level_len = len(block[0][-1])
    for row in block[1:]:
        if row[-1] and len(row[-1]) < level_len:
            row[-1] = row[-1] + ' ' * (level_len - len(row[-1]))

    text = condense(block)
    # Cosmetic issues
    for row in text[1:]:
        while '  ─' in row[1]:
            row[1] = row[1].replace('  ─', ' ──')
        row[1] = row[1].replace('─ ─', '───')
        row[1] = re.sub(r'([►─])([\w-]*)(\s+)([│├])', lambda
            m: f'{m.group(1)}{m.group(2)}{"─" * len(m.group(3))}{"┤" if m.group(4) == "│" else "┼"}',
                        row[1])
        row[1] = re.sub(r'►(─+)►', r'─\1►', row[1])
    for r, s in zip(extras, text):
        r.extend(s)
    return make_table(extras, insert_header=True)


class Prompt_Loader:
    def __init__(self) -> None:
        self.hover_program_fc = HOVER_REDUCED
        self.feverous_program_fc = FEVEROUS_PROGRAM_FC
        # with open('HOVER-chat-3.5.json', 'r', encoding='utf-8') as f:
        #     self.chatcot = json.load(f)
        with open('HOVER-chat-parsing-4.json', 'r', encoding='utf-8') as f:
            self.chatcot = json.load(f)
        with open('HOVER-adaptive-llama3.json', 'r', encoding='utf-8') as f:
            self.adaptive = json.load(f)
        self.HanLP = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)

    def prompt_construction(self, claim, dataset_name, chatcot=False, adaptive=False):
        template = None
        if adaptive:
            prompt = self.adaptive.copy()
            doc = self.HanLP(claim)
            prompt.append({"role": "user",
                           "content": f"Claim:{claim}\nConstituency parsing tree:\n{pretty_print(doc['con'])}\n Now "
                                      f"you need to briefly show how you reasoning using the parse tree(within 300 "
                                      f"words) and generate a single program in a code block."})
            return prompt
        if chatcot:
            prompt = self.chatcot.copy()
            doc = self.HanLP(claim)
            prompt.append({'role': "user",
                           "content": f"The claim is that {claim}.The constituency parsing tree is:\n{pretty_print(doc['con'])}"})
            return prompt
        if dataset_name == 'HOVER':
            template = self.hover_program_fc
        elif dataset_name == 'FEVEROUS':
            template = self.feverous_program_fc
        else:
            raise NotImplementedError

        return template.replace('[[CLAIM]]', claim)

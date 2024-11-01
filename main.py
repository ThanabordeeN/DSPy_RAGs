import os
import dspy
from semetics_sqlite_vec import SQLiteVecManager

lm = dspy.LM('ollama/granite3-moe:latest',api_base_url='http://localhost:11434')
dspy.configure(lm=lm)

class Retrive_pipe(dspy.Signature):
    """Generate Answer from Context"""
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()
    
class Keyword_pipe(dspy.Signature):
    """Generate Keywords from Question to search in Context"""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="should be a list of keywords")


class Agent_Retrive(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrive = dspy.ChainOfThought(Retrive_pipe)
        self.keyword = dspy.ChainOfThought(Keyword_pipe)
    
    def forward(self, question):
        db = SQLiteVecManager("state_union")
        key_word = self.keyword(question=question).answer
        
        
        context = db.query_text(key_word)
        print(context)
        print("--------------------------------")
        print(key_word)
        return self.retrive(question=question,context=context).answer
    
agent = Agent_Retrive()
print(agent("where is world cup 2026 going to be held?"))

print(agent("Number of Teams in world cup 2026?"))

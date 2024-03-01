SIMPLIFY_QUERY_PASSAGE = """I have a question:
[QUERY]
I want to solve it step by step. Now I get some information:
[KNOWLEDGE]

If these information can solve this question, please provide the answer directly. The answer should be as short as possible, like a word, a number or a shot span. You should return the answer in the form of '\{"question": question, "answer":answer\}'.
If the infomation is related to the question, please help me simplify this question. You can only simplify it by replace the corresponding entity in the question.
If these information is not related to the question, please return NO_SIMPLIFY.
Attention: You must make sure that you replace the completely same part, no more or less. 
Only give me the question without any explanation or description."""

SIMPLIFY_QUERY_CAPTION = """I have a question:
[QUERY]
I want to solve it step by step. Now I get some information:
[IMAGE]

If these information can solve this question, please provide the answer directly. The answer should be as short as possible, like a word, a number or a shot span. You should return the answer in the form of '\{"question": question, "answer":answer\}'.
If the infomation is related to the question, please help me simplify this question. You can only simplify it by replace the corresponding entity in the question.
If these information is not related to the question, please return NO_SIMPLIFY.
Attention: You must make sure that you replace the completely same part, no more or less. 
Only give me the question without any explanation or description."""

SIMPLIFY_QUERY_IMAGE = """I have a question:
[QUERY]
I want to solve it step by step. Now I get some information and image:
[IMAGE]

If these information and image can solve this question, please provide the answer directly. The answer should be as short as possible, like a word, a number or a shot span. You should return the answer in the form of '\{"question": question, "answer":answer\}'.
If the infomation and image is related to the question, please help me simplify this question. You can only simplify it by replace the corresponding entity in the question.
If these information and image is not related to the question, please return NO_SIMPLIFY.
Attention: You must make sure that you replace the completely same part, no more or less. 
Only give me the question without any explanation or description."""
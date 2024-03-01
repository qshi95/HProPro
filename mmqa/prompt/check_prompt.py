CHECK_PROMPT = """Please verify whether the semantics of the two strings meet the given conditions.
For example:
Q: ten > 9
Check: True
Q: 21 Nov, 2030 < 09-31-2021
Check: False
Q: Beijing == The capital of China
Check: True
Q: 2022/10/01 == Oct 1st, 2022
Check: True

Q: [STRING1] [REL] [STRING2] 
Check:"""

CHECK_SAME_PROMPT = """Please verify whether the semantics of the two strings are consistent. We don't need them to be exactly the same, as long as they contain semantic similarity. 
Attention:
0. If you think the two objects are always the same in any condition, return Ture; Else return False.
1. "August" and "2023-08" can also be judged to be similar (you return true), because they are the same month; but if the precise date or month is DIFFERENT, you should return False.
2. If it is other descriptive text(adj, adv), similar is ok. 
3. "NOT_AVAILABLE" is different with any text, return False.

Query : [QUERY]

String 1 : [STRING1]

String 2 : [STRING2]

You can only return 'True' or 'False' directly without any explanation."""
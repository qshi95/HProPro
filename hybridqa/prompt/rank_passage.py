

PASSAGE_RANK_PROMPT = """
Passages:
[PASSAGES]
Query:
[QUERY]

Which are the most useful passages to answer this query? You can choose several passages.
Let's think step by step, and end your answer with 
```
So my answer is 1. (If you choose 1)
```
or
```
So my answer is 0, 1, 2. (If you choose 0, 1 and 2)
```
"""
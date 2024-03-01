EXTRACT_INFO_CAPTION = """Read the following passage and find the answer from the passage according to the given query. Sometiomes the passage contains an image. If the image exists, a caption describing the image will be provided after the passage.

Passage:
[PASSAGE]

[CAPTION]

Query: [QUERY]

Let's find the Answer. If the answer is not available in the passage, the information should be marked as NOT_AVAILABLE. 
The answer should be as short as possible, like a word, a number or a shot span.
Please return the information directly without any explanation:"""

EXTRACT_IMAGE_INFO = """Read the following passage and image and find the answer from the passage and image according to the given query

Passage:
[PASSAGE]

Query: [QUERY]

Let's find the Answer. If the answer is not available in the passage and image, the information should be marked as NOT_AVAILABLE. 
The answer should be as short as possible, like a word, a number or a shot span.
Please return the information directly without any explanation:"""
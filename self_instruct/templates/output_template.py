template_2 = ''' Given an instruction, find the main strategies to solve or answer the instruction. Only answer with what you are confident is correct. 

Answer with at most 3 strategies.
Answer extremely concisely with no preamble.

Example:
instruction: Develop a marketing strategy for launching a new product, focusing on digital channels and social media engagement.
strategies: Identify target audience and eco-conscious segments.
Highlight product's eco-friendly benefits in messaging.
Use influencer partnerships with sustainability advocates.

Task:
instruction: {instruction}
strategies:

'''

template_3 = ''' Given an classification instruction, find the possible output labels. Only answer with what you are confident is correct. 

Answer extremely concisely with no preamble.

Example:
Task: Fact checking - tell me if the statement is true, false, or unknown, based on your knowledge and common sense.
labels: true, false, unknown

Example:
Task: Detect if the Reddit thread contains hate speech.
labels: yes, no

Example:
Task: Identify the pos tag of the word in the given sentence.
labels: noun, verb, adjective, adverb

Your task ask:
instruction: {instruction}
labels:

'''
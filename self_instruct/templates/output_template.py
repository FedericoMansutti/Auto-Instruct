template_2 = ''' Given an instruction, find the main strategies to solve or answer the instruction. Only answer with what you are confident is correct. 

Answer with 1, 2, or 3 strategies.
Answer extremely concisely with no preamble.
If needed, also provide the input for the instruction.

Example:
instruction: Develop a marketing strategy for launching a new product, focusing on digital channels and social media engagement.
input: None
strategies: Identify target audience and eco-conscious segments.
Highlight product's eco-friendly benefits in messaging.
Use influencer partnerships with sustainability advocates.

Example:
instruction: Please provide the equation you would like to solve for x.
input: x^2 + 2x + 1 = 0
strategies: Solve the equation by factoring. Use the quadratic formula.

Example:
instruction: What is the capital of France?
input: None
strategies: None

Example:
instruction: Summarize california's minimum wage laws, including any recent changes or exceptions.
input: None
strategies: None

Example:
instruction: Design a simple logo for a fictional company, explaining your design choices and color scheme.
input: None
strategies: Use minimalistic design for clarity and memorability. Choose colors that reflect the company's values and industry. 

Example:
instruction: Analyze the given text and summarize its main points in a concise paragraph.
input: Large "instruction-tuned" language models (i.e., finetuned to respond to instructions) have demonstrated a remarkable ability to generalize zero-shot to new tasks. Nevertheless, they depend heavily on human-written instruction data that is often limited in quantity, diversity, and creativity, therefore hindering the generality of the tuned model. We introduce Self-Instruct, a framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off their own generations. Our pipeline generates instructions, input, and output samples from a language model, then filters invalid or similar ones before using them to finetune the original model. Applying our method to the vanilla GPT3, we demonstrate a 33% absolute improvement over the original model on Super-NaturalInstructions, on par with the performance of InstructGPT-001, which was trained with private user data and human annotations. For further evaluation, we curate a set of expert-written instructions for novel tasks, and show through human evaluation that tuning GPT3 with Self-Instruct outperforms using existing public instruction datasets by a large margin, leaving only a 5% absolute gap behind InstructGPT-001. Self-Instruct provides an almost annotation-free method for aligning pre-trained language models with instructions, and we release our large synthetic dataset to facilitate future studies on instruction tuning
strategies: Identify the main points and structure the text coherently.

Now its your turn, provide strategies and input if needed. Follow the same format as the examples above. Make sure to also answer with 2 or even 1 strategy if needed.
Remember: generate first the input and AFTER the strategy/strategies. Think step by step.
instruction: {instruction}'''

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

Follow the same format as the examples above. Think step by step.
instruction: {instruction}
labels:'''
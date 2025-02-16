## What are Agents?
1. Reasoning and planning -> Execute actions
2. Vision Language model + Large language model + email or other tools etc -> Tools

## LLMs
1. tokens
2. encoder decoder based transformers
3. End of Sequence tokens. LLM decodes the next token until it reaches the EOS token.
for smolLM2 find special tokens here: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/main/tokenizer_config.json
4. for decoding the next word, a basic word probability is used. also known as greedy decoding.
5. For advanced decoding, beam search is used. Beam search, searches the best sequence of tokens and their total value (or score) is calculated. 

### Prompting the LLM
Considering that the only job of an LLM is to predict the next token by looking at every input token, and to choose which tokens are “important”, the wording of your input sequence is very important.

The input sequence you provide an LLM is called a prompt. Careful design of the prompt makes it easier to guide the generation of the LLM toward the desired output. **context length**, which refers to the maximum number of tokens the LLM can process, and the maximum **attention span** it has, are important factors. 

### Messages and Special tokens
Chat is just a UI. written prompt is converted to system prompt on the UI. This takes into account the given model's special and EOS tokens while converting to system prompt or message.

system message example: 
```python
    system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
    }
```
### conversation or context
series of messages between User and LLM (asssistant)
The whole converstaion is stored, concatenated and passed everytime a new message is exchanged. 
Every model has its own code for handling the conversation structure by using their Special Tokens.
ex. 
```python
    conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
    ]
```
            OR with smolLM2:
```python
    <|im_start|>system
    You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
    <|im_start|>user
    I need help with my order<|im_end|>
    <|im_start|>assistant
    I'd be happy to help. Could you provide your order number?<|im_end|>
    <|im_start|>user
    It's ORDER-123<|im_end|>
    <|im_start|>assistant
```

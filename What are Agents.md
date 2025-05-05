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

### Chat Templates
1. Base models: base model is trained on raw text data to predict the next token
2. Instruct model: fine tuned base model to follow instructions and engage in a conversation

#### Standard template for Chat is:
**ChatML** format with roles **system**, **user** and **assistant**. example
```python
    chat = [
    {"role": "system", "content": "You are a professional customer service agent. Always be polite, clear, and helpful."},
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
    {"role": "assistant", "content": "Thank you. Let me check on that for you."},
    ]
```
here is a simplified version of the instruct chat template:
```python
    {% for message in messages %}
    {% if loop.first and messages[0]['role'] != 'system' %}
    <|im_start|>system
    You are a helpful AI assistant named SmolLM, trained by Hugging Face
    <|im_end|>
    {% endif %}
    <|im_start|>{{ message['role'] }}
    {{ message['content'] }}<|im_end|>
    {% endfor %}
```
the above converts our converstaion and messages into system string. this is also known as the system prompt. This now goes to tokenization. 
After selecting the model, we must apply the chat template of teh model to convert it to teh system prompt before passing it into the tokenizer. 
```python
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```
This rendered prompt is now ready to be the input for the model. 


## Tools
a tool is a function given to the LLM with a clear objective. 
common ones are web search, image generation, retrieval, API interface

cant ask a base LLM model to give up to date results. For example, we cannot ask a base LLM to tell us the weather today without giving it access to a wetaher search engine or app. 

it should have 4 parts:
1. Textual description of what the function does
2. a callable (something to perform an action)
3. Arguments with typings
4. Outputs with typings (optional)

LLM generates the text in the form of code to invoke a tool. This is the AGENT. the output from the tool is recieved by the LLM and returned to the user. AGENT is in the background and the user does not see the AGENT's work as system prompt as in the normal conversation. 

We could provide the Python source code as the specification of the tool for the LLM, but the way the tool is implemented does not matter. All that matters is its name, what it does, the inputs it expects and the output it provides.

Use the @tool method to define a tool in python. this avoids making long class definitions for the simple tool. 
```python
    @tool
    def calculator(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    print(calculator.to_string())
```

## Thought -> Action -> Observation cycle

1. **Thought**: LLM part of teh AGENT decides on teh next steps
     **ReAct** approach of breaking down in to smaller tasks and thinking step by step. just a simple prompt of **let's think step by step** can be used to break down the task. This is the approach behind Deepseek and OpenAI O1 models; to show the Reasoning. however these models dont just have special prompting like ReAct but is a training method. system prompt <think> and </think>.

        | Type of Thought | Example |
        | --- | --- |
        | Planning | “I need to break this task into three steps: 1) gather data, 2) analyze trends, 3) generate report” |
        | Analysis | “Based on the error message, the issue appears to be with the database connection parameters” |
        | Decision Making | “Given the user’s budget constraints, I should recommend the mid-tier option” |
        | Problem Solving | “To optimize this code, I should first profile it to identify bottlenecks” |
        | Memory Integration | “The user mentioned their preference for Python earlier, so I’ll provide examples in Python” |
        | Self-Reflection | “My last approach didn’t work well, I should try a different strategy” |
        | Goal Setting | “To complete this task, I need to first establish the acceptance criteria” |
        | Prioritization | “The security vulnerability should be addressed before adding new features” |

2. **Action**: AGENT invokes the tool to perform the action
    Differnet types of agents: **JSON agent** where action to take is specified in a json format, **Code Agent** where the agent writes a code block that is interpreted externally and **Function-calling agent** which is a sub-category of JSON agent and is fine tuned to generate a new message for each action. crucial part of any agent is the ability **to stop generating new tokens when an action is complete**. Using the **Stop and Parse** approach we give a structured JSON format to output the action. This helps in halting the action, clear responses and avoiding erroneous tokens. for advanced handling, we can allow Code Agents which can interact with external systems and have more functionalities and flexibilities. 

3. **Observation**: AGENT observes the output of the tool and returns it to the user, if not satisfied with the output, it can invoke the tool again. The steps for Observation are simple, **Collects feedback** by confirming whether the action was successful or not, **Appends Results** by integrating results into the existing context and thus updating its memory and **Adapts its Strategy** by refining subsequent thoughts and actions based on the updated context. some examples of observation are:
    - “The database connection was successful, I can now proceed with the query”
    - “The API returned an error, I need to check the request parameters”
    - “The user confirmed that the report format is acceptable”
    -"the sensor readings were so and so..." etc. 

    #### Parse the Action -> Execute the Action -> Apend the result as an Observation 


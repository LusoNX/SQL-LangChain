# Simple AI backed Database query
A simple implementation of Langchain module (SQL Agent) that allows users to interact with their databases in a simpler format (similar to ChatGPT but applicable to your database). 

The script allows for the usage of different LLM models. By default OpenAI LLM model is used, since the application of the other models is not straightforward. 
Ideally, modules are runned locally, and this is a special consideration for production environment since pulling of data using OpenAI may not be advisable (basically, the data and context of the database is pulled to OpenAI, which can be troublesome regarding the privacy of the data). 

Nevertheless, the simple application is usfeul to taking the 1st steps in applying this sort of modules. 


### 1. Initiate the LLM Model (3 by default, but you can check additional models in https://huggingface.co/. HuggingFace is an open library that allow developers to share LLM models and apply costumized and fine tuned models for their needs. 
![image](https://github.com/LusoNX/SQL-LangChain/assets/84282116/19336b1a-94a4-45ae-aa3d-0e044c9b5da2)

### 2. Initiate the database and include the tables to be queried, and some row examples to allow the model to better contextualize the information
   NOTE: This is useful for two reasons. One you limit the ammount of tokens used in your prompt, by segmenting the querying only to the tables you wish to "talk to" and you limit the amount of information that is shared with the LLM (If runned locally, not a big issue, but if using an outside provider it may be important)
![image](https://github.com/LusoNX/SQL-LangChain/assets/84282116/90c03450-7061-4836-810f-d304ba7a9d6c)

### 3. Provide some few shots examples.
   This is one way of fine-tunning your model, by providing a context on how you wish the data to be queried. Simply format your question accordingly to what you want, and provide a SQL Query example on how to get there.

![image](https://github.com/LusoNX/SQL-LangChain/assets/84282116/272b2809-be24-4394-833f-8e2ccf694d71)


### 4. Initiate the agent.
   Finally, run the agent and test your model. 

![image](https://github.com/LusoNX/SQL-LangChain/assets/84282116/c202e75f-b86c-4171-8894-9239b4bf95b0)


(*) Important NOTES: First of all, this is a simple implementation of on agent available in the Langchain library. Multiple agents can be applied, and the final Agent (ex.: a ChatBot Assistent), responsible of interpreting the required information and direct the questions
to the proper action agent. That is, we can combine multiple agents into one final agent, and this agent uses the tools of each individual agent for the task in hands. This is helpful, even if agents are simply required to query the dataset, because it allows you to defined specific 
task for different tables within your database. Furthermore this agents may go beyong the querying of the database and apply different actions, specific for your use cases. For example, you may be interested in knowing what was the CAGR between two assets in the last 3 months.
Even if the data is not pre-processed in the database, you may create a "StatsTool" Agent, that calculates the the CAGR(3M), and the only information you are required to have is the price data. 

Finally, the example refers only to "prompt programming" fine-tuning of the models (the simpler format). However, you may find useful to fine-tune your models before actually prompt them. This is also achievable, but a bit more complex, and beyond the scope of this simple script.

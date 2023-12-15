import set_env
from langchain.agents import create_sql_agent 
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase 
from langchain.llms.openai import OpenAI 
from langchain.agents import AgentExecutor 
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain import HuggingFaceHub
from langchain.memory import ConversationBufferMemory,ReadOnlySharedMemory
from langchain.prompts import PromptTemplate,FewShotPromptTemplate

from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

class SQLReaderAgent:
    def __init__(self,database_directory):
        self.database_directory = database_directory

    def LLM_model(self,type_model):
        # replace by a dictionary 
        if type_model == "OpenAI":
            llm = OpenAI(temperature=0)

        elif type_model == "google_large":
            llm = HuggingFaceHub(repo_id="google/flan-t5-xl",model_kwargs = {"temperature":0.8,"max_length":254})
        
        elif type_model == "llama_7b_chat_hf":
            llm = HuggingFaceHub(repo_id="meta-llama/Llama-2-7b-chat-hf",model_kwargs = {"temperature":0.8,"max_length":253})

        else:
            print("Please provide one of the available llm models: 'OpenAI' |'google_large'| 'llama_7b_chat_hf'")
        return llm 
    

    
    def init_database(self):
        input_db = SQLDatabase.from_uri("sqlite:///{}".format(self.database_directory))

        input_db = SQLDatabase.from_uri(
            "sqlite:///{}".format(self.database_directory),
            include_tables=[
                "FundsIndex",
                "LabelIndex",
                "RankIndex"
            ],  # we include only one table to save tokens in the prompt :)
            sample_rows_in_table_info=2,
        )
       
        return input_db

    def set_few_shots(self):
        few_shots = {
        "List all funds.": "SELECT [Name],[ISIN] FROM FundsIndex;",
        "Provide me the most conservative funds that invest in equity": "SELECT  LI.Name, LI.ISIN FROM LabelIndex LI JOIN FundsIndex F ON LI.ID = A.ID WHERE LI.'Capital Protection' = 'High' AND A.'Investment Class' = 'Equity' LIMIT 5;",
        "Which are the best growth funds that invest in Germany":"SELECT LI.Name, LI.ISIN FROM LabelIndex LI JOIN FundsIndex F ON LI.ID = F.ID JOIN RankIndex R ON R.ID = F.ID WHERE A.'Investment Region'='Germany' AND CI.'Investment Style' = 'Growth'  ORDER BY R.'Rank' LIMIT 5;",
        "Customize your prompts accordingly to your needs...":""
        }
        return few_shots
        

    def init_agent(self,llm_type,_memory):
        from langchain.agents.agent_toolkits import create_retriever_tool
        from langchain.schema import Document
        from langchain.vectorstores import FAISS
        from langchain.embeddings.openai import OpenAIEmbeddings


        llm = self.LLM_model(llm_type)
        input_db = self.init_database()

        # Set the few shots examples and the necessary prompt
        few_shots = self.set_few_shots()
        few_shot_docs = [
            Document(page_content=question, metadata={"sql_query": few_shots[question]})
            for question in few_shots.keys()
        ]

        embeddings = OpenAIEmbeddings()

        # Vectorize the embeddings and create the retrieve function so that you can apply the few shots. 
        vector_db = FAISS.from_documents(few_shot_docs, embeddings)
        retriever = vector_db.as_retriever()
        tool_description = """
        This tool will help you understand similar examples to adapt them to the user question.
        Input to this tool should be the user question.
        """

        # Custom tool used to help the agent in contextualizing the date 
        retriever_tool = create_retriever_tool(
            retriever, name="sql_get_similar_examples", description=tool_description
        )
        custom_tool_list = [retriever_tool]


        llm = ChatOpenAI(model_name="gpt-4", temperature=0)

        toolkit = SQLDatabaseToolkit(db=input_db, llm=llm)

        custom_suffix = """
        I should first get the similar examples I know.
        If the examples are enough to construct the query, I can build it.
        Otherwise, I can then look at the tables in the database to see what I can query.
        Then I should query the schema of the most relevant tables
        """

        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            extra_tools=custom_tool_list,
            suffix=custom_suffix,
        )
        return agent



agent_instance = SQLReaderAgent(r"YOURDATABASE.db")
db_agent = agent_instance.init_agent("OpenAI","")
db_agent.run("Which are the best Equity Funds")

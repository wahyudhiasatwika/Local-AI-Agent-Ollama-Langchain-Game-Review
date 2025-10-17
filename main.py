from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You an expert in game reviews. Use the following reviews to answer the question.
Here are some relevant review: {reviews}
Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("==============================")
    question = input("Ask question (q to quit) : ")
    if question == 'q':
        break

    reviews = retriever.invoke(question)    
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
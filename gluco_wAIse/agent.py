from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from utils.generate_doc import generate_docx
from utils.rag_utils import retriever

load_dotenv()

# Tools and LLM setup
tools = [tool(generate_docx)]
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(
    content=(
        "You are a helpful assistant specialized in supporting people with diabetes. "
        "Use the following retrieved Q&A to help you answer user queries.\n"
        "If the user asks you to generate a summary, plan, or document, use the generate_docx_tool. "
        "Do not ask to generate a document unless explicitly requested by the user."
    )
)

# RAG prompt
rag_prompt = PromptTemplate.from_template(
    "The user asked: {question}\n\n"
    "Here are related Q&A entries retrieved from the knowledge base:\n\n"
    "{context}\n\n"
    "Use these to answer the user's query accurately and concisely. "
    "Respond using only this knowledge base context. "
    "In particular, do not answer questions that are unrelated to diabetes or "
    "to nutrition. "
    "If the user asks for a document, call the appropriate tool."
)
rag_chain = LLMChain(llm=llm, prompt=rag_prompt)


# Assistant node
def tool_calling_llm(state: MessagesState):
    user_query = state["messages"][-1].content

    # RAG context
    docs = retriever.get_relevant_documents(user_query)
    if not docs:
        context = "No relevant documents found."
    else:
        context = "\n\n".join(
            f"Q: {doc.page_content}\nA: {doc.metadata.get('answer', '[No answer provided]')}"
            for doc in docs
        )

    prompt_input = {"question": user_query, "context": context}
    rag_response = rag_chain.invoke(prompt_input)["text"]

    # DEBUG
    print("\n===== RAG DEBUG =====")
    print("User query:", user_query)
    print("RAG answer:", rag_response)
    if docs:
        print(f"Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs):
            print(f"  [{i + 1}] Q: {doc.page_content}")
            print(f"       A: {doc.metadata.get('answer', '[No answer]')}")
    else:
        print("No source documents retrieved!")
    print("=====================\n")

    # Prepare message list for potential tool invocation
    messages = [
        sys_msg,
        HumanMessage(content=user_query),
        AIMessage(content=rag_response),
    ]

    # Use tool-aware LLM only to detect tool call
    tool_response = llm_with_tools.invoke(messages)

    if tool_response.tool_calls:
        return {"messages": [HumanMessage(content=user_query), tool_response]}
    else:
        # Return raw RAG response directly
        return {
            "messages": [
                HumanMessage(content=user_query),
                AIMessage(content=rag_response),
            ]
        }


# LangGraph definition
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "__end__")
builder.add_edge("tool_calling_llm", "__end__")

graph = builder.compile()

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: run-agent '<your message>'")
        sys.exit(1)

    user_input = sys.argv[1]
    state = {"messages": [HumanMessage(content=user_input)]}
    result = graph.invoke(state)

    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.endswith(".docx"):
                print(f"[DOCX GENERATED] Path: {content}")
            else:
                print(content)

if __name__ == "__main__":
    main()

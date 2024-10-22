from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import warnings
import shutil
import requests
import operator
import functools
from langchain_openai import ChatOpenAI, OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from bs4 import BeautifulSoup
from typing import TypedDict, Annotated, Sequence

load_dotenv()

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    # Permitir todas las solicitudes de origen. Cambia esto a la URL específica de tu frontend para mayor seguridad.
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar clave de API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

warnings.filterwarnings("ignore", category=SyntaxWarning,
                        message="invalid escape sequence")


@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str) -> str:
    """Parse web content with BeautifulSoup"""
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()


@tool("generate_image", return_direct=False)
def generate_image(prompt: str) -> str:
    """Generate an image using DALL-E 3 based on the provided prompt"""
    client = OpenAI()
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    foto = requests.get(image_url, stream=True)

    if foto.status_code == 200:
        image_path = 'generated_image.png'
        with open(image_path, 'wb') as f:
            shutil.copyfileobj(foto.raw, f)
        return f"Image generated and saved to {image_path}, URL: {image_url}"
    else:
        return 'Error in accessing the image'


tools = [TavilySearchResults(max_results=1),
         process_search_tool, generate_image]


def create_new_agent(llm: ChatOpenAI, tools: list, system_prompt: str) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


content_marketing_team = ["online_researcher",
                          "blog_manager", "social_media_manager", "image_generator"]

system_prompt = (
    "As a content marketing manager, your role is to oversee the insight between these"
    " workers: {content_marketing_team}. Based on the user's request,"
    " determine which worker should take the next action. Each worker is responsible for"
    " executing a specific task and reporting back their findings and progress."
    " Once all tasks are completed, indicate 'FINISH'."
)

options = ["FINISH"] + content_marketing_team

function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}]}},
        "required": ["next"]
    }
}

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system",
     "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
]).partial(options=str(options), content_marketing_team=", ".join(content_marketing_team))

content_marketing_manager_chain = (prompt | llm.bind_functions(
    functions=[function_def], function_call="route") | JsonOutputFunctionsParser())

online_researcher_agent = create_new_agent(
    llm,
    tools,
    """Your primary role is to function as an intelligent online research assistant, adept at scouring 
    the internet for the latest and most relevant trending stories across various sectors like politics, technology, 
    health, culture, and global events. You possess the capability to access a wide range of online news sources, 
    blogs, and social media platforms to gather real-time information."""
)

online_researcher_node = functools.partial(
    agent_node, agent=online_researcher_agent, name="online_researcher")

blog_manager_agent = create_new_agent(
    llm, tools,
    """You are a Blog Manager. The role of a Blog Manager encompasses several critical responsibilities aimed at transforming initial drafts into polished, SEO-optimized blog articles that engage and grow an audience. Starting with drafts provided by online researchers, the Blog Manager must thoroughly understand the content, ensuring it aligns with the blog's tone, target audience, and thematic goals. Key responsibilities include:

1. Content Enhancement: Elevate the draft's quality by improving clarity, flow, and engagement. This involves refining the narrative, adding compelling headers, and ensuring the article is reader-friendly and informative.

2. SEO Optimization: Implement best practices for search engine optimization. This includes keyword research and integration, optimizing meta descriptions, and ensuring URL structures and heading tags enhance visibility in search engine results.

3. Compliance and Best Practices: Ensure the content adheres to legal and ethical standards, including copyright laws and truth in advertising. The Blog Manager must also keep up with evolving SEO strategies and blogging trends to maintain and enhance content effectiveness.

4. Editorial Oversight: Work closely with writers and contributors to maintain a consistent voice and quality across all blog posts. This may also involve managing a content calendar, scheduling posts for optimal engagement, and coordinating with marketing teams to support promotional activities.

5. Analytics and Feedback Integration: Regularly review performance metrics to understand audience engagement and preferences. Use this data to refine future content and optimize overall blog strategy.

In summary, the Blog Manager plays a pivotal role in bridging initial research and the final publication by enhancing content quality, ensuring SEO compatibility, and aligning with the strategic objectives of the blog. This position requires a blend of creative, technical, and analytical skills to successfully manage and grow the blog's presence online."""
)

blog_manager_node = functools.partial(
    agent_node, agent=blog_manager_agent, name="blog_manager")

social_media_manager_agent = create_new_agent(
    llm, tools,
    """You are a Social Media Manager. The role of a Social Media Manager, particularly for managing Twitter content, involves transforming research drafts into concise, engaging tweets that resonate with the audience and adhere to platform best practices. Upon receiving a draft from an online researcher, the Social Media Manager is tasked with several critical functions:

1. Content Condensation: Distill the core message of the draft into a tweet, which typically allows for only 280 characters. This requires a sharp focus on brevity while maintaining the essence and impact of the message.

2. Engagement Optimization: Craft tweets to maximize engagement. This includes the strategic use of compelling language, relevant hashtags, and timely topics that resonate with the target audience.

3. Compliance and Best Practices: Ensure that the tweets follow Twitter’s guidelines and best practices, including the appropriate use of mentions, hashtags, and links. Also, adhere to ethical standards, avoiding misinformation and respecting copyright norms.

In summary, the Social Media Manager's role is crucial in leveraging Twitter to disseminate information effectively, engage with followers, and build the brand’s presence online. This position combines creative communication skills with strategic planning and analysis to optimize social media impact."""
)

social_media_manager_node = functools.partial(
    agent_node, agent=social_media_manager_agent, name="social_media_manager")

image_generator_agent = create_new_agent(
    llm, tools,
    """You are an Image Generator. Your task is to generate a creative and high-quality image based on the given prompt provided by the user. Use the image generation tool to create visuals that match the intended concept."""
)

image_generator_node = functools.partial(
    agent_node, agent=image_generator_agent, name="image_generator")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


workflow = StateGraph(AgentState)

workflow.add_node(key="content_marketing_manager",
                  action=content_marketing_manager_chain)
workflow.add_node(key="online_researcher", action=online_researcher_node)
workflow.add_node(key="blog_manager", action=blog_manager_node)
workflow.add_node(key="social_media_manager", action=social_media_manager_node)
workflow.add_node(key="image_generator", action=image_generator_node)

for member in content_marketing_team:
    workflow.add_edge(start_key=member, end_key="content_marketing_manager")

conditional_map = {k: k for k in content_marketing_team}
conditional_map['FINISH'] = END

workflow.add_conditional_edges(
    "content_marketing_manager", lambda x: x["next"], conditional_map)
workflow.set_entry_point("content_marketing_manager")

multiagent = workflow.compile()


@app.post("/api/generate-content")
async def generate_content(request: BaseModel):
    for s in multiagent.stream(
        {
            "messages": [
                HumanMessage(
                    content=request.prompt
                )
            ],
        },
        # Maximum number of steps to take in the multiagent
        {"recursion_limit": 150}
    ):
        if not "__end__" in s:
            print(s, end="\n\n-----------------\n\n")
    return {"result": "Proceso completado"}

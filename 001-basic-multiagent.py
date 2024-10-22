# Importaciones y configuraciones iniciales
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, Annotated, Sequence
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
import warnings
import requests
import operator
import functools
from langchain_openai import ChatOpenAI
import os

# Cargar variables de entorno
load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]


# Inicializar el modelo de lenguaje
llm = ChatOpenAI(model="gpt-4o-mini")

warnings.filterwarnings("ignore", category=SyntaxWarning,
                        message="invalid escape sequence")

# Definición de herramientas


@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str) -> str:
    """Parsear contenido web con BeautifulSoup."""
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()


# Lista de herramientas disponibles para los agentes
tools = [TavilySearchResults(max_results=3),
         process_search_tool, ]

# Función para crear nuevos agentes


def create_new_agent(llm: ChatOpenAI,
                     tools: list,
                     system_prompt: str) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Función para ejecutar un agente y devolver el mensaje resultante


def agent_node(state, agent, name):
    result = agent.invoke(state)
    output_message = result["output"]
    return {"messages": [HumanMessage(content=output_message, name=name)]}


# Definición de los miembros del equipo de marketing de contenidos
content_marketing_team = ["online_researcher",
                          "blog_manager", "social_media_manager"]

# Definición del administrador de marketing de contenidos
system_prompt = (
    "Como gerente de marketing de contenidos, tu rol es supervisar la interacción entre estos "
    "trabajadores: {content_marketing_team}. Basado en la solicitud del usuario, "
    "determina qué trabajador debe tomar la siguiente acción. Cada trabajador es responsable de "
    "ejecutar una tarea específica y reportar sus hallazgos y progreso. "
    "Una vez que todas las tareas estén completadas, indica 'FINISH'."
)

options = ["FINISH"] + content_marketing_team

function_def = {
    "name": "route",
    "description": "Selecciona el próximo rol.",
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
     "Dada la conversación anterior, ¿quién debería actuar a continuación? ¿O deberíamos FINISH? Selecciona uno de: {options}"),
]).partial(options=str(options), content_marketing_team=", ".join(content_marketing_team))

content_marketing_manager_chain = (prompt | llm.bind_functions(
    functions=[function_def], function_call="route") | JsonOutputFunctionsParser())

# Definición del agente online_researcher_agent
online_researcher_agent = create_new_agent(
    llm,
    tools,
    """Eres un asistente de investigación en línea. Tu tarea es:

1. Buscar información relevante en línea sobre el tema proporcionado por el usuario.
2. Utilizar únicamente la herramienta process_search_tool para buscar y procesar contenido web válido.
3. Proporcionar un resumen detallado de tus hallazgos sin intentar  utilizar esquemas no válidos.

Recuerda  enfocarte en URLs válidas que comiencen con 'http://' o 'https://'."""
)


online_researcher_node = functools.partial(
    agent_node, agent=online_researcher_agent, name="online_researcher"
)

# Definición del agente blog_manager_agent
blog_manager_agent = create_new_agent(
    llm, tools,
    """Eres un Gerente de Blog. Tu rol implica transformar borradores iniciales en artículos de blog pulidos y optimizados para SEO que involucren y hagan crecer una audiencia. Comenzando con borradores proporcionados por investigadores en línea, debes asegurarte de que el contenido se alinee con el tono, la audiencia objetivo y los objetivos temáticos del blog. Tus responsabilidades incluyen:

1. Mejora de Contenido: Elevar la calidad del borrador mejorando la claridad, fluidez y engagement. Esto implica refinar la narrativa, agregar encabezados atractivos y asegurar que el artículo sea fácil de leer e informativo.

2. Optimización SEO: Implementar las mejores prácticas para la optimización en motores de búsqueda. Esto incluye investigación e integración de palabras clave, optimización de meta descripciones y asegurarse de que las estructuras de URL y etiquetas de encabezado mejoren la visibilidad en los resultados de búsqueda.

3. Cumplimiento y Mejores Prácticas: Asegurar que el contenido cumpla con los estándares legales y éticos, incluyendo leyes de derechos de autor y veracidad en la publicidad. Debes mantenerte al día con las estrategias SEO y tendencias de blogging en evolución para mantener y mejorar la efectividad del contenido.

4. Supervisión Editorial: Trabajar de cerca con escritores y colaboradores para mantener una voz y calidad consistentes en todas las publicaciones del blog. Esto puede implicar también gestionar un calendario de contenido, programar publicaciones para un engagement óptimo y coordinar con equipos de marketing para apoyar actividades promocionales.

5. Análisis e Integración de Retroalimentación: Revisar regularmente métricas de rendimiento para entender el engagement y preferencias de la audiencia. Utiliza estos datos para refinar contenido futuro y optimizar la estrategia general del blog.

En resumen, juegas un papel fundamental en la conexión entre la investigación inicial y la publicación final al mejorar la calidad del contenido, asegurar la compatibilidad con SEO y alinearte con los objetivos estratégicos del blog. Esta posición requiere una combinación de habilidades creativas, técnicas y analíticas para gestionar y hacer crecer exitosamente la presencia del blog en línea."""
)

blog_manager_node = functools.partial(
    agent_node, agent=blog_manager_agent, name="blog_manager"
)

# Definición del agente social_media_manager_agent con capacidad para generar imágenes
social_media_manager_agent = create_new_agent(
    llm, tools,
    """Eres un Social Media Manager. Tu tarea es:

1. Crear un tweet conciso y atractivo basado en el contenido proporcionado.
2. Generar una imagen relacionada utilizando la herramienta 'generate_image' con una descripción adecuada.
3. Proporcionar el tweet y el nombre del archivo de la imagen generada.

Asegúrate de que el contenido sea atractivo y que cumpla con las políticas y mejores prácticas de Twitter."""
)

social_media_manager_node = functools.partial(
    agent_node, agent=social_media_manager_agent, name="social_media_manager"
)

# Definición del estado del agente


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


# Configuración del flujo de trabajo con StateGraph
workflow = StateGraph(AgentState)

workflow.add_node(key="content_marketing_manager",
                  action=content_marketing_manager_chain)
workflow.add_node(key="online_researcher", action=online_researcher_node)
workflow.add_node(key="blog_manager", action=blog_manager_node)
workflow.add_node(key="social_media_manager", action=social_media_manager_node)

for member in content_marketing_team:
    workflow.add_edge(start_key=member, end_key="content_marketing_manager")

conditional_map = {k: k for k in content_marketing_team}
conditional_map['FINISH'] = END

workflow.add_conditional_edges(
    "content_marketing_manager", lambda x: x["next"], conditional_map)

workflow.set_entry_point("content_marketing_manager")

multiagent = workflow.compile()

# Ejecución del sistema multiagente
for s in multiagent.stream(
    {
        "messages": [
            HumanMessage(
                content="""Escribe un informe sobre los tipos de negocios con aplicaciones multiagente LLM. Después de investigar, envía los resultados al gestor del blog para generar el artículo final. Una vez hecho, envíalo al gestor de redes sociales para que escriba un tweet sobre el tema y genere una imagen relacionada."""
            )
        ],
    },
    {"recursion_limit": 150}
):
    if not "__end__" in s:
        # Procesa y muestra el contenido y la imagen
        print(s, end="\n\n-----------------\n\n")

import pandas as pd
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere.chat_models import ChatCohere
import os

os.environ['COHERE_API_KEY'] = ""
chat = ChatCohere(model="command-r-plus", temperature=0.7)
csv_file_path = ""
data = pd.read_csv(csv_file_path)
data_list = data.to_dict(orient="records")
@tool
def get_full_names() -> list[str]:
    """Fetch the full names of products based on the CSV data."""
    full_names = data["full-name"].unique().tolist()
    return full_names

@tool
def get_domains() -> list[str]:
    """Fetch the domains of products based on the CSV data."""
    domains = data["domain"].unique().tolist()
    return domains

@tool
def get_sub_domains() -> list[str]:
    """Fetch the sub-domains of products based on the CSV data."""
    sub_domains = data["sub-domain"].unique().tolist()
    return sub_domains

@tool
def get_super_products() -> list[str]:
    """Fetch the super-products based on the CSV data."""
    super_products = data["super-product"].unique().tolist()
    return super_products

@tool
def get_products() -> list[str]:
    """Fetch the products based on the CSV data."""
    products = data["product"].unique().tolist()
    return products

@tool
def get_categories() -> list[str]:
    """Fetch the categories based on the CSV data."""
    categories = data["category"].unique().tolist()
    return categories

@tool
def get_brands() -> list[str]:
    """Fetch the brands based on the CSV data."""
    brands = data["brand"].unique().tolist()
    return brands

@tool
def get_sub_brands() -> list[str]:
    """Fetch the sub-brands based on the CSV data."""
    sub_brands = data["sub-brand"].unique().tolist()
    return sub_brands

@tool
def get_sizes() -> list[str]:
    """Fetch the sizes based on the CSV data."""
    sizes = data["size"].unique().tolist()
    return sizes

@tool
def get_flavors() -> list[str]:
    """Fetch the flavors based on the CSV data."""
    flavors = data["flavor"].unique().tolist()
    return flavors

@tool
def get_countries() -> list[str]:
    """Fetch the countries based on the CSV data."""
    countries = data["country"].unique().tolist()
    return countries

@tool
def get_price() -> list[dict]:
    """Fetch the prices based on the CSV data."""
    prices = data[["full-name", "price"]].to_dict(orient="records")
    return prices


prompt = ChatPromptTemplate.from_template("{input}")
agent = create_cohere_react_agent(
    llm=chat, 
    tools=[
        get_full_names,
        get_domains,
        get_sub_domains,
        get_super_products,
        get_products,
        get_categories,
        get_brands,
        get_sub_brands,
        get_sizes,
        get_flavors,
        get_countries,
        get_price
    ],
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=[
    get_full_names,
    get_domains,
    get_sub_domains,
    get_super_products,
    get_products,
    get_categories,
    get_brands,
    get_sub_brands,
    get_sizes,
    get_flavors,
    get_countries,
    get_price
], verbose=True)

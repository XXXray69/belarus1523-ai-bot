from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os


load_dotenv()


PDF_FILE = "knowledge/manual.pdf"
DB_PATH = "database"


print("=" * 50)
print("Создание базы знаний МТЗ-1523")
print("=" * 50)





if os.path.exists(DB_PATH):

    print("\nВНИМАНИЕ!")
    print("Папка database уже существует.")
    print("Удалите её вручную и запустите скрипт снова.")
    print("Путь:")
    print(os.path.abspath(DB_PATH))

    exit()





print("\n1. Загрузка руководства...")


loader = PyPDFLoader(
    PDF_FILE
)


documents = loader.load()


print(
    "Страниц загружено:",
    len(documents)
)





print("\n2. Разбиение текста...")


splitter = RecursiveCharacterTextSplitter(

    chunk_size=2000,

    chunk_overlap=400,

    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]

)


chunks = splitter.split_documents(
    documents
)


print(
    "Создано частей:",
    len(chunks)
)





print("\n3. Создание модели поиска...")


embeddings = HuggingFaceEmbeddings(

    model_name="intfloat/multilingual-e5-large",

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    }

)


print(
    "Модель загружена"
)





print("\n4. Создание базы Chroma...")


db = Chroma.from_documents(

    documents=chunks,

    embedding=embeddings,

    persist_directory=DB_PATH

)


db.persist()



print("\n" + "=" * 50)

print("БАЗА ЗНАНИЙ СОЗДАНА!")

print(
    "Фрагментов:",
    len(chunks)
)

print(
    "Путь:",
    os.path.abspath(DB_PATH)
)

print("=" * 50)





print("\n5. Проверка поиска...")


questions = [

    "назначение трактора МТЗ-1523",

    "для чего предназначен трактор",

    "технические характеристики",

    "коробка передач"

]


for q in questions:

    print("\n----------------")
    print("Запрос:")
    print(q)


    result = db.similarity_search(
        q,
        k=1
    )


    if result:

        print(
            result[0].page_content[:500]
        )



print("\nГотово.")

import os
import re

from dotenv import load_dotenv

from telegram import Update

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI


load_dotenv()


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ROUTERAI_API_KEY = os.getenv("ROUTERAI_API_KEY")


print("Загрузка базы знаний...")


embeddings = HuggingFaceEmbeddings(

    model_name="intfloat/multilingual-e5-large",

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    }

)


db = Chroma(

    persist_directory="database",

    embedding_function=embeddings

)


print("База знаний загружена")


llm = ChatOpenAI(

    model="~deepseek/deepseek-v4-flash-latest",

    api_key=ROUTERAI_API_KEY,

    base_url="https://routerai.ru/api/v1",

    temperature=0.2

)


print("ИИ модель подключена")



def clean_text(text):

    # убираем номера разделов
    text = re.sub(
        r'\b\d+\.\d+(\.\d+)*\b',
        '',
        text
    )


    # убираем рисунки
    text = re.sub(
        r'рисунок\s*\d+(\.\d+)*',
        '',
        text,
        flags=re.IGNORECASE
    )


    # убираем таблицы
    text = re.sub(
        r'таблица\s*\d+(\.\d+)*',
        '',
        text,
        flags=re.IGNORECASE
    )


    # убираем страницы
    text = re.sub(
        r'стр\.?\s*\d+',
        '',
        text,
        flags=re.IGNORECASE
    )


    text = re.sub(
        r'\s+',
        ' ',
        text
    )


    return text.strip()



def get_roots(text):

    words = [
        x.lower().strip(".,!?")
        for x in text.split()
    ]


    return [
        w[:5]
        for w in words
        if len(w) >= 5
    ]



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
"""
Я ИИ-помощник по трактору БЕЛАРУС МТЗ-1523.

Задавайте вопросы:
- обслуживание;
- неисправности;
- регулировки;
- устройство;
- характеристики.

Можно написать:
"кратко про двигатель"
"подробно про безопасность"
"""
    )



async def answer(update: Update, context: ContextTypes.DEFAULT_TYPE):

    question = update.message.text.strip()


    await update.message.chat.send_action(
        action="typing"
    )


    try:


        results = db.similarity_search_with_score(
            question,
            k=30
        )



        # короткий запрос - поиск тем

        if len(question.split()) <= 2:


            roots = get_roots(question)

            variants = []


            for doc, score in results:


                for line in doc.page_content.split("\n"):


                    line = clean_text(line)


                    if len(line) < 20:
                        continue


                    low = line.lower()


                    if any(
                        r in low
                        for r in roots
                    ):

                        variants.append(line)



            variants = list(
                dict.fromkeys(variants)
            )


            if variants:


                await update.message.reply_text(

                    "Нашёл информацию по темам:\n\n"
                    +
                    "\n".join(
                        "- " + v[:180]
                        for v in variants[:8]
                    )
                    +
                    "\n\nНапишите, что именно рассказать."

                )

                return



        context_text = ""


        for doc, score in results:

            context_text += (

                "\n\n"
                +
                doc.page_content

            )



        prompt = f"""

Ты технический помощник по трактору
БЕЛАРУС МТЗ-1523.


Ответь пользователю по технической документации.


Правила:

- отвечай сразу по сути;
- не показывай номера разделов;
- не показывай страницы;
- не упоминай рисунки;
- не упоминай таблицы;
- не говори "согласно разделу";
- не описывай процесс поиска;
- используй только техническую информацию.


Если пользователь написал короткий запрос,
сначала объясни тему простыми словами.


Если информации недостаточно:
скажи это кратко.



Текст руководства:

{context_text}



Вопрос:

{question}

"""


        response = llm.invoke(prompt)


        answer = clean_text(
            response.content
        )


        await update.message.reply_text(
            answer
        )



    except Exception as e:

        print(e)


        await update.message.reply_text(
            "Не удалось обработать запрос. Попробуйте уточнить тему."
        )




app = Application.builder().token(
    TELEGRAM_TOKEN
).build()


app.add_handler(
    CommandHandler(
        "start",
        start
    )
)


app.add_handler(
    MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        answer
    )
)


print("Бот запущен")


app.run_polling()
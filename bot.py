import os
import re
import threading


from flask import Flask


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



TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_TOKEN"
)


ROUTERAI_API_KEY = os.getenv(
    "ROUTERAI_API_KEY"
)



PORT = int(
    os.environ.get(
        "PORT",
        10000
    )
)



if not TELEGRAM_TOKEN:

    raise Exception(
        "Нет TELEGRAM_TOKEN"
    )



if not ROUTERAI_API_KEY:

    raise Exception(
        "Нет ROUTERAI_API_KEY"
    )








web_app = Flask(__name__)



@web_app.route("/")
def home():

    return "BELARUS 1523 bot is running"




def run_web():

    web_app.run(

        host="0.0.0.0",

        port=PORT

    )








print(
    "Загрузка базы знаний..."
)



embeddings = HuggingFaceEmbeddings(

    model_name=
    "intfloat/multilingual-e5-base",

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



print(
    "База знаний загружена"
)








llm = ChatOpenAI(

    model=
    "deepseek/deepseek-v4-flash-latest",

    api_key=
    ROUTERAI_API_KEY,

    base_url=
    "https://routerai.ru/api/v1",

    temperature=0.2

)



print(
    "ИИ модель подключена"
)








def clean_text(text):


    # номера разделов

    text = re.sub(

        r'\b\d+(\.\d+)+\b',

        '',

        text

    )



    # рисунки

    text = re.sub(

        r'рисунок\s*\d+(\.\d+)*',

        '',

        text,

        flags=re.IGNORECASE

    )



    # таблицы

    text = re.sub(

        r'таблица\s*\d+(\.\d+)*',

        '',

        text,

        flags=re.IGNORECASE

    )



    # страницы

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

        x.lower().strip(
            ".,!?"
        )

        for x in text.split()

    ]


    return [

        w[:5]

        for w in words

        if len(w) >= 5

    ]








async def start(

        update: Update,

        context: ContextTypes.DEFAULT_TYPE

):


    await update.message.reply_text(

"""
Я ИИ-помощник по трактору БЕЛАРУС МТЗ-1523.

Помогу найти информацию по:

- обслуживанию;
- устройству;
- регулировкам;
- неисправностям;
- характеристикам.

Можно написать:

"кратко про двигатель"

"подробно про безопасность"
"""

    )








async def answer(

        update: Update,

        context: ContextTypes.DEFAULT_TYPE

):


    question = update.message.text.strip()



    await update.message.chat.send_action(

        action="typing"

    )



    try:


        results = db.similarity_search_with_score(

            question,

            k=30

        )



    


        if len(question.split()) <= 2:


            roots = get_roots(
                question
            )


            variants = []



            for doc, score in results:


                for line in doc.page_content.split("\n"):


                    line = clean_text(
                        line
                    )



                    if len(line) < 20:

                        continue



                    if any(

                        root in line.lower()

                        for root in roots

                    ):

                        variants.append(
                            line
                        )



            variants = list(

                dict.fromkeys(
                    variants
                )

            )



            if variants:


                await update.message.reply_text(

                    "Нашёл похожие темы:\n\n"

                    +

                    "\n".join(

                        "- " + x[:180]

                        for x in variants[:8]

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


Используй только техническую информацию
из руководства.


Правила:

- отвечай сразу по сути;
- не показывай номера разделов;
- не показывай страницы;
- не упоминай рисунки;
- не упоминай таблицы;
- не говори "в разделе";
- не рассказывай процесс поиска;
- не придумывай технические данные.


Если есть похожая информация,
используй её.


Если информации действительно нет:

скажи:

"В руководстве нет точного описания этого вопроса."


Материал:

{context_text}



Вопрос:

{question}

"""



        response = llm.invoke(
            prompt
        )



        await update.message.reply_text(

            clean_text(
                response.content
            )

        )



    except Exception as e:


        print(

            "Ошибка:",

            e

        )


        await update.message.reply_text(

            "Не удалось обработать запрос."

        )







telegram_app = Application.builder().token(

    TELEGRAM_TOKEN

).build()



telegram_app.add_handler(

    CommandHandler(

        "start",

        start

    )

)



telegram_app.add_handler(

    MessageHandler(

        filters.TEXT & ~filters.COMMAND,

        answer

    )

)





print(
    "Бот запускается"
)



threading.Thread(

    target=run_web,

    daemon=True

).start()



print(
    "Бот запущен"
)



telegram_app.run_polling()

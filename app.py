import streamlit as slt
from fastai.vision.all import *
import pathlib
import plotly.express as ax
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# streamlit run .\app.py
# title
slt.title("Telefon, qushlar va bosh kiyimlarni kassifikatsiya qilib beradi")

# rasmni joylash - ya'ni tugma qo'shish
file = slt.file_uploader("Rasm yuklash", type=['png', 'jpg', 'jpeg', 'gif'])
if file:
    slt.image(file)
    # PIL convert
    img = PILImage.create(file)

    # model
    model = load_learner("aralash_model.pkl")

    # prediction
    pred, pred_id, probs = model.predict(img)
    slt.success(f"Bashorat -> {pred}")
    slt.info(f"Ehtimollik  -> {probs[pred_id] * 100 :.1f} %")

    # plotting - ekranga ustun shaklida chiqarish
    fig = ax.bar(x=probs*100, y=model.dls.vocab)
    slt.plotly_chart(fig)
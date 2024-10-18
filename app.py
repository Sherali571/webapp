  
import streamlit as st

from fastai.vision.all import *
import pathlib
import plotly.express as pxn25
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Rasmlarni klassifikatsiya qilish")

st.write("Bu model bizga Mashina, O'yinchoq, Eshik, Ichimliklar, Telefon, To'p, Baliq, Ayiq, Qush, Daraxt, Samolyot, Kema, Qurol kabi rasmlarni bashorat qilib beradi.")

files=st.file_uploader("Rasm yuklash", type=["jpg","svg","png"])

if files:
    st.image(files, caption="Yuklangan rasim", width=300)
    
    # Save the uploaded file temporarily
    img = PILImage.create(files)
    
    # Load your pre-trained model (make sure the 'model.pkl' file is accessible)
    model = load_learner('model.pkl')
    
    # Make a prediction
    pred, pred_idx, probs = model.predict(img)
    
    # O'zbek tilida natijalarni chiqarish
    #Car Toy Door Drink Telephone Ball Fish Bear Bird Tree Airplane Boat Weapon
    tarjimalar = {
        'Car': 'Mashina',
        'Toy': 'O\'yinchoq',
        'Door': 'Eshik',
        'Drink': 'Ichimliklar',
        'Telephone': 'Telefon',
        'Ball': 'To\'p',
        'Fish': 'Baliq',
        'Bear': 'Ayiq',
        'Bird': 'Qush',
        'Tree': 'Daraxt',
        'Airplane': 'Samolyot',
        'Boat': 'Kema',
        'Weapon': 'Qurol',
        # boshqa klasslarni ham shu yerga qo'shishingiz mumkin
    }
    
    bashorat_uz = tarjimalar.get(str(pred), str(pred))  # Agar tarjima topilmasa, asl natijani chiqaradi
    
    # Display the prediction and the probability
    st.write(f"Bashorat: {bashorat_uz}")
    st.write(f"Ishonch darajasi: {probs[pred_idx]:.4f}")
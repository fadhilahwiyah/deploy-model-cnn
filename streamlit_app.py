import streamlit as st
!pip install tensorflow
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Judul halaman
st.title("Klasifikasi Citra Batik dengan CNN")

# Muat model CNN
model = tf.keras.models.load_model("model_batik.h5", compile=False)


# Ukuran input yang diharapkan model (ubah sesuai arsitektur modelmu)
IMAGE_SIZE = (224, 224)

# Daftar nama kelas sesuai urutan
nama_kelas = [
    "batik_betawi", "batik_bokor_kencono", "batik_buketan", "batik_dayak",
    "batik_jlamprang", "batik_kawung", "batik_liong", "batik_mega_mendung",
    "batik_parang", "batik_sekarjagad", "batik_sidoluhur", "batik_sidomukti",
    "batik_sidomulyo", "batik_singa_barong", "batik_srikaton", "batik_tribusono",
    "batik_tujuh_rupa", "batik_tuntrum", "batik_wahyu_tumurun", "batik_wirasat"
]

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar batik", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)

    # Preprocessing gambar
    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:  # jika gambar RGBA
        img_array = img_array[..., :3]  # konversi ke RGB
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    pred_nama = nama_kelas[pred_index]

    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi:")
    st.write(f"Nama Batik: **{pred_nama}** (kelas ke-{pred_index})")

    st.write("Probabilitas tiap kelas:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{nama_kelas[i]}: {prob:.4f}")

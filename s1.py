from huggingface_hub import hf_hub_download
import tensorflow
@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="Muthuswamy/Face-mask1",  # 👈 your path
        filename="model.h5"
    )
    return tensorflow.keras.models.load_model(path)


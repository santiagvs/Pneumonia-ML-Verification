# %% [markdown]
# # Projeto IA - Classificação de diagnóstico de pneumonia
# 
# Este projeto tem como objetivo a classificação de diagnóstico de pneumonia em imagens de raio-x de tórax. O dataset utilizado foi obtido no Kaggle e pode ser encontrado [aqui](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
# 
# O dataset é composto por 5.856 imagens de raio-x de tórax, sendo 4.232 imagens de pacientes diagnosticados com pneumonia e 1.624 imagens de pacientes saudáveis. As imagens estão divididas em 3 pastas: `train`, `test` e `val` (para fazer a validação), sendo que a pasta `train` contém 5.216 imagens, a pasta `test` contém 624 imagens e a pasta `val` contém 16 imagens.
# 
# O dataset utilizado teve validação de 3 especialistas clínicos para filtrar as imagens usadas e para certificar o que foi classificado em cada pasta como "NORMAL" ou "PNEUMONIA".
# 
# ## Rodar o projeto:
# 
# Faça a instalação do Python 3.10+ na máquina (recomendo usar o venv), instale o PIP para baixar as dependências e execute o comando abaixo para instalar as dependências necessárias:
# 
# ```bash
# pip install tensorflow keras scipy numpy matplotlib opencv-python kagglehub
# ```
# 
# Esse comando instalará o TensorFlow, Keras, SciPy (que serve como dependência do Keras), NumPy, Matplotlib, OpenCV e KaggleHub.

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub

path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset: ", path)

train_dir = f'{path}/chest_xray/train'
val_dir = f'{path}/chest_xray/val'
test_dir = f'{path}/chest_xray/test'

IMG_SIZE = 150

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='binary'
)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Saída binária: pneumonia ou não
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# %%
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# %%
test_loss, test_acc = model.evaluate(test_generator)
print(f'Acurácia no conjunto de teste: {test_acc * 100:.2f}%')

# %%
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Acurácia Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
# print(f"Quantidade de épocas: {len(history.history['accuracy'])}")
# for i in range(len(history.history['accuracy'])):
#     print(f"Valor da acurácia da época {i + 1}: {history.history['accuracy'][i]:.2f}")

print(f"\n======= Acurácia de validação =======")
for i in range(len(history.history['val_accuracy'])):
    print(f"Valor da acurácia da época {i + 1}: {history.history['val_accuracy'][i]:.2f}")
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='Perda Treino')
plt.plot(history.history['val_loss'], label='Perda Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend(loc='upper right')
plt.show()


# %%
import numpy as np
import json
import matplotlib.pyplot as plt

val_generator.reset()
predictions = model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1)

predicted_classes = (predictions > 0.5).astype(int).flatten()

true_labels = val_generator.classes

errors = np.where(predicted_classes != true_labels)[0]
print(f"Total de erros no conjunto de validação: {len(errors)}")

error_info = []

for idx in errors:
    image_path = val_generator.filepaths[idx]
    true_label = "PNEUMONIA" if true_labels[idx] == 1 else "NORMAL"
    predicted_label = "PNEUMONIA" if predicted_classes[idx] == 1 else "NORMAL"
    error_info.append({
        "image_path": image_path,
        "true_label": true_label,
        "predicted_label": predicted_label
    })

with open("validation_errors.json", "w") as f:
    json.dump(error_info, f, indent=4)

print("Informações sobre os erros salvas em 'validation_errors.json'.")

def plot_errors(errors, num_images=5):
    for i, error in enumerate(errors[:num_images]):
        image = plt.imread(error["image_path"])
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap='gray')
        plt.title(f"Real: {error['true_label']} | Previsto: {error['predicted_label']}")
        plt.axis('off')
        plt.show()

plot_errors(error_info)



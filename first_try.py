import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

X1 = np.random.randint(1, 10, size=200)  # 200 случайных чисел от 1 до 9
X2 = np.random.randint(1, 10, size=200)

# Целевое значение — сумма двух чисел
y_raw = X1 + X2

# Собираем X в матрицу (200 строк, 2 столбца)
X = np.vstack([X1, X2]).T          # форма: (200, 2)
y = y_raw[None].T                   # форма: (200, 1)

print('Форма X:', X.shape)          # должно быть (200, 2)
print('Форма y:', y.shape)          # должно быть (200, 1)
print('Первые 5 примеров:')
for i in range(5):
    print(f'  X1={X[i,0]}, X2={X[i,1]} → y={y[i,0]}')
    
# ── 3. Масштабирование ───────────────────────────────────────
# MinMaxScaler переводит каждый признак в диапазон [0, 1]
# Формула: x_new = (x - x_min) / (x_max - x_min)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_norm = scaler_X.fit_transform(X)  # масштабируем входы
y_norm = scaler_y.fit_transform(y)  # масштабируем выходы тоже!

# Важно: y тоже нужно масштабировать, потому что сеть предсказывает
# числа в диапазоне [0,1] (из-за масштабированных входов)

print('X до масштабирования:', X[0])
print('X после масштабирования:', X_norm[0].round(3))


# ── 4. Строим модель ─────────────────────────────────────────
tf.random.set_seed(9)

model = Sequential([
    # Первый (скрытый) слой: 3 нейрона, 2 входа
    # input_shape=(2,) означает: каждый пример имеет 2 числа
    Dense(3, input_shape=(2,), activation='linear'),
    
    # Выходной слой: 1 нейрон — он выдаёт финальный ответ
    Dense(1, activation='linear')
])

model.summary()  # покажет архитектуру и количество параметров

# ── 5. Компиляция и обучение ─────────────────────────────────
# loss='mse' — Mean Squared Error, считаем среднеквадратичную ошибку
# optimizer='adam' — алгоритм обновления весов (умный градиентный спуск)

model.compile(loss='mse', optimizer='adam')

# verbose=0 — не выводить прогресс в консоль (сохраним историю сами)
history = model.fit(
    X_norm, y_norm,
    epochs=300,          # 300 проходов по всем данным
    validation_split=0.2, # 20% данных оставить для проверки
    verbose=0
)

print('Обучение завершено!')
print(f'Финальная ошибка: {history.history["loss"][-1]:.6f}')

# ── 6. График обучения (как падает ошибка) ───────────────────
plt.figure(figsize=(10, 4))

plt.plot(history.history['loss'], label='Ошибка на обучении', color='blue')
plt.plot(history.history['val_loss'], label='Ошибка на проверке', color='orange')

plt.title('Как модель обучалась (чем ниже — тем лучше)')
plt.xlabel('Эпоха (проход по данным)')
plt.ylabel('Ошибка MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def predict(a, b):
    """Предсказать сумму a + b с помощью нейросети"""
    # Масштабируем входные данные так же, как при обучении
    inp = scaler_X.transform([[a, b]])
    # Получаем предсказание в масштабированном виде
    pred_norm = model.predict(inp, verbose=0)
    # Возвращаем в исходный масштаб
    pred = scaler_y.inverse_transform(pred_norm)
    return pred[0][0]

test_cases = [(3, 5), (7, 2), (9, 9), (1, 1)]
print('Тест предсказаний:')
print(f'{"Входы":<15} {"Правильно":<15} {"Нейросеть":<15} {"Ошибка"}')
print('-' * 55)
for a, b in test_cases:
    true_val = a + b
    predicted = predict(a, b)
    error = abs(true_val - predicted)
    print(f'{a} + {b} = {true_val:<10} {true_val:<15} {predicted:<15.2f} {error:.3f}')
    
# ── 8. Визуализация архитектуры с кружочками ─────────────────
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def draw_neural_net(layer_sizes, layer_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.axis('off')

    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    h_spacing = 2.5  # расстояние между слоями
    v_spacing = 1.2  # расстояние между нейронами

    neuron_positions = []
    
   # Рисуем нейроны
    for i, n in enumerate(layer_sizes):
        positions = []
        x = i * h_spacing
        y_start = (max_neurons - n) / 2 * v_spacing
        for j in range(n):
            y = y_start + j * v_spacing
            positions.append((x, y))
            color = '#4A90D9' if i == 0 else ('#E74C3C' if i == n_layers - 1 else '#2ECC71')
            circle = plt.Circle((x, y), 0.35, color=color, zorder=3)
            ax.add_patch(circle)
            label = f'X{j+1}' if i == 0 else ('y' if i == n_layers - 1 else f'h{j+1}')
            ax.text(x, y, label, ha='center', va='center', fontsize=9,
                    color='white', fontweight='bold', zorder=4)
        neuron_positions.append(positions)

    # Рисуем связи (стрелки = веса)
    for i in range(n_layers - 1):
        for pos1 in neuron_positions[i]:
            for pos2 in neuron_positions[i + 1]:
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                        'gray', alpha=0.4, linewidth=0.8, zorder=1)

    # Подписи слоёв
    for i, name in enumerate(layer_names):
        x = i * h_spacing
        ax.text(x, -0.7, name, ha='center', va='top', fontsize=10,
                fontweight='bold', color='#333')

    # Легенда
    legend_elements = [
        mpatches.Patch(color='#4A90D9', label='Входной слой (2 нейрона)'),
        mpatches.Patch(color='#2ECC71', label='Скрытый слой (3 нейрона)'),
        mpatches.Patch(color='#E74C3C', label='Выходной слой (1 нейрон)'),
    ]
    ax.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9)

    ax.set_title('Архитектура нейросети: 2 → 3 → 1', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()

# Рисуем: входной слой (2), скрытый (3), выходной (1)
draw_neural_net(
    layer_sizes=[2, 3, 1],
    layer_names=['Вход\n(X1, X2)', 'Скрытый слой\n(3 нейрона)', 'Выход\n(y = X1+X2)']
)

# ── 9. График: предсказания vs реальные значения ─────────────
# Это показывает насколько хорошо сеть научилась

y_pred_norm = model.predict(X_norm, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_norm)  # возвращаем в исходный масштаб

plt.figure(figsize=(7, 6))
plt.scatter(y, y_pred, alpha=0.4, color='steelblue', s=20, label='Предсказания')

# Идеальная линия — если бы предсказания были идеальными
min_val, max_val = y.min(), y.max()
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеал')

plt.xlabel('Реальные значения (X1 + X2)')
plt.ylabel('Предсказания нейросети')
plt.title('Реальные vs Предсказанные значения\n(точки на линии = идеальное предсказание)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


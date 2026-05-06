import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns

# Dados do dataset
data = {
    'lightness': [2.834754098360656, 3.329180327868852, 3.6904918032786886, 4.812459016393442, 4.812459016393442, 
                  4.92655737704918, 1.2563934426229508, 1.63672131147541, 1.9980327868852457, 2.872786885245902,
                  3.2340983606557376, 3.861639344262296, 2.093114754098361, 2.093114754098361, 2.7016393442622952,
                  2.834754098360656, 2.834754098360656, 2.093114754098361, 2.207213114754098, 2.3403278688524587,
                  3.081967213114754, 0.8, 1.4655737704918033, 1.598688524590164, 3.329180327868852, 3.4432786885245905,
                  3.576393442622951, 3.785573770491803, 3.8996721311475406, 4.14688524590164, 4.14688524590164,
                  4.7744262295081965, 6.6, 1.2563934426229508, 3.481311475409836, 4.14688524590164, 3.614426229508197,
                  1.3324590163934429, 1.4655737704918033, 0.9331147540983606, 1.63672131147541, 2.055081967213115,
                  2.055081967213115, 2.663606557377049, 2.872786885245902, 2.9868852459016395, 3.12, 3.12,
                  3.2340983606557376, 3.3672131147540982, 3.4052459016393444, 3.975737704918033, 4.470163934426229,
                  4.60327868852459, 4.717377049180328, 5.097704918032787, 5.268852459016393, 5.592131147540983,
                  6.010491803278688, 6.3337704918032784, 4.432131147540983, 4.717377049180328, 2.663606557377049,
                  5.135737704918033, 3.4052459016393444, 2.5875409836065573, 3.576393442622951, 4.070819672131147,
                  4.5652459016393445, 5.173770491803278, 5.915409836065574, 1.1232786885245902, 1.2563934426229508,
                  1.3704918032786886, 5.74, 6.469677419354839, 6.581935483870968, 7.0683870967741935, 7.199354838709677,
                  7.929032258064517, 8.284516129032259, 8.284516129032259, 8.284516129032259, 8.770967741935484,
                  5.16, 7.723225806451614, 7.835483870967742, 8.078709677419354, 8.078709677419354, 7.798064516129033,
                  8.284516129032259, 7.0683870967741935, 7.274193548387098, 4.0, 6.787741935483871, 6.993548387096775,
                  7.610967741935484, 8.003870967741936, 8.134838709677421, 4.636129032258065, 4.972903225806451,
                  5.16, 5.646451612903226, 5.945806451612904, 6.394838709677419, 6.638064516129033, 6.787741935483871,
                  6.918709677419355, 6.993548387096775, 7.0683870967741935, 7.0683870967741935, 7.236774193548388,
                  7.367741935483872, 7.5922580645161295, 7.723225806451614, 7.760645161290324, 8.452903225806452,
                  9.538064516129031, 4.767096774193549, 5.571612903225807, 6.058064516129033, 6.095483870967742,
                  6.338709677419355, 7.274193548387098, 7.311612903225807, 7.51741935483871, 8.976774193548387,
                  5.365806451612904, 6.226451612903226, 8.789677419354838, 9.8],
    'width': [21.087142857142855, 18.877142857142857, 19.824285714285715, 17.759999999999998, 16.497142857142855,
              16.181428571428572, 15.55, 20.14, 19.97, 18.24571428571429, 18.707142857142856, 19.192857142857143,
              17.395714285714284, 15.962857142857144, 17.395714285714284, 18.974285714285717, 16.27857142857143,
              20.67428571428572, 19.24142857142857, 18.294285714285717, 20.504285714285714, 20.67428571428572,
              14.7, 21.13571428571429, 19.605714285714285, 18.027142857142856, 17.225714285714286, 16.545714285714286,
              18.44, 17.662857142857142, 17.347142857142856, 17.03142857142857, 16.084285714285713, 21.5,
              14.991428571428573, 18.707142857142856, 15.282857142857145, 21.087142857142855, 16.497142857142855,
              18.658571428571427, 19.38714285714285, 18.512857142857143, 16.764285714285712, 19.46, 19.70285714285714,
              16.764285714285712, 17.80857142857143, 16.91, 15.647142857142857, 16.861428571428572, 20.237142857142857,
              16.23, 16.545714285714286, 15.185714285714283, 18.124285714285712, 14.7, 15.04, 15.647142857142857,
              18.82857142857143, 16.667142857142856, 20.334285714285716, 19.217142857142857, 20.285714285714285,
              19.654285714285717, 17.08, 15.307142857142855, 16.254285714285714, 17.201428571428572, 15.768571428571429,
              16.715714285714284, 16.254285714285714, 19.92142857142857, 17.395714285714284, 18.34285714285714,
              17.551957295373665, 20.891459074733092, 17.69928825622776, 18.828825622775803, 18.97615658362989,
              19.14804270462633, 19.786476868327405, 17.551957295373665, 15.955871886120995, 14.998220640569397,
              18.779715302491105, 17.502846975088968, 19.565480427046264, 19.09893238434164, 18.46049822064057,
              17.920284697508897, 19.516370106761567, 16.962633451957295, 18.01850533807829, 17.69928825622776,
              19.66370106761566, 21.529893238434163, 16.324199288256228, 16.790747330960855, 20.62135231316726,
              18.75516014234876, 16.8644128113879, 19.982918149466197, 18.8779359430605, 19.418149466192173,
              21.8, 15.955871886120995, 18.28861209964413, 19.09893238434164, 20.154804270462638, 19.393594306049824,
              18.43594306049822, 17.551957295373665, 20.42491103202847, 18.23950177935943, 18.558718861209965,
              21.33345195729537, 18.23950177935943, 20.30213523131673, 19.982918149466197, 20.71957295373665,
              18.19039145907473, 20.64590747330961, 18.8779359430605, 20.891459074733092, 14.9, 19.46725978647687,
              18.65693950177936, 21.308896797153025, 20.20391459074733, 16.054092526690393, 17.06085409252669],
    'species': [0]*73 + [1]*57
}

# Criar DataFrame
df = pd.DataFrame(data)

# Preparar os dados
X = df[['lightness', 'width']].values
y = df['species'].values

# Normalizar os dados (importante para o Perceptron)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std

# Adicionar bias (coluna de 1s)
X_with_bias = np.c_[np.ones(X_normalized.shape[0]), X_normalized]

# Implementação do Perceptron
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.errors = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for _ in range(self.n_iterations):
            error_count = 0
            for idx, x_i in enumerate(X):
                # Predição linear
                linear_output = np.dot(x_i, self.weights)
                y_predicted = self._activation(linear_output)
                
                # Atualização dos pesos
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                error_count += int(update != 0.0)
            
            self.errors.append(error_count)
            
            # Critério de parada antecipada
            if error_count == 0:
                break
    
    def _activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights)
        return np.array([self._activation(x) for x in linear_output])

# Treinar o Perceptron
perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
perceptron.fit(X_with_bias, y)

# Fazer predições
y_pred = perceptron.predict(X_with_bias)

# Calcular métricas
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

# Matriz de confusão
cm = confusion_matrix(y, y_pred)

# Visualizar resultados
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Matriz de Confusão
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Matriz de Confusão')
axes[0, 0].set_xlabel('Predito')
axes[0, 0].set_ylabel('Real')
axes[0, 0].set_xticklabels(['Salmão (0)', 'Robalo (1)'])
axes[0, 0].set_yticklabels(['Salmão (0)', 'Robalo (1)'])

# 2. Superfície de Decisão
x_min, x_max = X_normalized[:, 0].min() - 1, X_normalized[:, 0].max() + 1
y_min, y_max = X_normalized[:, 1].min() - 1, X_normalized[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Preparar dados do meshgrid para predição
mesh_points = np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()]
Z = perceptron.predict(mesh_points)
Z = Z.reshape(xx.shape)

# Plotar superfície de decisão
axes[0, 1].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
scatter = axes[0, 1].scatter(X_normalized[:, 0], X_normalized[:, 1], 
                            c=y, cmap='RdYlBu', edgecolor='black', s=50)
axes[0, 1].set_xlabel('Lightness (normalizado)')
axes[0, 1].set_ylabel('Width (normalizado)')
axes[0, 1].set_title('Superfície de Decisão do Perceptron')
legend1 = axes[0, 1].legend(*scatter.legend_elements(), title="Espécie")
axes[0, 1].add_artist(legend1)

# 3. Métricas em um gráfico de barras
metrics = [accuracy, precision, recall]
metric_names = ['Acurácia', 'Precisão', 'Revocação']
bars = axes[1, 0].bar(metric_names, metrics, color=['blue', 'green', 'red'])
axes[1, 0].set_ylim([0, 1])
axes[1, 0].set_ylabel('Valor')
axes[1, 0].set_title('Métricas de Desempenho')

# Adicionar valores nas barras
for bar, value in zip(bars, metrics):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{value:.3f}', ha='center', va='bottom')

# 4. Convergência do Perceptron
axes[1, 1].plot(range(1, len(perceptron.errors) + 1), perceptron.errors, 
               marker='o', linestyle='-', color='purple')
axes[1, 1].set_xlabel('Épocas')
axes[1, 1].set_ylabel('Número de Erros')
axes[1, 1].set_title('Convergência do Perceptron')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Imprimir métricas
print("="*50)
print("RESULTADOS DO PERCEPTRON PARA CLASSIFICAÇÃO DE PEIXES")
print("="*50)
print(f"Acurácia:  {accuracy:.4f}")
print(f"Precisão:  {precision:.4f}")
print(f"Revocação: {recall:.4f}")
print("\nMatriz de Confusão:")
print(pd.DataFrame(cm, 
                   index=['Real: Salmão', 'Real: Robalo'],
                   columns=['Pred: Salmão', 'Pred: Robalo']))
print(f"\nNúmero de épocas para convergir: {len(perceptron.errors)}")
print(f"Pesos finais: {perceptron.weights}")

# Visualização adicional: Dados originais com classificação
fig, ax = plt.subplots(figsize=(10, 6))

# Plotar pontos classificados corretamente e incorretamente
correct = y_pred == y
incorrect = y_pred != y

# Salmão (classe 0)
salmon_mask = y == 0
ax.scatter(X[salmon_mask & correct, 0], X[salmon_mask & correct, 1], 
          c='blue', marker='o', s=80, label='Salmão (correto)', edgecolors='black')
ax.scatter(X[salmon_mask & incorrect, 0], X[salmon_mask & incorrect, 1], 
          c='blue', marker='x', s=100, label='Salmão (incorreto)', linewidths=2)

# Robalo (classe 1)
seabass_mask = y == 1
ax.scatter(X[seabass_mask & correct, 0], X[seabass_mask & correct, 1], 
          c='red', marker='s', s=80, label='Robalo (correto)', edgecolors='black')
ax.scatter(X[seabass_mask & incorrect, 0], X[seabass_mask & incorrect, 1], 
          c='red', marker='x', s=100, label='Robalo (incorreto)', linewidths=2)

ax.set_xlabel('Lightness')
ax.set_ylabel('Width')
ax.set_title('Classificação dos Peixes - Dados Originais')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.show()
import numpy as np

# Definição da função de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Função que ajusta os valores para o intervalo (0, 1)

def sigmoid_derivada(x):
    return x * (1 - x)  # Variação da função sigmoid

def relu(x):
    return np.maximum(0, x)  # Função ReLU retorna 0 para valores negativos

def relu_derivada(x):
    return np.where(x > 0, 1, 0)  # Variação da ReLU (1 para positivos, 0 para negativos)

# Classe para criar e gerenciar a Rede Neural
class NeuralNetwork:
    def __init__(self, quantidade_neuronios, camada_oculta_neuronios, quant_saida_neuronios, tipo_funcao_ativacao='sigmoid'):
        """
        Configura os parâmetros iniciais da rede neural.
        parametro quantidade_neuronios: Quantidade de neurônios na camada de entrada.
        parametro camada_oculta_neuronios: Quantidade de neurônios na camada oculta.
        parametro quant_saida_neuronios: Quantidade de neurônios na camada de saída.
        parametro tipo_funcao_ativacao: Tipo de função de ativação ('sigmoid' ou 'relu').
        """
        # Iniciando os pesos e os bias aleatorios
        self.pesos_entrada_oculta = np.random.uniform(-1, 1, (quantidade_neuronios, camada_oculta_neuronios))
        self.bias_ocultas = np.random.uniform(-1, 1, camada_oculta_neuronios)
        self.pesos_saida_oculta = np.random.uniform(-1, 1, (camada_oculta_neuronios, quant_saida_neuronios))
        self.bias_saida = np.random.uniform(-1, 1, quant_saida_neuronios)

        # Definindo a função de ativação
        if tipo_funcao_ativacao == 'sigmoid':
            self.ativacao_funcao = sigmoid
            self.derivada_ativacao = sigmoid_derivada
        elif tipo_funcao_ativacao == 'relu':
            self.ativacao_funcao = relu
            self.derivada_ativacao = relu_derivada
        else:
            raise ValueError("Escolha uma função de ativação: 'sigmoid' ou 'relu'.")

    def forward_pass(self, inputs):
        """
        Executa o cálculo de saída da rede (forward pass).
        parametro inputs: Dados de entrada.
        returno: Saída da camada oculta e saída final.
        """
        # Processamento da camada oculta
        entrada_camada_oculta = np.dot(inputs, self.pesos_entrada_oculta) + self.bias_ocultas
        saida_camada_oculta = self.ativacao_funcao(entrada_camada_oculta)

        # Processamento da camada de saída
        final_layer_input = np.dot(saida_camada_oculta, self.pesos_saida_oculta) + self.bias_saida
        saida_final = self.ativacao_funcao(final_layer_input)

        return saida_camada_oculta, saida_final

    def backpropagation(self, inputs, saida_esperada, saida_oculta, saida_final, aprendizado):
        """
        Ajusta os pesos e os bias com base nos erros calculados.
        parametros inputs: Dados de entrada.
        parametros saida_esperada: Saídas esperadas.
        parametros saida_oculta: Saídas da camada oculta.
        parametros saida_final: Saída final calculada.
        parametros aprendizado: Taxa de aprendizado.
        """
        # Calcula o erro na camada de saída
        output_error = saida_esperada - saida_final
        delta_output = output_error * self.derivada_ativacao(saida_final)

        # Calcula o erro na camada oculta
        hidden_error = np.dot(delta_output, self.pesos_saida_oculta.T)
        delta_hidden = hidden_error * self.derivada_ativacao(saida_oculta)

        # Atualiza os pesos e os bias (camada oculta -> saída)
        self.pesos_saida_oculta += aprendizado * np.dot(saida_oculta.T, delta_output)
        self.bias_saida += aprendizado * delta_output.sum(axis=0)

        # Atualiza os pesos e os bias (entrada -> camada oculta)
        self.pesos_entrada_oculta += aprendizado * np.dot(inputs.T, delta_hidden)
        self.bias_ocultas += aprendizado * delta_hidden.sum(axis=0)

    def train(self, treinamento_entrada, treinamento_saida, aprendizado, ciclos):
        """
        Treina a rede com os exemplos fornecidos.
        parametros treinamento_entrada: Dados de entrada para treinamento.
        parametros treinamento_saida: Saídas esperadas para treinamento.
        parametros aprendizado: Taxa de aprendizado.
        parametros ciclos: Número de ciclos de treinamento.
        """
        for epoch in range(ciclos):
            total_error = 0
            for inputs, expected in zip(treinamento_entrada, treinamento_saida):
                inputs = inputs.reshape(1, -1)
                expected = expected.reshape(1, -1)

                # Passagem direta (Forward pass)
                saida_oculta, saida_final = self.forward_pass(inputs)

                # Passagem reversa (Backpropagation)
                self.backpropagation(inputs, expected, saida_oculta, saida_final, aprendizado)

                # Calculando o erro total 
                total_error += np.sum((expected - saida_final) ** 2)

            # Exibindo o erro
            if epoch % 1000 == 0:
                print(f"Época {epoch}, Erro Total: {total_error:.4f}")

    def predict(self, inputs):
        """
        Faz previsões com base nos ajustes dos pesos.
        parametros inputs: Dados para a previsão.
        returno: Resultado previsto.
        """
        _, saida_final = self.forward_pass(inputs)
        return saida_final

# Testando a Rede Neural com XOR
if __name__ == "__main__":
    # Tabela verdade XOR (dados de entrada e saídas esperadas)
    xor_entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_saidas = np.array([[0], [1], [1], [0]])

    # Configurando e treinando a rede neural
    trabalho_neural = NeuralNetwork(quantidade_neuronios=2, camada_oculta_neuronios=2, quant_saida_neuronios=1, tipo_funcao_ativacao='sigmoid')
    trabalho_neural.train(treinamento_entrada=xor_entradas, treinamento_saida=xor_saidas, aprendizado=0.1, ciclos=10000)

    # Testando a rede neural após o treinamento
    print("\nResultados após treinamento:")
    for test_input in xor_entradas:
        previsao_saida = trabalho_neural.predict(test_input.reshape(1, -1))
        print(f"Entrada: {test_input}, Saída Prevista: {previsao_saida.flatten()[0]:.4f}")

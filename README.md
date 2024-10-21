# Perceptron Simples com Regra de Aprendizado Delta

Este projeto implementa um Perceptron de camada única que utiliza a **Regra de Aprendizado Delta** para resolver o problema das portas lógicas OR, que é linearmente separável. Utilizamos a função sigmoide como função de ativação e ajustamos os pesos com base na derivada da sigmoide para melhorar a convergência.

## Descrição

O Perceptron é um algoritmo de aprendizado supervisionado utilizado para resolver problemas de classificação binária. No caso deste projeto, ele é treinado para aprender a resolver a porta lógica OR.

### Problemas resolvidos:

- **Porta OR**: O perceptron aprenderá a classificar corretamente as saídas para o problema da porta OR, onde as entradas são valores binários 0 e 1.

### O que é a Regra Delta?

A Regra Delta é uma técnica usada para ajustar os pesos de um perceptron durante o processo de aprendizado. O objetivo é minimizar o erro entre a saída desejada (dada pelo professor) e a saída calculada pelo perceptron. Neste projeto, usamos a função sigmoide para mapear os valores de entrada, e ajustamos os pesos conforme a seguinte fórmula:

\[
W^{k+1} = W^k + crx
\]

Onde:

- \( W^k \) é o vetor de pesos atual.
- \( c \) é a taxa de aprendizado.
- \( r \) é o sinal de aprendizado, que depende do erro e da derivada da função sigmoide.
- \( x \) são as entradas de dados.

## Funcionalidades

1. **Função de Ativação Sigmoide**: Usamos a função sigmoide para mapear as entradas do perceptron. A função sigmoide retorna valores entre -1 e 1.
2. **Atualização de Pesos pela Regra Delta**: A cada iteração, os pesos são ajustados com base no erro entre a saída desejada e a saída real.

3. **Critério de Parada**: O treinamento para quando o erro total cai abaixo de um valor desejado ou quando o número máximo de iterações é atingido.

4. **Treinamento para Porta OR**: A rede neural é treinada para resolver o problema da porta OR, onde:
   - Entrada: (1, 1) → Saída: 1
   - Entrada: (1, 0) → Saída: 1
   - Entrada: (0, 1) → Saída: 1
   - Entrada: (0, 0) → Saída: 0

## Estrutura do Código

### Arquivo: `brother.py`

- **Função Sigmoide (`sigmoid`)**: A função sigmoide transforma os valores de entrada e gera uma saída entre -1 e 1.

- **Derivada da Função Sigmoide (`sigmoid_derivative`)**: Usada para calcular a correção dos pesos durante o treinamento.

- **Função de Saída (`findOutput`)**: Calcula a saída do perceptron com base no vetor de pesos atual e nas entradas fornecidas.

- **Conjunto de Dados**:

  - As entradas para o problema da porta OR estão na matriz `p`.
  - As saídas desejadas (verdadeiras) estão no vetor `d`.

- **Treinamento**:

  - O perceptron passa por várias iterações onde o erro é calculado e os pesos são ajustados com base na Regra Delta.

- **Teste**:
  - Após o treinamento, o perceptron é testado com as entradas da porta OR para verificar se aprendeu a classificação correta.

## Como Usar

### Pré-requisitos:

- Python 3.x
- Pacote `numpy`
- Pacote `matplotlib` (opcional para visualização)

### Instalação de Dependências

Você pode instalar as dependências usando o seguinte comando:

```bash
pip install numpy matplotlib
```

### Execução

Para rodar o código, execute o seguinte comando no terminal:

```bash
python3 main.py
```

### Resultado Esperado

Após o treinamento, o Perceptron deve classificar corretamente as entradas da porta OR. O modelo exibirá o erro total a cada iteração e os pesos ajustados. Além disso, ele imprimirá o resultado para cada teste de entrada.

Por exemplo:

```plaintext
Testes com a rede treinada (OR)
0.987...  # Para entrada [1, 1, -1]
0.956...  # Para entrada [1, -1, -1]
0.954...  # Para entrada [-1, 1, -1]
-0.987...  # Para entrada [-1, -1, -1]
```

Esses valores devem estar próximos de 1 ou -1, indicando a classificação correta.

## Licença

Este projeto está sob a licença MIT.

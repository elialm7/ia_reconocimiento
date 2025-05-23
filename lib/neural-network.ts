// Implementación simplificada de un perceptrón multicapa
export class NeuralNetwork {
  private weights: number[][][] = []
  private biases: number[][] = []
  private layers: number[] = []

  constructor(layers: number[]) {
    this.layers = layers

    // Inicializar pesos y sesgos con valores aleatorios
    for (let i = 0; i < layers.length - 1; i++) {
      const inputSize = layers[i]
      const outputSize = layers[i + 1]

      const layerWeights: number[][] = []
      const layerBiases: number[] = []

      // Inicializar pesos para cada neurona en la capa
      for (let j = 0; j < outputSize; j++) {
        const neuronWeights: number[] = []

        // Inicializar pesos para cada conexión de entrada
        for (let k = 0; k < inputSize; k++) {
          // Inicialización Xavier/Glorot
          const stddev = Math.sqrt(2.0 / (inputSize + outputSize))
          neuronWeights.push(randomNormal(0, stddev))
        }

        layerWeights.push(neuronWeights)
        layerBiases.push(0) // Inicializar sesgos en cero
      }

      this.weights.push(layerWeights)
      this.biases.push(layerBiases)
    }

    console.log("Modelo inicializado con capas:", layers)
    console.log(
      "Estructura de pesos:",
      this.weights.map((layer) => layer.length),
    )
  }

  // ReLU activation function
  private relu(x: number): number {
    return Math.max(0, x)
  }

  // Derivative of ReLU for backpropagation
  private reluDerivative(x: number): number {
    return x > 0 ? 1 : 0
  }

  // Softmax function for the output layer
  private softmax(x: number[]): number[] {
    // Restar el máximo para estabilidad numérica
    const maxVal = Math.max(...x)
    const expValues = x.map((val) => Math.exp(val - maxVal))
    const sumExp = expValues.reduce((acc, val) => acc + val, 0)

    // Evitar división por cero
    if (sumExp === 0) {
      return x.map(() => 1 / x.length) // Distribución uniforme
    }

    return expValues.map((val) => val / sumExp)
  }

  // Forward pass
  private forward(input: number[]): { activations: number[][]; outputs: number[][] } {
    // Validar entrada
    if (input.length !== this.layers[0]) {
      console.error(`Error de dimensión: La entrada tiene ${input.length} elementos, se esperaban ${this.layers[0]}`)
      return { activations: [], outputs: [] }
    }

    const activations: number[][] = [input]
    const outputs: number[][] = []

    // Iterar a través de cada capa
    for (let i = 0; i < this.weights.length; i++) {
      const layerOutput: number[] = []
      const layerWeights = this.weights[i]
      const layerBiases = this.biases[i]
      const prevActivation = activations[activations.length - 1]

      // Calcular salida para cada neurona en esta capa
      for (let j = 0; j < layerWeights.length; j++) {
        const neuronWeights = layerWeights[j]
        let sum = layerBiases[j]

        // Sumar productos ponderados de entradas y pesos
        for (let k = 0; k < neuronWeights.length; k++) {
          sum += neuronWeights[k] * prevActivation[k]
        }

        layerOutput.push(sum)
      }

      outputs.push(layerOutput)

      // Aplicar función de activación
      let activation: number[]
      if (i === this.weights.length - 1) {
        // Capa de salida: softmax
        activation = this.softmax(layerOutput)
      } else {
        // Capas ocultas: ReLU
        activation = layerOutput.map(this.relu)
      }

      activations.push(activation)
    }

    return { activations, outputs }
  }

  // Backpropagation
  private backward(input: number[], target: number[], learningRate: number): number {
    const { activations, outputs } = this.forward(input)

    if (activations.length === 0) {
      console.error("Forward pass falló durante el backpropagation")
      return 1.0 // Valor alto de pérdida
    }

    // Calcular error de capa de salida (derivada de la pérdida cross-entropy)
    const outputActivations = activations[activations.length - 1]
    const outputDeltas: number[] = []

    // Calcular pérdida cross-entropy y deltas
    let loss = 0
    for (let i = 0; i < outputActivations.length; i++) {
      // Calcular pérdida cross-entropy
      if (target[i] > 0) {
        loss -= Math.log(Math.max(outputActivations[i], 1e-15)) * target[i]
      }

      // Delta para softmax con cross-entropy es simplemente (output - target)
      outputDeltas.push(outputActivations[i] - target[i])
    }

    // Array de arrays para almacenar deltas de todas las capas (empezando por la salida)
    const deltas: number[][] = [outputDeltas]

    // Retropropagar errores
    for (let i = this.weights.length - 1; i >= 0; i--) {
      const currentDeltas = deltas[0]
      const layerWeights = this.weights[i]

      // Si no es la primera capa, calcular deltas para la capa anterior
      if (i > 0) {
        const prevOutput = outputs[i - 1]
        const prevDeltas: number[] = []

        // Para cada neurona en la capa anterior
        for (let j = 0; j < prevOutput.length; j++) {
          let delta = 0

          // Sumar contribuciones de todas las neuronas en la capa actual
          for (let k = 0; k < currentDeltas.length; k++) {
            delta += currentDeltas[k] * layerWeights[k][j]
          }

          // Multiplicar por la derivada de la activación
          delta *= this.reluDerivative(prevOutput[j])
          prevDeltas.push(delta)
        }

        deltas.unshift(prevDeltas)
      }

      // Actualizar pesos y sesgos
      const layerInput = activations[i]

      for (let j = 0; j < layerWeights.length; j++) {
        // Actualizar sesgo
        this.biases[i][j] -= learningRate * currentDeltas[j]

        // Actualizar pesos
        for (let k = 0; k < layerWeights[j].length; k++) {
          this.weights[i][j][k] -= learningRate * currentDeltas[j] * layerInput[k]
        }
      }
    }

    return loss
  }

  // Entrenar la red con un lote de ejemplos
  async train(inputs: Float32Array[], targets: Float32Array[], learningRate = 0.01): Promise<{ loss: number }> {
    if (inputs.length === 0 || targets.length === 0) {
      return { loss: 0 }
    }

    let totalLoss = 0

    for (let i = 0; i < inputs.length; i++) {
      const input = Array.from(inputs[i])
      const target = Array.from(targets[i])
      const loss = this.backward(input, target, learningRate)
      totalLoss += loss
    }

    return { loss: totalLoss / inputs.length }
  }

  // Evaluar la red en un conjunto de prueba
  async evaluate(inputs: Float32Array[], targets: Float32Array[]): Promise<number> {
    if (inputs.length === 0) return 0

    let correct = 0

    for (let i = 0; i < inputs.length; i++) {
      const input = Array.from(inputs[i])
      const target = Array.from(targets[i])

      const { prediction } = await this.predict(input)
      const trueLabel = target.indexOf(Math.max(...target))

      if (prediction === trueLabel) {
        correct++
      }
    }

    return correct / inputs.length
  }

  // Realizar una predicción
  async predict(input: Float32Array | number[]): Promise<{ prediction: number; confidences: number[] }> {
    try {
      // Convertir a array regular si es Float32Array
      const inputArray = Array.isArray(input) ? input : Array.from(input)

      // Verificar si hay suficientes valores no cero
      const nonZeroValues = inputArray.filter((val) => val > 0.01).length
      if (nonZeroValues < 5) {
        console.log("Entrada insuficiente: demasiados ceros", nonZeroValues)
        return {
          prediction: 0,
          confidences: Array(10).fill(0.1),
        }
      }

      // Normalizar entrada si es necesario
      const maxVal = Math.max(...inputArray)
      const normalizedInput = maxVal > 1 ? inputArray.map((val) => val / maxVal) : inputArray

      // Realizar forward pass
      const { activations } = this.forward(normalizedInput)

      if (activations.length === 0) {
        console.error("Forward pass falló durante la predicción")
        return {
          prediction: 0,
          confidences: Array(10).fill(0.1),
        }
      }

      // Obtener activaciones de salida
      const outputActivations = activations[activations.length - 1]

      // Validar salida
      if (!outputActivations || outputActivations.length === 0) {
        console.error("No hay activaciones de salida válidas")
        return {
          prediction: 0,
          confidences: Array(10).fill(0.1),
        }
      }

      // Encontrar la clase con mayor probabilidad
      let maxIdx = 0
      let maxVal2 = outputActivations[0]

      for (let i = 1; i < outputActivations.length; i++) {
        if (outputActivations[i] > maxVal2) {
          maxVal2 = outputActivations[i]
          maxIdx = i
        }
      }

      // Verificar NaN o valores inválidos
      for (let i = 0; i < outputActivations.length; i++) {
        if (isNaN(outputActivations[i]) || !isFinite(outputActivations[i])) {
          console.error("Valores de salida inválidos detectados")
          return {
            prediction: 0,
            confidences: Array(10).fill(0.1),
          }
        }
      }

      return {
        prediction: maxIdx,
        confidences: outputActivations,
      }
    } catch (error) {
      console.error("Error en predicción:", error)
      return {
        prediction: 0,
        confidences: Array(10).fill(0.1),
      }
    }
  }
}

// Función de utilidad para generar números aleatorios con distribución normal
function randomNormal(mean: number, stddev: number): number {
  const u = 1 - Math.random()
  const v = 1 - Math.random()
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
  return z * stddev + mean
}

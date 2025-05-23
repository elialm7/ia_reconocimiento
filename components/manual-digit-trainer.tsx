"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Loader2, Trash2, Save, Play, Plus, RefreshCw } from "lucide-react"

// Modelo simplificado de red neuronal
class SimpleNeuralNetwork {
  private weights: number[][] = []
  private biases: number[] = []
  private learningRate = 0.1

  constructor() {
    // Inicializar con pesos aleatorios para 10 clases (dígitos 0-9)
    // Entrada: 784 píxeles (28x28), Salida: 10 clases
    this.weights = Array(10)
      .fill(0)
      .map(() =>
        Array(784)
          .fill(0)
          .map(() => (Math.random() - 0.5) * 0.01),
      )
    this.biases = Array(10).fill(0)
  }

  // Función para calcular la puntuación para cada clase
  private calculateScores(input: number[]): number[] {
    const scores = Array(10).fill(0)

    // Para cada clase (dígito)
    for (let i = 0; i < 10; i++) {
      let sum = this.biases[i]

      // Sumar productos de pesos y entradas
      for (let j = 0; j < input.length; j++) {
        sum += this.weights[i][j] * input[j]
      }

      scores[i] = sum
    }

    return scores
  }

  // Función softmax para convertir puntuaciones en probabilidades
  private softmax(scores: number[]): number[] {
    const maxScore = Math.max(...scores)
    const expScores = scores.map((score) => Math.exp(score - maxScore))
    const sumExp = expScores.reduce((a, b) => a + b, 0)

    return expScores.map((exp) => exp / sumExp)
  }

  // Predecir la clase para una entrada
  predict(input: number[]): { prediction: number; confidences: number[] } {
    const scores = this.calculateScores(input)
    const probabilities = this.softmax(scores)

    // Encontrar la clase con mayor probabilidad
    let maxIndex = 0
    let maxProb = probabilities[0]

    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i]
        maxIndex = i
      }
    }

    return {
      prediction: maxIndex,
      confidences: probabilities,
    }
  }

  // Entrenar el modelo con un ejemplo
  train(input: number[], targetDigit: number): { loss: number } {
    // Calcular predicción actual
    const scores = this.calculateScores(input)
    const probabilities = this.softmax(scores)

    // Calcular pérdida (cross-entropy para la clase correcta)
    const loss = -Math.log(Math.max(probabilities[targetDigit], 1e-15))

    // Calcular gradientes
    const gradients = [...probabilities]
    gradients[targetDigit] -= 1 // dL/dS = p - y (donde y=1 para la clase correcta)

    // Actualizar pesos y sesgos
    for (let i = 0; i < 10; i++) {
      // Actualizar sesgo
      this.biases[i] -= this.learningRate * gradients[i]

      // Actualizar pesos
      for (let j = 0; j < input.length; j++) {
        // Solo actualizar si hay un valor de entrada (ahorra cálculos)
        if (input[j] !== 0) {
          this.weights[i][j] -= this.learningRate * gradients[i] * input[j]
        }
      }
    }

    return { loss }
  }

  // Establecer la tasa de aprendizaje
  setLearningRate(rate: number): void {
    this.learningRate = rate
  }

  // Reiniciar el modelo
  reset(): void {
    this.weights = Array(10)
      .fill(0)
      .map(() =>
        Array(784)
          .fill(0)
          .map(() => (Math.random() - 0.5) * 0.01),
      )
    this.biases = Array(10).fill(0)
  }
}

// Tipo para los ejemplos guardados
interface Example {
  imageData: number[]
  digit: number
}

export function ManualDigitTrainer() {
  const [activeTab, setActiveTab] = useState("draw")
  const [currentDigit, setCurrentDigit] = useState<number | null>(null)
  const [prediction, setPrediction] = useState<number | null>(null)
  const [confidence, setConfidence] = useState<number[]>([])
  const [examples, setExamples] = useState<Example[]>([])
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [trainingLoss, setTrainingLoss] = useState(0)
  const [trainingEpoch, setTrainingEpoch] = useState(0)
  const [isDrawing, setIsDrawing] = useState(false)
  const [learningRate, setLearningRate] = useState(0.1)
  const [modelInitialized, setModelInitialized] = useState(false)
  const [hasDrawing, setHasDrawing] = useState(false)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const modelRef = useRef<SimpleNeuralNetwork | null>(null)
  const isDrawingRef = useRef(false)

  // Inicializar el modelo
  useEffect(() => {
    modelRef.current = new SimpleNeuralNetwork()
    setModelInitialized(true)
    initializeCanvas()
  }, [])

  // Inicializar el canvas cuando cambia la pestaña
  useEffect(() => {
    initializeCanvas()
  }, [activeTab])

  // Función para inicializar el canvas
  const initializeCanvas = () => {
    const canvas = canvasRef.current
    if (canvas) {
      const ctx = canvas.getContext("2d")
      if (ctx) {
        // Cambiar el color de fondo a blanco
        ctx.fillStyle = "white"
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        // Usar un trazo más grueso (30 en lugar de 20)
        ctx.lineWidth = 30
        ctx.lineCap = "round"
        ctx.lineJoin = "round"
        // Cambiar el color del trazo a negro
        ctx.strokeStyle = "black"
      }
    }
    setHasDrawing(false)
  }

  // Funciones para dibujar en el canvas
  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    isDrawingRef.current = true
    setIsDrawing(true)
    setHasDrawing(true)

    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let x, y

    if ("touches" in e) {
      // Evento táctil
      const rect = canvas.getBoundingClientRect()
      x = e.touches[0].clientX - rect.left
      y = e.touches[0].clientY - rect.top
    } else {
      // Evento de ratón
      x = e.nativeEvent.offsetX
      y = e.nativeEvent.offsetY
    }

    ctx.beginPath()
    ctx.moveTo(x, y)
    ctx.lineTo(x, y)
    ctx.stroke()
  }

  const stopDrawing = () => {
    isDrawingRef.current = false
    setIsDrawing(false)
  }

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current) return

    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let x, y

    if ("touches" in e) {
      // Evento táctil
      const rect = canvas.getBoundingClientRect()
      x = e.touches[0].clientX - rect.left
      y = e.touches[0].clientY - rect.top
    } else {
      // Evento de ratón
      x = e.nativeEvent.offsetX
      y = e.nativeEvent.offsetY
    }

    ctx.lineTo(x, y)
    ctx.stroke()
  }

  // Función para limpiar el canvas
  const clearCanvas = () => {
    const canvas = canvasRef.current
    if (canvas) {
      const ctx = canvas.getContext("2d")
      if (ctx) {
        // Cambiar el color de fondo a blanco
        ctx.fillStyle = "white"
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        ctx.beginPath()
      }
    }
    setPrediction(null)
    setConfidence([])
    setHasDrawing(false)
  }

  // Función para obtener los datos de imagen del canvas
  const getImageData = (): number[] => {
    const canvas = canvasRef.current
    if (!canvas) return []

    // Crear canvas temporal de 28x28
    const tempCanvas = document.createElement("canvas")
    tempCanvas.width = 28
    tempCanvas.height = 28
    const tempCtx = tempCanvas.getContext("2d")
    if (!tempCtx) return []

    // Redimensionar imagen
    tempCtx.fillStyle = "white"
    tempCtx.fillRect(0, 0, 28, 28)
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28)

    // Obtener datos de imagen
    const imageData = tempCtx.getImageData(0, 0, 28, 28)
    const data = imageData.data

    // Convertir a array de valores normalizados (invertidos porque ahora dibujamos en negro sobre blanco)
    const result = []
    for (let i = 0; i < 784; i++) {
      // Invertir el valor: 0 = blanco (fondo), 1 = negro (trazo)
      // 255 - valor = invertir (255 - 255 = 0 para blanco, 255 - 0 = 255 para negro)
      result.push((255 - data[i * 4]) / 255)
    }

    return result
  }

  // Función para guardar un ejemplo
  const saveExample = () => {
    if (currentDigit === null) {
      alert("Por favor, selecciona primero un dígito")
      return
    }

    const imageData = getImageData()

    // Verificar que hay suficientes píxeles activos (reducido el umbral a 5)
    const activePixels = imageData.filter((val) => val > 0.1).length
    if (activePixels < 5) {
      alert("Por favor, dibuja un dígito más visible")
      return
    }

    // Guardar ejemplo
    setExamples((prev) => [...prev, { imageData, digit: currentDigit }])

    // Limpiar canvas para el siguiente ejemplo
    clearCanvas()

    // Mostrar mensaje de confirmación
    console.log(`Ejemplo guardado para el dígito ${currentDigit}`)
    alert(`Ejemplo guardado para el dígito ${currentDigit}`)
  }

  // Función para predecir el dígito
  const predictDigit = () => {
    if (!modelRef.current) return

    const imageData = getImageData()

    // Verificar que hay suficientes píxeles activos (reducido el umbral a 3)
    const activePixels = imageData.filter((val) => val > 0.1).length
    if (activePixels < 3) {
      alert("Por favor, dibuja un dígito más visible")
      return
    }

    // Realizar predicción
    const result = modelRef.current.predict(imageData)

    // Actualizar estados
    setPrediction(result.prediction)
    setConfidence(result.confidences)
  }

  // Función para entrenar el modelo con los ejemplos guardados
  const trainModel = async () => {
    if (!modelRef.current || examples.length === 0) return

    setIsTraining(true)
    setTrainingProgress(0)
    setTrainingEpoch(0)

    // Establecer tasa de aprendizaje
    modelRef.current.setLearningRate(learningRate)

    const epochs = 50
    const totalIterations = epochs * examples.length
    let currentIteration = 0

    try {
      for (let epoch = 0; epoch < epochs; epoch++) {
        setTrainingEpoch(epoch + 1)

        // Mezclar ejemplos para cada época
        const shuffledIndices = Array.from({ length: examples.length }, (_, i) => i)
        shuffleArray(shuffledIndices)

        let epochLoss = 0

        for (let i = 0; i < shuffledIndices.length; i++) {
          const idx = shuffledIndices[i]
          const example = examples[idx]

          // Entrenar con este ejemplo
          const { loss } = modelRef.current.train(example.imageData, example.digit)
          epochLoss += loss

          // Actualizar progreso
          currentIteration++
          setTrainingProgress((currentIteration / totalIterations) * 100)

          // Pequeña pausa para permitir actualizaciones de UI
          if (i % 5 === 0) {
            await new Promise((resolve) => setTimeout(resolve, 0))
          }
        }

        // Actualizar pérdida promedio
        setTrainingLoss(epochLoss / examples.length)

        // Pequeña pausa entre épocas
        await new Promise((resolve) => setTimeout(resolve, 10))
      }

      console.log("Entrenamiento completado con", examples.length, "ejemplos")
    } catch (error) {
      console.error("Error durante el entrenamiento:", error)
    } finally {
      setIsTraining(false)
    }
  }

  // Función para mezclar un array
  const shuffleArray = (array: number[]) => {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[array[i], array[j]] = [array[j], array[i]]
    }
  }

  // Función para reiniciar el modelo
  const resetModel = () => {
    if (!modelRef.current) return

    modelRef.current.reset()
    setPrediction(null)
    setConfidence([])
    setTrainingLoss(0)
    setTrainingEpoch(0)
  }

  // Función para eliminar todos los ejemplos
  const clearExamples = () => {
    setExamples([])
  }

  // Renderizar ejemplos guardados
  const renderExamples = () => {
    // Agrupar ejemplos por dígito
    const groupedExamples: { [key: number]: number } = {}
    examples.forEach((ex) => {
      groupedExamples[ex.digit] = (groupedExamples[ex.digit] || 0) + 1
    })

    return (
      <div className="grid grid-cols-5 gap-2 mb-4">
        {Array.from({ length: 10 }, (_, i) => i).map((digit) => (
          <div key={digit} className="text-center p-2 border rounded-md">
            <div className="text-lg font-bold">{digit}</div>
            <div className="text-sm">{groupedExamples[digit] || 0} ejemplos</div>
          </div>
        ))}
      </div>
    )
  }

  return (
    <Tabs defaultValue="draw" onValueChange={setActiveTab}>
      <TabsList className="grid w-full grid-cols-3">
        <TabsTrigger value="draw">Dibujar y Enseñar</TabsTrigger>
        <TabsTrigger value="train">Entrenar Modelo</TabsTrigger>
        <TabsTrigger value="test">Probar Modelo</TabsTrigger>
      </TabsList>

      <TabsContent value="draw">
        <Card>
          <CardHeader>
            <CardTitle>Dibujar Ejemplos</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="text-sm text-gray-500">
                Dibuja un dígito, selecciona qué número es, y guárdalo como ejemplo para entrenar el modelo.
              </div>

              <div className="flex justify-center">
                <div className="border rounded-md overflow-hidden">
                  <canvas
                    ref={canvasRef}
                    width={280}
                    height={280}
                    onMouseDown={startDrawing}
                    onMouseMove={draw}
                    onMouseUp={stopDrawing}
                    onMouseLeave={stopDrawing}
                    onTouchStart={startDrawing}
                    onTouchMove={draw}
                    onTouchEnd={stopDrawing}
                    className="touch-none cursor-crosshair"
                    style={{ touchAction: "none" }}
                  />
                </div>
              </div>

              <div className="flex space-x-2">
                <Button onClick={clearCanvas} variant="outline" className="flex-1">
                  <Trash2 className="mr-2 h-4 w-4" />
                  Limpiar
                </Button>
              </div>

              <div className="space-y-2">
                <div className="text-sm font-medium">Selecciona el dígito que dibujaste:</div>
                <div className="grid grid-cols-5 gap-2">
                  {Array.from({ length: 10 }, (_, i) => i).map((digit) => (
                    <Button
                      key={digit}
                      variant={currentDigit === digit ? "default" : "outline"}
                      onClick={() => setCurrentDigit(digit)}
                      className="h-12"
                    >
                      {digit}
                    </Button>
                  ))}
                </div>
              </div>

              <Button onClick={saveExample} disabled={currentDigit === null || !hasDrawing} className="w-full">
                <Save className="mr-2 h-4 w-4" />
                Guardar como Ejemplo
              </Button>

              <div className="pt-4 border-t">
                <div className="flex justify-between items-center mb-2">
                  <h3 className="text-lg font-medium">Ejemplos Guardados</h3>
                  <Button variant="outline" size="sm" onClick={clearExamples} disabled={examples.length === 0}>
                    <Trash2 className="mr-2 h-3 w-3" />
                    Borrar Todo
                  </Button>
                </div>

                {examples.length > 0 ? (
                  <>
                    {renderExamples()}
                    <div className="text-sm text-gray-500 mt-2">
                      Total: {examples.length} ejemplos. Ve a la pestaña "Entrenar Modelo" para entrenar con estos
                      ejemplos.
                    </div>
                  </>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No hay ejemplos guardados. Dibuja algunos dígitos y guárdalos.
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="train">
        <Card>
          <CardHeader>
            <CardTitle>Entrenar Modelo</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="text-sm text-gray-500">
                Entrena el modelo con los ejemplos que has guardado. Cuantos más ejemplos tengas de cada dígito, mejor
                será el modelo.
              </div>

              {examples.length > 0 ? (
                <>
                  {renderExamples()}

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <div className="text-sm font-medium">Tasa de Aprendizaje:</div>
                      <div className="text-sm">{learningRate.toFixed(3)}</div>
                    </div>
                    <input
                      type="range"
                      min="0.001"
                      max="0.5"
                      step="0.001"
                      value={learningRate}
                      onChange={(e) => setLearningRate(Number.parseFloat(e.target.value))}
                      className="w-full"
                      disabled={isTraining}
                    />
                    <div className="text-xs text-gray-500">
                      Valores más altos aprenden más rápido pero pueden ser menos precisos. Valores recomendados: 0.01 -
                      0.1
                    </div>
                  </div>

                  {isTraining && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Progreso: Época {trainingEpoch}/50</span>
                        <span>{Math.round(trainingProgress)}%</span>
                      </div>
                      <Progress value={trainingProgress} />
                      <div className="text-center">
                        <p className="text-sm text-gray-500">Pérdida</p>
                        <p className="text-lg font-medium">{trainingLoss.toFixed(4)}</p>
                      </div>
                    </div>
                  )}

                  <div className="flex space-x-2">
                    <Button onClick={trainModel} disabled={isTraining || examples.length === 0} className="flex-1">
                      {isTraining ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Entrenando...
                        </>
                      ) : (
                        <>
                          <Play className="mr-2 h-4 w-4" />
                          Iniciar Entrenamiento
                        </>
                      )}
                    </Button>
                    <Button onClick={resetModel} variant="outline" disabled={isTraining} className="flex-1">
                      <RefreshCw className="mr-2 h-4 w-4" />
                      Reiniciar Modelo
                    </Button>
                  </div>

                  {trainingEpoch > 0 && !isTraining && (
                    <div className="rounded-md bg-green-50 p-4">
                      <div className="flex">
                        <div className="flex-shrink-0">
                          <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                            <path
                              fillRule="evenodd"
                              d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                              clipRule="evenodd"
                            />
                          </svg>
                        </div>
                        <div className="ml-3">
                          <p className="text-sm font-medium text-green-800">
                            Entrenamiento completado con {examples.length} ejemplos. Ve a la pestaña "Probar Modelo"
                            para probar el modelo.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center py-10 text-gray-500">
                  <p>No hay ejemplos para entrenar. Ve a la pestaña "Dibujar y Enseñar" para crear ejemplos.</p>
                  <Button variant="outline" className="mt-4" onClick={() => setActiveTab("draw")}>
                    <Plus className="mr-2 h-4 w-4" />
                    Crear Ejemplos
                  </Button>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="test">
        <Card>
          <CardHeader>
            <CardTitle>Probar Modelo</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="text-sm text-gray-500">
                Dibuja un dígito y prueba si el modelo puede reconocerlo correctamente.
              </div>

              <div className="flex justify-center">
                <div className="border rounded-md overflow-hidden">
                  <canvas
                    ref={canvasRef}
                    width={280}
                    height={280}
                    onMouseDown={startDrawing}
                    onMouseMove={draw}
                    onMouseUp={stopDrawing}
                    onMouseLeave={stopDrawing}
                    onTouchStart={startDrawing}
                    onTouchMove={draw}
                    onTouchEnd={stopDrawing}
                    className="touch-none cursor-crosshair"
                    style={{ touchAction: "none" }}
                  />
                </div>
              </div>

              <div className="flex space-x-2">
                <Button onClick={clearCanvas} variant="outline" className="flex-1">
                  <Trash2 className="mr-2 h-4 w-4" />
                  Limpiar
                </Button>
                <Button
                  onClick={predictDigit}
                  disabled={!modelInitialized || trainingEpoch === 0 || !hasDrawing}
                  className="flex-1"
                >
                  Predecir
                </Button>
              </div>

              {prediction !== null && confidence.length > 0 && (
                <div className="mt-4">
                  <h4 className="text-center text-lg font-medium">Predicción: {prediction}</h4>
                  <div className="mt-2">
                    <p className="text-sm text-gray-500 mb-1">Confianza por dígito:</p>
                    <div className="grid grid-cols-5 gap-2">
                      {confidence.map((conf, idx) => {
                        // Asegurarse de que conf es un número válido
                        const validConf = isNaN(conf) || !isFinite(conf) ? 0 : conf
                        return (
                          <div key={idx} className="text-center">
                            <div className="text-xs text-gray-500">Dígito {idx}</div>
                            <div className="h-20 relative bg-gray-200 rounded-sm overflow-hidden">
                              <div
                                className={`absolute bottom-0 w-full ${
                                  idx === prediction ? "bg-green-500" : "bg-blue-500"
                                }`}
                                style={{ height: `${Math.max(validConf * 100, 1)}%` }}
                              ></div>
                            </div>
                            <div className="text-xs font-medium mt-1">
                              {isNaN(validConf) ? "0.0" : (validConf * 100).toFixed(1)}%
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </div>
              )}

              {trainingEpoch === 0 && (
                <div className="rounded-md bg-yellow-50 p-4">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                        <path
                          fillRule="evenodd"
                          d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <p className="text-sm font-medium text-yellow-800">
                        Primero debes entrenar el modelo antes de poder probarlo. Ve a la pestaña "Entrenar Modelo".
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  )
}

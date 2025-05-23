"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { NeuralNetwork } from "@/lib/neural-network"
import { MnistData } from "@/lib/mnist-data"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { Loader2, Brain, BarChart3, ImageIcon, Info, RefreshCw } from "lucide-react"

export function ImageRecognition() {
  const [isTraining, setIsTraining] = useState(false)
  const [isTrained, setIsTrained] = useState(false)
  const [progress, setProgress] = useState(0)
  const [epoch, setEpoch] = useState(0)
  const [accuracy, setAccuracy] = useState(0)
  const [loss, setLoss] = useState(0)
  const [trainingHistory, setTrainingHistory] = useState<Array<{ epoch: number; accuracy: number; loss: number }>>([])
  const [prediction, setPrediction] = useState<number | null>(null)
  const [confidence, setConfidence] = useState<number[]>([])
  const [activeTab, setActiveTab] = useState("train")
  const [dataLoaded, setDataLoaded] = useState(false)
  const [dataLoadError, setDataLoadError] = useState<string | null>(null)
  const [showDebugInfo, setShowDebugInfo] = useState(false)
  const [debugImages, setDebugImages] = useState<string[]>([])
  const [isPredicting, setIsPredicting] = useState(false)
  const [modelError, setModelError] = useState<string | null>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const modelRef = useRef<NeuralNetwork | null>(null)
  const dataRef = useRef<MnistData | null>(null)
  const isDrawing = useRef(false)

  // Inicializar el modelo y los datos
  useEffect(() => {
    initializeModel()
  }, [])

  const initializeModel = async () => {
    console.log("Inicializando modelo y datos MNIST...")

    try {
      // Limpiar estados previos
      setModelError(null)
      setDataLoadError(null)
      setDataLoaded(false)
      setIsTrained(false)

      // Crear instancia del modelo
      modelRef.current = new NeuralNetwork([784, 128, 64, 10])
      console.log("Modelo creado correctamente")

      // Cargar datos MNIST
      dataRef.current = new MnistData()
      await dataRef.current.load()

      setDataLoaded(true)
      console.log("Datos MNIST cargados correctamente")

      // Cargar algunas imágenes de ejemplo para depuración
      if (dataRef.current) {
        const { xs } = dataRef.current.nextTestBatch(5)
        const debugImgs = []

        for (let i = 0; i < 5; i++) {
          const canvas = document.createElement("canvas")
          canvas.width = 28
          canvas.height = 28
          const ctx = canvas.getContext("2d")
          if (ctx) {
            const imageData = ctx.createImageData(28, 28)
            for (let j = 0; j < 784; j++) {
              const value = Math.floor(xs[i][j] * 255)
              imageData.data[j * 4] = value
              imageData.data[j * 4 + 1] = value
              imageData.data[j * 4 + 2] = value
              imageData.data[j * 4 + 3] = 255
            }
            ctx.putImageData(imageData, 0, 0)
            debugImgs.push(canvas.toDataURL())
          }
        }
        setDebugImages(debugImgs)
      }
    } catch (error) {
      console.error("Error al inicializar:", error)
      setDataLoadError("Error al cargar los datos MNIST. Verifica tu conexión a Internet.")
      setModelError("Error al inicializar el modelo. Recarga la página e intenta de nuevo.")
    }
  }

  // Efecto para inicializar el canvas cuando se cambia a la pestaña de prueba
  useEffect(() => {
    if (activeTab === "test") {
      initializeCanvas()
    }
  }, [activeTab])

  // Función para inicializar el canvas
  const initializeCanvas = () => {
    const canvas = canvasRef.current
    if (canvas) {
      const ctx = canvas.getContext("2d")
      if (ctx) {
        ctx.fillStyle = "black"
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        ctx.lineWidth = 20
        ctx.lineCap = "round"
        ctx.lineJoin = "round"
        ctx.strokeStyle = "white"
      }
    }
  }

  // Función para entrenar el modelo
  const trainModel = async () => {
    if (!modelRef.current || !dataRef.current || !dataLoaded) {
      setModelError("No se puede entrenar: modelo o datos no disponibles")
      return
    }

    try {
      setIsTraining(true)
      setProgress(0)
      setEpoch(0)
      setTrainingHistory([])
      setModelError(null)

      console.log("Iniciando entrenamiento...")
      const totalEpochs = 20 // Aumentado para mejor entrenamiento
      const batchSize = 32 // Reducido para mejor estabilidad

      for (let i = 0; i < totalEpochs; i++) {
        setEpoch(i + 1)

        // Entrenar con múltiples lotes en cada época
        let epochLoss = 0
        const batchesPerEpoch = 10 // Más lotes por época
        let learningRate = 0.1 // Declare the learningRate variable here

        for (let j = 0; j < batchesPerEpoch; j++) {
          // Obtener lote de entrenamiento
          const { xs, ys } = dataRef.current.nextTrainBatch(batchSize)

          // Tasa de aprendizaje adaptativa (decae con el tiempo)
          learningRate = 0.1 * Math.pow(0.95, i)

          // Entrenar con este lote
          const { loss: currentLoss } = await modelRef.current.train(xs, ys, learningRate)
          epochLoss += currentLoss

          // Pequeña pausa para permitir actualizaciones de UI
          await new Promise((resolve) => setTimeout(resolve, 0))
        }

        // Calcular pérdida promedio para esta época
        const avgLoss = epochLoss / batchesPerEpoch

        // Evaluar en conjunto de validación cada 2 épocas
        let currentAccuracy = accuracy
        if (i % 2 === 0 || i === totalEpochs - 1) {
          const { xs: valXs, ys: valYs } = dataRef.current.nextTestBatch(200)
          currentAccuracy = await modelRef.current.evaluate(valXs, valYs)
        }

        console.log(
          `Época ${i + 1}/${totalEpochs}: Pérdida=${avgLoss.toFixed(4)}, Precisión=${(currentAccuracy * 100).toFixed(2)}%, LR=${learningRate.toFixed(4)}`,
        )

        // Actualizar estado
        setLoss(avgLoss)
        setAccuracy(currentAccuracy)
        setProgress(((i + 1) / totalEpochs) * 100)

        // Actualizar historial de entrenamiento
        setTrainingHistory((prev) => [...prev, { epoch: i + 1, accuracy: currentAccuracy, loss: avgLoss }])

        // Pequeña pausa para permitir actualizaciones de UI
        await new Promise((resolve) => setTimeout(resolve, 50))
      }

      setIsTrained(true)
      console.log("Entrenamiento completado. Modelo listo para predicciones.")
    } catch (error) {
      console.error("Error durante el entrenamiento:", error)
      setModelError("Error durante el entrenamiento. Intenta de nuevo.")
    } finally {
      setIsTraining(false)
    }
  }

  // Funciones para dibujar en el canvas
  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    isDrawing.current = true

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
    isDrawing.current = false
  }

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing.current) return

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
        ctx.fillStyle = "black"
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        ctx.beginPath() // Importante: reiniciar el path
      }
    }
    setPrediction(null)
    setConfidence([])
  }

  // Función simplificada para predecir el dígito dibujado
  const predictDigit = async () => {
    console.log("=== INICIANDO PREDICCIÓN ===")

    if (!modelRef.current) {
      console.log("❌ No hay modelo disponible")
      setModelError("Modelo no disponible. Reinicia la aplicación.")
      return
    }

    if (!isTrained) {
      console.log("❌ Modelo no entrenado")
      setModelError("Primero debes entrenar el modelo.")
      return
    }

    if (!canvasRef.current) {
      console.log("❌ Canvas no disponible")
      return
    }

    setIsPredicting(true)
    setModelError(null)

    try {
      const canvas = canvasRef.current
      const ctx = canvas.getContext("2d")
      if (!ctx) {
        console.log("❌ No se pudo obtener contexto del canvas")
        return
      }

      console.log("✅ Canvas obtenido, procesando imagen...")

      // Crear canvas temporal de 28x28
      const tempCanvas = document.createElement("canvas")
      tempCanvas.width = 28
      tempCanvas.height = 28
      const tempCtx = tempCanvas.getContext("2d")
      if (!tempCtx) {
        console.log("❌ No se pudo crear canvas temporal")
        return
      }

      // Redimensionar imagen
      tempCtx.fillStyle = "black"
      tempCtx.fillRect(0, 0, 28, 28)
      tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28)

      // Obtener datos de imagen
      const imageData = tempCtx.getImageData(0, 0, 28, 28)
      const data = imageData.data

      // Convertir a entrada del modelo
      const input = new Float32Array(784)
      for (let i = 0; i < 784; i++) {
        input[i] = data[i * 4] / 255 // Canal R normalizado
      }

      // Verificar que hay contenido
      const totalPixels = Array.from(input).reduce((sum, val) => sum + val, 0)
      console.log("📊 Total de píxeles activos:", totalPixels)

      if (totalPixels < 5) {
        console.log("⚠️ Imagen muy vacía, no se puede predecir")
        setModelError("Dibuja un dígito más visible")
        setPrediction(null)
        setConfidence([])
        return
      }

      console.log("🔮 Realizando predicción...")

      // Realizar predicción
      const result = await modelRef.current.predict(input)

      console.log("🎯 Resultado:", result)
      console.log("🏆 Predicción:", result.prediction)
      console.log("📈 Confianzas:", result.confidences)

      // Verificar si hay valores NaN en las confianzas
      const hasNaN = result.confidences.some((val) => isNaN(val) || !isFinite(val))
      if (hasNaN) {
        console.error("❌ Valores inválidos en confianzas")
        setModelError("Error en la predicción: valores inválidos. Intenta con otro dígito.")
        return
      }

      // Actualizar estados
      setPrediction(result.prediction)
      setConfidence(result.confidences)

      console.log("✅ Estados actualizados correctamente")
    } catch (error) {
      console.error("💥 Error en predicción:", error)
      setModelError("Error durante la predicción. Intenta con otro dígito.")
    } finally {
      setIsPredicting(false)
      console.log("=== FIN DE PREDICCIÓN ===")
    }
  }

  return (
    <Tabs defaultValue="train" onValueChange={setActiveTab}>
      <TabsList className="grid w-full grid-cols-3">
        <TabsTrigger value="train">
          <Brain className="mr-2 h-4 w-4" />
          Entrenar Modelo
        </TabsTrigger>
        <TabsTrigger value="stats">
          <BarChart3 className="mr-2 h-4 w-4" />
          Estadísticas
        </TabsTrigger>
        <TabsTrigger value="test">
          <ImageIcon className="mr-2 h-4 w-4" />
          Probar Modelo
        </TabsTrigger>
      </TabsList>

      {modelError && (
        <div className="rounded-md bg-red-50 p-4 my-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-red-800">{modelError}</p>
              <div className="mt-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setModelError(null)
                    initializeModel()
                  }}
                  className="flex items-center"
                >
                  <RefreshCw className="mr-2 h-3 w-3" />
                  Reiniciar modelo
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      <TabsContent value="train" className="space-y-4">
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Entrenamiento del Modelo</h3>
                <p className="text-sm text-gray-500">
                  Entrena un perceptrón multicapa para reconocer dígitos escritos a mano del conjunto de datos MNIST.
                </p>
              </div>

              {dataLoadError && (
                <div className="rounded-md bg-red-50 p-4">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                        <path
                          fillRule="evenodd"
                          d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414 1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <p className="text-sm font-medium text-red-800">{dataLoadError}</p>
                    </div>
                  </div>
                </div>
              )}

              {!dataLoaded && !dataLoadError && (
                <div className="flex items-center justify-center py-4">
                  <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
                  <span className="ml-2 text-gray-500">Cargando datos MNIST...</span>
                </div>
              )}

              {isTraining && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Progreso: Época {epoch}/20</span>
                    <span>{Math.round(progress)}%</span>
                  </div>
                  <Progress value={progress} />
                  <div className="grid grid-cols-2 gap-4 pt-2">
                    <div className="text-center">
                      <p className="text-sm text-gray-500">Precisión</p>
                      <p className="text-lg font-medium">{(accuracy * 100).toFixed(2)}%</p>
                    </div>
                    <div className="text-center">
                      <p className="text-sm text-gray-500">Pérdida</p>
                      <p className="text-lg font-medium">{loss.toFixed(4)}</p>
                    </div>
                  </div>
                </div>
              )}

              <Button onClick={trainModel} disabled={isTraining || !dataLoaded} className="w-full">
                {isTraining ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Entrenando...
                  </>
                ) : isTrained ? (
                  "Entrenar de nuevo"
                ) : (
                  "Iniciar entrenamiento"
                )}
              </Button>

              {isTrained && !isTraining && (
                <div className="rounded-md bg-green-50 p-4">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                        <path
                          fillRule="evenodd"
                          d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414 1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <p className="text-sm font-medium text-green-800">
                        ¡Entrenamiento completado! Precisión final: {(accuracy * 100).toFixed(2)}%
                      </p>
                    </div>
                  </div>
                </div>
              )}

              <div className="pt-2">
                <Button variant="outline" size="sm" className="w-full" onClick={() => setShowDebugInfo(!showDebugInfo)}>
                  <Info className="mr-2 h-4 w-4" />
                  {showDebugInfo ? "Ocultar información de depuración" : "Mostrar información de depuración"}
                </Button>
              </div>

              {showDebugInfo && debugImages.length > 0 && (
                <div className="mt-4 border rounded-md p-4">
                  <h4 className="text-sm font-medium mb-2">Ejemplos de dígitos MNIST (28x28 píxeles):</h4>
                  <div className="flex space-x-2 overflow-x-auto pb-2">
                    {debugImages.map((img, idx) => (
                      <div key={idx} className="border p-1 bg-white">
                        <img
                          src={img || "/placeholder.svg"}
                          alt={`Ejemplo ${idx}`}
                          className="w-16 h-16 object-contain"
                        />
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    Estos son ejemplos de dígitos del conjunto de datos MNIST. Tu modelo está entrenado para reconocer
                    imágenes en este formato.
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="stats">
        <Card>
          <CardContent className="pt-6 space-y-4">
            <h3 className="text-lg font-medium">Estadísticas de Entrenamiento</h3>

            {trainingHistory.length > 0 ? (
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainingHistory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" label={{ value: "Época", position: "insideBottom", offset: -5 }} />
                    <YAxis yAxisId="left" label={{ value: "Precisión", angle: -90, position: "insideLeft" }} />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      label={{ value: "Pérdida", angle: 90, position: "insideRight" }}
                    />
                    <Tooltip />
                    <Legend />
                    <Line yAxisId="left" type="monotone" dataKey="accuracy" stroke="#4f46e5" name="Precisión" />
                    <Line yAxisId="right" type="monotone" dataKey="loss" stroke="#ef4444" name="Pérdida" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="text-center py-10 text-gray-500">
                <p>Entrena el modelo para ver las estadísticas</p>
              </div>
            )}
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="test" className="space-y-4">
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Probar el Modelo</h3>
                <p className="text-sm text-gray-500">
                  Dibuja un dígito (0-9) en el canvas y el modelo intentará reconocerlo.
                </p>
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
                  Limpiar
                </Button>
                <Button onClick={predictDigit} disabled={!isTrained || isPredicting} className="flex-1">
                  {isPredicting ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Prediciendo...
                    </>
                  ) : (
                    "Predecir"
                  )}
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

              {!isTrained && (
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
                        Primero debes entrenar el modelo antes de poder probarlo.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {showDebugInfo && (
                <div className="mt-4 border rounded-md p-4">
                  <h4 className="text-sm font-medium mb-2">Consejos para mejores resultados:</h4>
                  <ul className="text-xs text-gray-600 list-disc pl-5 space-y-1">
                    <li>Dibuja dígitos grandes y centrados en el canvas</li>
                    <li>Usa trazos gruesos y claros</li>
                    <li>Intenta que tu dígito se parezca a los ejemplos de MNIST (arriba)</li>
                    <li>El modelo funciona mejor con dígitos simples y bien definidos</li>
                    <li>Si el modelo no reconoce tu dígito, intenta con otro estilo de escritura</li>
                    <li>Abre la consola del navegador (F12) para ver logs de depuración</li>
                  </ul>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  )
}

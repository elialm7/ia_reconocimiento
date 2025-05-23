// Clase para cargar y manejar el conjunto de datos MNIST
export class MnistData {
  private trainImages: Float32Array[] = []
  private trainLabels: Float32Array[] = []
  private testImages: Float32Array[] = []
  private testLabels: Float32Array[] = []
  private trainIndices: number[] = []
  private testIndices: number[] = []
  private NUM_CLASSES = 10
  private NUM_DATASET_ELEMENTS = 65000
  private NUM_TRAIN_ELEMENTS = 55000
  private MNIST_IMAGES_SPRITE_PATH = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png"
  private MNIST_LABELS_PATH = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8"

  constructor() {}

  // Cargar los datos MNIST
  async load(): Promise<void> {
    // Cargar imágenes
    const img = new Image()
    const canvas = document.createElement("canvas")
    const ctx = canvas.getContext("2d")

    const imgRequest = new Promise<void>((resolve, reject) => {
      img.crossOrigin = ""
      img.onload = () => {
        img.width = img.naturalWidth
        img.height = img.naturalHeight

        const datasetBytesBuffer = new ArrayBuffer(this.NUM_DATASET_ELEMENTS * 784 * 4)
        const datasetImages = new Float32Array(datasetBytesBuffer)

        canvas.width = img.width
        canvas.height = img.height
        ctx!.drawImage(img, 0, 0)

        const imgData = ctx!.getImageData(0, 0, canvas.width, canvas.height)

        // Convertir datos de imagen a Float32Array
        for (let i = 0; i < this.NUM_DATASET_ELEMENTS; i++) {
          const idx = i * 28 * 28
          for (let j = 0; j < 28 * 28; j++) {
            // Todos los canales tienen el mismo valor ya que es una imagen en escala de grises
            const pixelValue = imgData.data[(i * 28 * 28 + j) * 4 + 0] / 255
            datasetImages[idx + j] = pixelValue
          }
        }

        // Dividir en conjuntos de entrenamiento y prueba
        this.trainImages = []
        this.testImages = []

        for (let i = 0; i < this.NUM_TRAIN_ELEMENTS; i++) {
          const idx = i * 28 * 28
          const image = new Float32Array(784)
          for (let j = 0; j < 784; j++) {
            image[j] = datasetImages[idx + j]
          }
          this.trainImages.push(image)
        }

        for (let i = this.NUM_TRAIN_ELEMENTS; i < this.NUM_DATASET_ELEMENTS; i++) {
          const idx = i * 28 * 28
          const image = new Float32Array(784)
          for (let j = 0; j < 784; j++) {
            image[j] = datasetImages[idx + j]
          }
          this.testImages.push(image)
        }

        resolve()
      }

      img.onerror = (error) => {
        reject(error)
      }

      img.src = this.MNIST_IMAGES_SPRITE_PATH
    })

    // Cargar etiquetas
    const labelsRequest = fetch(this.MNIST_LABELS_PATH)
      .then((response) => response.arrayBuffer())
      .then((buffer) => {
        const labelsData = new Uint8Array(buffer)

        // Dividir en conjuntos de entrenamiento y prueba
        this.trainLabels = []
        this.testLabels = []

        for (let i = 0; i < this.NUM_TRAIN_ELEMENTS; i++) {
          const label = labelsData[i]
          const oneHot = new Float32Array(this.NUM_CLASSES)
          oneHot[label] = 1
          this.trainLabels.push(oneHot)
        }

        for (let i = this.NUM_TRAIN_ELEMENTS; i < this.NUM_DATASET_ELEMENTS; i++) {
          const label = labelsData[i]
          const oneHot = new Float32Array(this.NUM_CLASSES)
          oneHot[label] = 1
          this.testLabels.push(oneHot)
        }
      })

    // Esperar a que se carguen tanto las imágenes como las etiquetas
    await Promise.all([imgRequest, labelsRequest])

    // Inicializar índices para barajar los datos
    this.trainIndices = Array.from(Array(this.trainImages.length).keys())
    this.testIndices = Array.from(Array(this.testImages.length).keys())

    // Barajar los índices
    this.shuffle(this.trainIndices)
    this.shuffle(this.testIndices)
  }

  // Obtener el siguiente lote de entrenamiento
  nextTrainBatch(batchSize: number): { xs: Float32Array[]; ys: Float32Array[] } {
    return this.nextBatch(batchSize, this.trainImages, this.trainLabels, this.trainIndices)
  }

  // Obtener el siguiente lote de prueba
  nextTestBatch(batchSize: number): { xs: Float32Array[]; ys: Float32Array[] } {
    return this.nextBatch(batchSize, this.testImages, this.testLabels, this.testIndices)
  }

  // Obtener un lote de datos
  private nextBatch(
    batchSize: number,
    images: Float32Array[],
    labels: Float32Array[],
    indices: number[],
  ): { xs: Float32Array[]; ys: Float32Array[] } {
    const xs: Float32Array[] = []
    const ys: Float32Array[] = []

    for (let i = 0; i < batchSize; i++) {
      const idx = indices[i % indices.length]
      xs.push(images[idx])
      ys.push(labels[idx])
    }

    // Rotar los índices para el siguiente lote
    this.shuffle(indices)

    return { xs, ys }
  }

  // Barajar un array (algoritmo de Fisher-Yates)
  private shuffle(array: number[]): void {
    let currentIndex = array.length
    let temporaryValue, randomIndex

    // Mientras queden elementos a barajar
    while (0 !== currentIndex) {
      // Seleccionar un elemento restante
      randomIndex = Math.floor(Math.random() * currentIndex)
      currentIndex -= 1

      // Intercambiar con el elemento actual
      temporaryValue = array[currentIndex]
      array[currentIndex] = array[randomIndex]
      array[randomIndex] = temporaryValue
    }
  }
}

import { ManualDigitTrainer } from "@/components/manual-digit-trainer"

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <header className="border-b bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-gray-800">Reconocimiento de Dígitos con Entrenamiento Manual - Inteligencia Artificial II </h1>
          <p className="text-sm text-gray-600">Entrena tu propio modelo de reconocimiento de dígitos</p>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-4">Entrenamiento Manual de Reconocimiento de Dígitos</h2>
            <p className="text-lg text-gray-600">
              Esta aplicación te permite entrenar un modelo de reconocimiento de dígitos dibujando tus propios ejemplos.
              Dibuja dígitos, etiquétalos, y entrena el modelo para ver cómo aprende a reconocerlos.
            </p>
            <p>
             <b>Este Proyecto fue desarrollado por alumnos de la Facultad Politecnica - 7mo semestre</b>
            </p>
             <ul className="list-disc pl-5 space-y-2 text-gray-600">
              <li>Rodolfo Elias Ojeda Almada</li>
              <li>Derlis Diaz</li>
              <li>Allam Diaz</li>
              <li>Victor Montiel</li>
              <li>Claudio Portillo</li>
            </ul>
          </div>

          <div className="bg-white rounded-xl shadow-lg overflow-hidden border p-6">
            <ManualDigitTrainer />
          </div>

          <div className="mt-8 bg-white rounded-xl shadow-lg overflow-hidden border p-6">
            <h3 className="text-xl font-bold text-gray-800 mb-4">Cómo Funciona</h3>
            <p className="text-gray-600 mb-4">
              Este proyecto implementa un clasificador lineal simple que aprende a reconocer dígitos escritos a mano:
            </p>
            <ul className="list-disc pl-5 space-y-2 text-gray-600">
              <li>Dibuja dígitos y etiquétalos para crear ejemplos de entrenamiento</li>
              <li>Entrena el modelo con tus ejemplos</li>
              <li>Prueba el modelo dibujando nuevos dígitos</li>
              <li>Mejora el modelo añadiendo más ejemplos de los dígitos que no reconoce bien</li>
              <li>Experimenta con diferentes tasas de aprendizaje</li>
            </ul>
            <p className="text-gray-600 mt-4">
              Cuantos más ejemplos proporciones de cada dígito, mejor será el modelo. Intenta proporcionar al menos 5-10
              ejemplos de cada dígito para obtener buenos resultados.
            </p>
          </div>
        </div>
      </main>

      <footer className="bg-gray-800 text-white py-6 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p>© 2025 Proyecto de Inteligencia Artificial</p>
        </div>
      </footer>
    </div>
  )
}
